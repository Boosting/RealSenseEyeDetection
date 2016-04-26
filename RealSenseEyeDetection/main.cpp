// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

/////////////////////////////////////////////////////
// librealsense tutorial #1 - Accessing depth data //
/////////////////////////////////////////////////////

#pragma once
#include "EyeDetector.h"
#include "PupilLocator.h"

// Include librealsense and OpenCV header files
#include <librealsense/rs.hpp>
#include <opencv2/core.hpp>			// Mat is in here
#include <opencv2/highgui.hpp>		// imshow is in here
#include <cstdio>
#include <map>

#define DEPTH_PIXEL_SEARCH_RANGE (15)

using namespace cv;
using namespace std;

vector<Point> FindDepthPoint(vector<Point>& srcPts, const uint16_t * depth_frame, float scale, rs::intrinsics& depth_intrin, rs::extrinsics& depth_to_color_extrin, rs::intrinsics& color_intrin);

int main() try
{
	EyeDetector myEyeDetector;
	PupilLocator myPupilLocator;

	Mat rgb_img(Size(640, 480), CV_8UC3);
	Mat depth_to_color_img(Size(640, 480), CV_16UC1);
	Mat depth_img(Size(640, 480), CV_16UC1);
	cvNamedWindow("Color Image", WINDOW_AUTOSIZE);
	cvNamedWindow("DTC Image", WINDOW_AUTOSIZE);
	cvNamedWindow("Depth Image", WINDOW_AUTOSIZE);

	// Create a context object. This object owns the handles to all connected realsense devices.
	rs::context ctx;
	printf("There are %d connected RealSense devices.\n", ctx.get_device_count());
	if (ctx.get_device_count() == 0) return EXIT_FAILURE;

	// This tutorial will access only a single device, but it is trivial to extend to multiple devices
	rs::device * dev = ctx.get_device(0);
	printf("\nUsing device 0, an %s\n", dev->get_name());
	printf("    Serial number: %s\n", dev->get_serial());
	printf("    Firmware version: %s\n", dev->get_firmware_version());

	// Configure all streams to run at VGA resolution at 30 frames per second
	dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 30);
	dev->enable_stream(rs::stream::color, 640, 480, rs::format::bgr8, 30);
	dev->start();

	while (true)
	{
		// This call waits until a new coherent set of frames is available on a device
		// Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
		dev->wait_for_frames();

		// Determine depth value corresponding to one meter
		float scale = dev->get_depth_scale();
		uint16_t one_meter = static_cast<uint16_t>(1.0f / scale);

		// Retrieve camera parameters for mapping between depth and color
		rs::intrinsics depth_intrin = dev->get_stream_intrinsics(rs::stream::depth);
		rs::extrinsics depth_to_color_extrin = dev->get_extrinsics(rs::stream::depth, rs::stream::color);
		rs::intrinsics color_intrin = dev->get_stream_intrinsics(rs::stream::color);

		// Retrieve depth data, which was previously configured as a 640 x 480 image of 16-bit depth values
		const uint16_t * depth_frame = reinterpret_cast<const uint16_t*>(dev->get_frame_data(rs::stream::depth));
		memcpy(depth_img.data, depth_frame, depth_img.cols*depth_img.rows * sizeof(uint16_t));
		const uint8_t * rgb_frame = reinterpret_cast<const uint8_t*>(dev->get_frame_data(rs::stream::color));
		memcpy(rgb_img.data, rgb_frame, rgb_img.cols*rgb_img.rows * sizeof(uint8_t)*rgb_img.channels());
		const uint16_t * depth_to_color_frame = reinterpret_cast<const uint16_t*>(dev->get_frame_data(rs::stream::depth_aligned_to_color));
		memcpy(depth_to_color_img.data, depth_to_color_frame, depth_to_color_img.cols*depth_to_color_img.rows * sizeof(uint16_t));

		// Detect / Draw faces and eyes in color image
		myEyeDetector.ImageProcessAndDetect(rgb_img, depth_to_color_img, one_meter);
		myEyeDetector.DrawFacesAndEyes(rgb_img);
		myEyeDetector.DrawFacesAndEyes(depth_to_color_img);

		// Convert eye position from color image to depth image,
		size_t numEyes = myEyeDetector.getEyesSize();

		if (numEyes != 0) {
			vector<Point> colorEyePoints;

			for (int i = 0; i < numEyes; ++i) {
				Rect curEye = myEyeDetector.getEyeLoc(i);
				colorEyePoints.push_back(Point(curEye.x, curEye.y));
				colorEyePoints.push_back(Point(curEye.x + curEye.width, curEye.y + curEye.height));
			}

			vector<Point> depthEyePoints = FindDepthPoint(colorEyePoints, depth_frame, scale, depth_intrin, depth_to_color_extrin, color_intrin);

			for (int i = 0; i < numEyes; ++i) {
				myPupilLocator.AddEye(cvRect(depthEyePoints[i * 2].x, depthEyePoints[i * 2].y, depthEyePoints[i * 2 + 1].x - depthEyePoints[i * 2].x, depthEyePoints[i * 2 + 1].y - depthEyePoints[i * 2].y));
			}

			// Detect / Draw pupils in depth image
			myPupilLocator.DrawEyes(depth_img);
			myPupilLocator.DetectPupils(depth_img);
			myPupilLocator.DrawPupils(depth_img);
		}

		// Show images
		Mat dst_dtc_Image(depth_to_color_img.size(), CV_8UC1);
		for (int i = 0; i < dst_dtc_Image.rows; ++i) {
			for (int j = 0; j < dst_dtc_Image.cols; ++j) {
				uint16_t val = depth_to_color_img.at<uint16_t>(i, j);
				double val2;

				// Assign pixels with a depth value of zero, which is used to indicate no data
				if (val == 0) {
					dst_dtc_Image.at<uint8_t>(i, j) = (uint8_t)val;
				}
				else {
					val2 = (double)val * (-51.0) / one_meter + 255.0;
					dst_dtc_Image.at<uint8_t>(i, j) = (uint8_t)val2;
				}
			}
		}

		Mat dst_depth_Image(depth_img.size(), CV_8UC1);
		for (int i = 0; i < dst_depth_Image.rows; ++i) {
			for (int j = 0; j < dst_depth_Image.cols; ++j) {
				uint16_t val = depth_img.at<uint16_t>(i, j);
				double val2;

				// Assign pixels with a depth value of zero, which is used to indicate no data
				if (val == 0) {
					dst_depth_Image.at<uint8_t>(i, j) = (uint8_t)val;
				}
				else {
					val2 = (double)val * (-51.0) / one_meter + 255.0;
					dst_depth_Image.at<uint8_t>(i, j) = (uint8_t)val2;
				}
			}
		}

		imshow("Depth Image", dst_depth_Image);
		imshow("DTC Image", dst_dtc_Image);
		imshow("Color Image", rgb_img);

		waitKey(30);

		dst_dtc_Image.release();
		dst_depth_Image.release();

		myPupilLocator.ClearInfo();
		myEyeDetector.ClearInfo();
	}

	return EXIT_SUCCESS;
}
catch (const rs::error & e)
{
	// Method calls against librealsense objects may throw exceptions of type rs::error
	printf("rs::error was thrown when calling %s(%s):\n", e.get_failed_function().c_str(), e.get_failed_args().c_str());
	printf("    %s\n", e.what());
	return EXIT_FAILURE;
}


vector<Point> FindDepthPoint(vector<Point>& srcPts, const uint16_t * depth_frame, float scale, rs::intrinsics& depth_intrin, rs::extrinsics& depth_to_color_extrin, rs::intrinsics& color_intrin) {
	size_t size = srcPts.size();
	vector<int> distList(size, INT_MAX);
	vector<Point> resultVec(size, Point(0, 0));

	for (int dy = 0; dy<depth_intrin.height; ++dy)
	{
		for (int dx = 0; dx<depth_intrin.width; ++dx)
		{
			// Retrieve the 16-bit depth value and map it into a depth in meters
			uint16_t depth_value = depth_frame[dy * depth_intrin.width + dx];
			// Skip over pixels with a depth value of zero, which is used to indicate no data
			if (depth_value == 0) continue;
			float depth_in_meters = depth_value * scale;

			// Map from pixel coordinates in the depth image to pixel coordinates in the color image
			rs::float2 depth_pixel = { (float)dx, (float)dy };
			rs::float3 depth_point = depth_intrin.deproject(depth_pixel, depth_in_meters);
			rs::float3 color_point = depth_to_color_extrin.transform(depth_point);
			rs::float2 color_pixel = color_intrin.project(color_point);

			// Use the color from the nearest color pixel, or pure white if this point falls outside the color image
			const int cx = (int)round(color_pixel.x), cy = (int)round(color_pixel.y);
			if (cx < 0 || cy < 0 || cx >= color_intrin.width || cy >= color_intrin.height) continue;
			
			for (int i = 0; i < size; ++i) {
				int dist = (int)(norm(srcPts[i] - Point(cx, cy)) * 10.);
				if (dist < distList[i]) {
					distList[i] = dist;
					resultVec[i] = Point((int)round(dx), (int)round(dy));
				}
			}
		}
	}

	return resultVec;
}