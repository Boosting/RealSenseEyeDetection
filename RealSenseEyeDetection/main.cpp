// License: Apache 2.0. See LICENSE file in root directory.
// Copyright(c) 2015 Intel Corporation. All Rights Reserved.

/////////////////////////////////////////////////////
// librealsense tutorial #1 - Accessing depth data //
/////////////////////////////////////////////////////

// Include librealsense and OpenCV header files
#include <librealsense/rs.hpp>
#include <opencv2/core.hpp>			// Mat is in here
#include <opencv2/highgui.hpp>		// imshow is in here

#include <cstdio>
#include "EyeDetector.h"

using namespace cv;
using namespace std;

int main() try
{
	EyeDetector myEyeDetector;

	Mat rgb_img(Size(640, 480), CV_8UC3);
	Mat depth_to_color_img(Size(640, 480), CV_16UC1);
	cvNamedWindow("Color Image", WINDOW_AUTOSIZE);
	cvNamedWindow("DTC Image", WINDOW_AUTOSIZE);

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

	// Determine depth value corresponding to one meter
	const uint16_t one_meter = static_cast<uint16_t>(1.0f / dev->get_depth_scale());

	while (true)
	{
		// This call waits until a new coherent set of frames is available on a device
		// Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called
		dev->wait_for_frames();

		// Retrieve depth data, which was previously configured as a 640 x 480 image of 16-bit depth values
		// const uint16_t * depth_frame = reinterpret_cast<const uint16_t*>(dev->get_frame_data(rs::stream::depth));
		// memcpy(depth_img.data, depth_frame, depth_img.cols*depth_img.rows * sizeof(uint16_t));
		const uint8_t * rgb_frame = reinterpret_cast<const uint8_t*>(dev->get_frame_data(rs::stream::color));
		memcpy(rgb_img.data, rgb_frame, rgb_img.cols*rgb_img.rows * sizeof(uint8_t)*rgb_img.channels());
		const uint16_t * depth_to_color_frame = reinterpret_cast<const uint16_t*>(dev->get_frame_data(rs::stream::depth_aligned_to_color));
		memcpy(depth_to_color_img.data, depth_to_color_frame, depth_to_color_img.cols*depth_to_color_img.rows * sizeof(uint16_t));

		myEyeDetector.ImageProcessAndDetect(rgb_img, depth_to_color_img, one_meter);
		myEyeDetector.DrawFacesAndEyes(rgb_img);

		Mat dstImage(depth_to_color_img.size(), CV_8UC1);
		for (int i = 0; i < dstImage.rows; ++i) {
			for (int j = 0; j < dstImage.cols; ++j) {
				uint16_t val = depth_to_color_img.at<uint16_t>(i, j);
				double val2;
				if (val ==0) {
					dstImage.at<uint8_t>(i, j) = (uint8_t)val;
				}
				else {
					val2 = (double)val * (-51.0) / one_meter + 255.0;
					dstImage.at<uint8_t>(i, j) = (uint8_t)val2;
				}
			}
		}

		imshow("DTC Image", dstImage);
		dstImage.release();

		imshow("Color Image", rgb_img);
		waitKey(30);
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
