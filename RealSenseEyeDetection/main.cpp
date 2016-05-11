

#pragma once
//#define MY_WINDOWS_SOCKET
#define _WINSOCK_DEPRECATED_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS

#include "EyeDetector.h"
#include "PupilLocator.h"

#include <iostream>
#include <iomanip>

// Include librealsense and OpenCV header files
#include <librealsense/rs.hpp>
#include <opencv2/core.hpp>			// cv::Mat is in here
#include <opencv2/highgui.hpp>		// imshow is in here

#ifdef MY_WINDOWS_SOCKET
#include <winsock2.h>
#pragma comment(lib,"ws2_32.lib")
#define PORT 4000
#define IP_ADDRESS "169.237.118.42"
#else
#define MAX_PATH 260
#endif

using namespace std;

// Compute convert matrix (lookup table) saving coordinates from color to depth.
// i.e. A pixel in color[200,300] matches depth[250,350], then convert_imgX[200,300] = 250, convert_imgY[200,300] = 350.
// @param dstX : convert matrix X
// @param dstY : convert matrix Y
// @param depth_frame
// @param scale : depth scale from dev->get_depth_scale();
// @param depth_intrin
// @param depth_to_color_extrin
// @param color_intrin
void ComputeLookupTable(cv::Mat& dstX, cv::Mat& dstY, const uint16_t * depth_frame, float scale, rs::intrinsics& depth_intrin, rs::extrinsics& depth_to_color_extrin, rs::intrinsics& color_intrin);

int main() try
{
#ifdef MY_WINDOWS_SOCKET
/////////////////////////////////////////////////////////////
////////////  Windows socket things /////////////////////////
/////////////////////////////////////////////////////////////
	WSADATA Ws;
	SOCKET ClientSocket;
	struct sockaddr_in ServerAddr;
	int Ret = 0;

	//Init Windows Socket
	if (WSAStartup(MAKEWORD(2, 2), &Ws) != 0)
	{
		cout << "Init Windows Socket Failed::" << GetLastError() << endl;
		return -1;
	}
	//Create Socket
	ClientSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
	if (ClientSocket == INVALID_SOCKET)
	{
		cout << "Create Socket Failed::" << GetLastError() << endl;
		return -1;
	}

	ServerAddr.sin_family = AF_INET;
	ServerAddr.sin_addr.s_addr = inet_addr(IP_ADDRESS);
	ServerAddr.sin_port = htons(PORT);
	memset(ServerAddr.sin_zero, 0x00, 8);

	Ret = connect(ClientSocket, (struct sockaddr*)&ServerAddr, sizeof(ServerAddr));
	if (Ret == SOCKET_ERROR)
	{
		cout << "Connect Error::" << GetLastError() << endl;
		return -1;
	}
	else
	{
		cout << "Connected!" << endl;
	}
/////////////////////////////////////////////////////////////
#endif // MY_WINDOWS_SOCKET

	EyeDetector myEyeDetector;
	PupilLocator myPupilLocator;

	cv::Mat rgb_img(cv::Size(640, 480), CV_8UC3);
	cv::Mat depth_to_color_img(cv::Size(640, 480), CV_16UC1);
	cv::Mat depth_img(cv::Size(640, 480), CV_16UC1);

	vector<rs::float3> pupilLocations;

	cvNamedWindow("Color Image", cv::WINDOW_AUTOSIZE);
	cvNamedWindow("DTC Image", cv::WINDOW_AUTOSIZE);
	cvNamedWindow("Depth Image", cv::WINDOW_AUTOSIZE);

	// Create a context object. This object owns the handles to all connected realsense devices.
	rs::context ctx;
	printf("There are %d connected RealSense devices.\n", ctx.get_device_count());
	if (ctx.get_device_count() == 0) return EXIT_FAILURE;

	// This tutorial will access only a single device, but it is trivial to extend to multiple devices.
	rs::device * dev = ctx.get_device(0);
	printf("\nUsing device 0, an %s\n", dev->get_name());
	printf("    Serial number: %s\n", dev->get_serial());
	printf("    Firmware version: %s\n", dev->get_firmware_version());

	// Configure all streams to run at VGA resolution at 30 frames per second.
	dev->enable_stream(rs::stream::depth, 640, 480, rs::format::z16, 30);
	dev->enable_stream(rs::stream::color, 640, 480, rs::format::bgr8, 30);
	dev->start();

	while (true)
	{
		char SendBuffer[MAX_PATH];
		double t = (double)cvGetTickCount();

		// A convert matrix (lookup table) saving coordinates from color to depth.
		// i.e. A pixel in color[200,300] matches depth[250,350], then convert_imgX[200,300] = 250, convert_imgY[200,300] = 350.
		cv::Mat convert_imgX = cv::Mat::zeros(cv::Size(640, 480), CV_16UC1);
		cv::Mat convert_imgY = cv::Mat::zeros(cv::Size(640, 480), CV_16UC1);

		// This call waits until a new coherent set of frames is available on a device.
		// Calls to get_frame_data(...) and get_frame_timestamp(...) on a device will return stable values until wait_for_frames(...) is called.
		dev->wait_for_frames();

		// Determine depth value corresponding to one meter.
		float scale = dev->get_depth_scale();
		uint16_t one_meter = static_cast<uint16_t>(1.0f / scale);

		// Retrieve camera parameters for mapping between depth and color.
		rs::intrinsics depth_intrin = dev->get_stream_intrinsics(rs::stream::depth);
		rs::extrinsics depth_to_color_extrin = dev->get_extrinsics(rs::stream::depth, rs::stream::color);
		rs::intrinsics color_intrin = dev->get_stream_intrinsics(rs::stream::color);

		// Retrieve depth data, color data, and depth_to_color data.
		const uint16_t * depth_frame = reinterpret_cast<const uint16_t*>(dev->get_frame_data(rs::stream::depth));
		memcpy(depth_img.data, depth_frame, depth_img.cols*depth_img.rows * sizeof(uint16_t));
		const uint8_t * rgb_frame = reinterpret_cast<const uint8_t*>(dev->get_frame_data(rs::stream::color));
		memcpy(rgb_img.data, rgb_frame, rgb_img.cols*rgb_img.rows * sizeof(uint8_t)*rgb_img.channels());
		const uint16_t * depth_to_color_frame = reinterpret_cast<const uint16_t*>(dev->get_frame_data(rs::stream::depth_aligned_to_color));
		memcpy(depth_to_color_img.data, depth_to_color_frame, depth_to_color_img.cols*depth_to_color_img.rows * sizeof(uint16_t));

		// Detect & Draw faces, eyes and pupils in color image.
		myEyeDetector.ImageProcessAndDetect(rgb_img, depth_to_color_img, one_meter);
		myEyeDetector.DrawDetectedInfo(rgb_img);
		myEyeDetector.DrawDetectedInfo(depth_to_color_img);

		// Convert eye position from color image to depth image.
		size_t numFaces = myEyeDetector.getFacesSize();
		size_t numPupils = myEyeDetector.getPupilsSize();
		
		if (numPupils != 0 && numFaces * 2 == numPupils) {
			// Build lookup table from depth image to color image.
			ComputeLookupTable(convert_imgX, convert_imgY, depth_frame, scale, depth_intrin, depth_to_color_extrin, color_intrin);

			// Add detected pupils from color image to myPupilLocator.
			for (int i = 0; i < numPupils; ++i) {
				cv::Point curPupil = myEyeDetector.getPupilLoc(i);
				myPupilLocator.AddPupil(curPupil);
			}

			if (!myPupilLocator.FindPupilDepthPoints(convert_imgX, convert_imgY)) {
				cout << "Couldn't locate pupil in depth image!" << endl;
			}
			
			myPupilLocator.DrawPupils(depth_img);

			// Save all detected pupils to a vector named pupilLocations.
			size_t numDepthPupils = myPupilLocator.getPupilDepthSize();
			for (int i = 0; i < numDepthPupils; ++i) {
				cv::Point pointD1 = myPupilLocator.getPupilReferenceLoc(i * 2);
				uint16_t depth_value = depth_frame[pointD1.y * depth_intrin.width + pointD1.x];
				float depth_in_meters = depth_value * scale;
				rs::float2 depth_pixel = { (float)pointD1.x, (float)pointD1.y };
				rs::float3 depth_point1 = depth_intrin.deproject(depth_pixel, depth_in_meters);

				cv::Point pointD2 = myPupilLocator.getPupilReferenceLoc(i * 2 + 1);
				depth_value = depth_frame[pointD2.y * depth_intrin.width + pointD2.x];
				depth_in_meters = depth_value * scale;
				depth_pixel = { (float)pointD2.x, (float)pointD2.y };
				rs::float3 depth_point2 = depth_intrin.deproject(depth_pixel, depth_in_meters);

				rs::float3 depth_point;
				depth_point.x = (depth_point1.x + depth_point2.x) / 2.;
				depth_point.y = (depth_point1.y + depth_point2.y) / 2.;
				depth_point.z = (depth_point1.z + depth_point2.z) / 2.;

				pupilLocations.push_back(depth_point);
			}
		}
		
		// Show images.
		cv::Mat dst_dtc_Image(depth_to_color_img.size(), CV_8UC1);
		for (int i = 0; i < dst_dtc_Image.rows; ++i) {
			for (int j = 0; j < dst_dtc_Image.cols; ++j) {
				uint16_t val = depth_to_color_img.at<uint16_t>(i, j);
				double val2;

				// Assign pixels with a depth value of zero, which is used to indicate no data.
				if (val == 0) {
					dst_dtc_Image.at<uint8_t>(i, j) = (uint8_t)val;
				}
				else {
					val2 = (double)val * (-51.0) / one_meter + 255.0;
					dst_dtc_Image.at<uint8_t>(i, j) = (uint8_t)val2;
				}
			}
		}

		cv::Mat dst_depth_Image(depth_img.size(), CV_8UC1);
		for (int i = 0; i < dst_depth_Image.rows; ++i) {
			for (int j = 0; j < dst_depth_Image.cols; ++j) {
				uint16_t val = depth_img.at<uint16_t>(i, j);
				double val2;

				// Assign pixels with a depth value of zero, which is used to indicate no data.
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
		cv::waitKey(1);

		t = (double)cvGetTickCount() - t;
		printf("numFaces = %d, detection time = %g ms\n", myEyeDetector.getFacesSize(), t / ((double)cvGetTickFrequency()*1000.));

		// Print out pupil 3D locations to screen and send them through socket.
		int pupilSize = pupilLocations.size();
		SendBuffer[0] = (char)pupilSize;
		for (int i = 0; i < pupilSize; ++i) {
			cout << right << setw(10) << setprecision(4) << pupilLocations[i].x;
			cout << right << setw(10) << setprecision(4) << pupilLocations[i].y;
			cout << right << setw(10) << setprecision(4) << pupilLocations[i].z;
			cout << endl;
			memcpy(SendBuffer + 1 + i * 12, &pupilLocations[i].x, 12);
		}
		SendBuffer[1 + pupilSize * 12] = '\0';

#ifdef MY_WINDOWS_SOCKET
		if (pupilSize != 0 && numFaces * 2 == pupilSize) {
			Ret = send(ClientSocket, SendBuffer, (int)strlen(SendBuffer), 0);
			if (Ret == SOCKET_ERROR) cout << "Send Info Error::" << GetLastError() << endl;
		}
#endif

		dst_dtc_Image.release();
		dst_depth_Image.release();

		myPupilLocator.ClearInfo();
		myEyeDetector.ClearInfo();
		pupilLocations.clear();
	}

#ifdef MY_WINDOWS_SOCKET
	// close sockets
	closesocket(ClientSocket);
	WSACleanup();
#endif

	return EXIT_SUCCESS;
}
catch (const rs::error & e)
{
	// Method calls against librealsense objects may throw exceptions of type rs::error
	printf("rs::error was thrown when calling %s(%s):\n", e.get_failed_function().c_str(), e.get_failed_args().c_str());
	printf("    %s\n", e.what());
	return EXIT_FAILURE;
}

void ComputeLookupTable(cv::Mat& dstX, cv::Mat& dstY, const uint16_t * depth_frame, float scale, rs::intrinsics& depth_intrin, rs::extrinsics& depth_to_color_extrin, rs::intrinsics& color_intrin) {
	for (int dy = 0; dy<depth_intrin.height; ++dy)
	{
		for (int dx = 0; dx<depth_intrin.width; ++dx)
		{
			// Retrieve the 16-bit depth value and map it into a depth in meters.
			uint16_t depth_value = depth_frame[dy * depth_intrin.width + dx];
			// Skip over pixels with a depth value of zero, which is used to indicate no data.
			if (depth_value == 0) continue;
			float depth_in_meters = depth_value * scale;

			// Map from pixel coordinates in the depth image to pixel coordinates in the color image.
			rs::float2 depth_pixel = { (float)dx, (float)dy };
			rs::float3 depth_point = depth_intrin.deproject(depth_pixel, depth_in_meters);
			rs::float3 color_point = depth_to_color_extrin.transform(depth_point);
			rs::float2 color_pixel = color_intrin.project(color_point);

			// Use the color from the nearest color pixel, or pure white if this point falls outside the color image.
			const int cx = (int)round(color_pixel.x), cy = (int)round(color_pixel.y);
			if (cx < 0 || cy < 0 || cx >= color_intrin.width || cy >= color_intrin.height) continue;

			dstX.at<uint16_t>(cy, cx) = (uint16_t)dx;
			dstY.at<uint16_t>(cy, cx) = (uint16_t)dy;
		}
	}
}