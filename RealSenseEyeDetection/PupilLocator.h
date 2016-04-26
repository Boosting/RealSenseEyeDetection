#pragma once
#include <iostream>
#include <opencv2/core.hpp>			// Mat is in here
#include <opencv2/imgproc.hpp>		// circle is in here
#include <opencv2/highgui.hpp>		// imshow is in here

using namespace cv;
using namespace std;

class PupilLocator
{
private:
	const Scalar colors[8] = { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };
	const double scale = 1.0;

	vector<Rect> detectedEyes;

public:
	PupilLocator();
	~PupilLocator();

	void AddEye(Rect detectedEye);
	void DrawEyes(Mat& depthImg);
	void DetectPupils(Mat& depthImg);
	void DrawPupils(Mat& depthImg);
	void ClearInfo();
};

