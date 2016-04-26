#pragma once
#include <iostream>
#include <opencv2/core.hpp>			// Mat is in here
#include <opencv2/imgproc.hpp>		// circle is in here
#include <opencv2/objdetect.hpp>	// CascadeClassifier is in here

#include "Parameters.h"

using namespace cv;
using namespace std;

class EyeDetector
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
	const double rorates[5] = {0.0, 15.0, -15.0, 30.0, -30.0};
	const double scale = 1.0;
	CascadeClassifier cascade;
	CascadeClassifier nestedCascade;

	vector<Rect> rawFaces;
	vector<Rect> rawEyes;
	vector<Rect> resultFaces;	// Faces: saving absolute locations
	vector<Rect> resultEyes;	// Eyes: saving absolute locations
	vector<CvPoint> resultPupils;	// Pupils: pupil center corresponding to eyes, absolute locations.

	CvSize originalImage;
	CvPoint roi_lt_point, roi_rb_point;
	Mat roi_Image;
	Mat rotate_Image;

	bool IsFaceOverlap(Rect& newFace);
	void CascadeDetection(Mat& colorImg);
	void HoughPupilDetection(Mat& colorImg);

	CvPoint rotateBackPoints(CvPoint srcPoint, Mat& rbMat);
	void rotateBackRawInfo(Mat& rbMat);


public:
	EyeDetector();
	~EyeDetector();

	void ImageProcessAndDetect(Mat& colorImg, Mat& depth_to_color_img, const uint16_t one_meter);
	void DrawFacesAndEyes(Mat& colorImg);
	void ClearInfo();

	const size_t getEyesSize();
	const Rect getEyeLoc(int num);
};

