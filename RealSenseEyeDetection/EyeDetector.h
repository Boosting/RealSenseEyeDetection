#pragma once
#include <iostream>
#include <opencv2/core.hpp>			// cv::Mat is in here
#include <opencv2/imgproc.hpp>		// cv::circle is in here
#include <opencv2/highgui.hpp>		// cv::imshow is in here
#include <opencv2/objdetect.hpp>	// CascadeClassifier is in here

class EyeDetector
{
private:
	const cv::Scalar colors[8] = { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };
	const double rorates[5] = {0.0, 15.0, -15.0, 30.0, -30.0};
	double scale;
	cv::CascadeClassifier cascade;
	cv::CascadeClassifier nestedCascade;

	std::vector<cv::Rect> rawFaces;
	std::vector<cv::Rect> rawEyes;
	// Faces: saving absolute locations
	std::vector<cv::Rect> resultFaces;
	// Eyes: saving local locations based on faces, 1 face for 2 eyes.
	std::vector<cv::Rect> resultEyes;

	CvSize originalImage;
	CvPoint roi_lt_point, roi_rb_point;
	cv::Mat roi_Image;
	cv::Mat rotate_Image;

	bool IsFaceOverlap(cv::Rect& newFace);
	void CascadeDetection(cv::Mat& colorImg);
	void DrawFacesAndEyes(cv::Mat& colorImg);
	void clearLastFrameInfo();


public:
	EyeDetector();
	~EyeDetector();

	void ImageProcessAndDetect(cv::Mat& colorImg, cv::Mat& depth_to_color_img, const uint16_t one_meter);
};

