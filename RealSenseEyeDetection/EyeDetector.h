#pragma once
#include <iostream>
#include <opencv2/core.hpp>			// cv::Mat is in here
#include <opencv2/imgproc.hpp>		// cv::circle is in here
#include <opencv2/highgui.hpp>		// cv::imshow is in here
#include <opencv2/objdetect.hpp>	// CascadeClassifier is in here

class EyeDetector
{
private:
	double scale;
	bool tryflip;
	cv::CascadeClassifier cascade;
	cv::CascadeClassifier nestedCascade;

public:
	EyeDetector();
	~EyeDetector();

	void CascadeDetection(cv::Mat& colorImg);
};

