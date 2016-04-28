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
	const Scalar color = CV_RGB(255, 255, 255);
	const double scale = 1.0;
	const int max_reference_search_pixels = 10;

	vector<Point> detectedPupils;			// Pupil points in color images
	vector<Point> pupilDepthPoints;			// Pupil points in depth images

	// It is common that pupil points in depth images doesn't have depth value.
	// To get the depth of pupil, I pick closest 2 points in 2 directions as referesce points with equal distance which has depth value.
	// They are either top & bottom pairs, or left & right pairs
	vector<Point> pupilReferencePoints;		// Reference points of pupils in depth images

public:
	PupilLocator();
	~PupilLocator();

	void AddPupil(Point detectedPupil);
	bool FindPupilDepthPoints(Mat& convert_imgX, Mat& convert_imgY);
	void DrawPupils(Mat& depthImg);
	void ClearInfo();

	const size_t getPupilDepthSize();
	const Point getPupilDepthLoc(int num);
	const size_t getPupilReferenceSize();
	const Point getPupilReferenceLoc(int num);
};

