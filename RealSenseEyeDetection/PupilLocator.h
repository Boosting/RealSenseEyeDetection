#pragma once
#include <iostream>
#include <opencv2/core.hpp>			// cv::Mat is in here
#include <opencv2/imgproc.hpp>		// circle is in here
#include <opencv2/highgui.hpp>		// imshow is in here

using namespace std;

class PupilLocator
{
private:
	// White to mark detected pupil
	const cv::Scalar color = CV_RGB(255, 255, 255);
	// Scale of resizing an image - no scaling.
	const double scale = 1.0;
	// Used in FindPupilDepthPoints.
	const int max_reference_search_pixels = 10;

	vector<cv::Point> detectedPupils;			// Pupil points in color images
	vector<cv::Point> pupilDepthPoints;			// Pupil points in depth images
	vector<cv::Point> pupilReferencePoints;		// Reference points of pupils in depth images

public:
	PupilLocator();
	~PupilLocator();

	// Add a (cv::Point)detectedPupil to detectedPupils vector.
	void AddPupil(cv::Point detectedPupil);

	// It is common that pupil points in depth images doesn't have depth value.
	// To get the depth of pupil, I pick closest 2 points (as reference points) in 2 directions as referesce points with equal distance which has depth value.
	// They are either top & bottom pairs, or left & right pairs. Each pupil has exact 2 reference points.
	// i.e. One pupil matches detectedPupils[i], also matches pupilDepthPoints[i], and has 2 reference points which are pupilReferencePoints[i*2] and pupilReferencePoints[i*2+1].
	// @out return false: couldn't find any reference points for some pupils; true otherwise.
	// @param convert_imgX : lookup table saving X coordinates from color to depth
	// @param convert_imgY : lookup table saving Y coordinates from color to depth
	bool FindPupilDepthPoints(cv::Mat& convert_imgX, cv::Mat& convert_imgY);

	// Draw detected pupils (from pupilDepthPoints) to depthImg.
	void DrawPupils(cv::Mat& depthImg);

	// Clear all raw and result info for current frame.
	void ClearInfo();

	const size_t getPupilDepthSize();
	// Get info of an result pupil depth location.
	const cv::Point getPupilDepthLoc(int num);
	const size_t getPupilReferenceSize();
	// Get info of one of an result pupil's reference location.
	const cv::Point getPupilReferenceLoc(int num);
};

