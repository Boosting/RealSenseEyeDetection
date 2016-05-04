#pragma once
#include <iostream>
#include <opencv2/core.hpp>			// cv::Mat is in here
#include <opencv2/imgproc.hpp>		// circle is in here
#include <opencv2/objdetect.hpp>	// cv::CascadeClassifier is in here
#include <opencv2/highgui.hpp>		// imshow is in here
#include <queue>

using namespace std;

class EyeDetector
{
private:

	/////////////// constants in eye detection //////////////////////

	const string casdace_face_xml_location = "D:/Lib/OpenCV/build/etc/haarcascades/haarcascade_frontalface_default.xml";
	const string casdace_eye_xml_location = "D:/Lib/OpenCV/build/etc/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

	// Closest and Farthest face detection distance (in meters)
	const float closest_depth_distance = 0.5;
	const float farthest_depth_distance = 3.0;

	const cv::Scalar colors[8] = { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };
	const double rorates[3] = {0.0, -20.0, 20.0};
	const double scale = 1.0;

	// Algorithm Parameters
	const int kFastEyeWidth = 50;
	const int kWeightBlurSize = 5;
	const bool kEnableWeight = true;
	const float kWeightDivisor = 1.0;
	const double kGradientThreshold = 50.0;

	cv::CascadeClassifier cascade;
	cv::CascadeClassifier nestedCascade;

	vector<cv::Rect> rawFaces;		// rawFaces: saving absolute locations
	vector<cv::Rect> rawEyes;		// rawEyes: before rotate back: saving location offset corresponding to faces
								//          after rotate back: saving absolute locations
	vector<cv::Point> rawPupils;	// rawPupils: before rotate back: saving location offset corresponding to eyes
								//            after rotate back: saving absolute locations
	vector<cv::Rect> resultFaces;	// Faces: saving absolute locations
	vector<cv::Rect> resultEyes;	// Eyes: saving absolute locations
	vector<cv::Point> resultPupils;	// Pupils: pupil center corresponding to eyes, absolute locations.

	CvSize original_Image_size;
	cv::Point roi_lt_point, roi_rb_point;
	cv::Mat gray_Image;
	cv::Mat roi_Image;
	cv::Mat rotated_Image;

	bool IsFaceOverlap(cv::Rect& newFace);
	void CascadeDetection(cv::Mat& inputImg);
	void PupilDetection(cv::Mat& inputImg);

	cv::Point rotateBackPoints(cv::Point srcPoint, cv::Mat& rbMat);
	void rotateBackRawInfo(cv::Mat& rbMat);

	// eyeLike functions - proessing
	cv::Point findEyeCenter(cv::Mat& face, cv::Rect& eye);
	void scaleToFastSize(const cv::Mat &src, cv::Mat &dst);
	cv::Mat computeMatXGradient(const cv::Mat &Mat);
	cv::Mat matrixMagnitude(const cv::Mat &MatX, const cv::Mat &MatY);
	double computeDynamicThreshold(const cv::Mat &Mat, double stdDevFactor);
	void EyeDetector::testPossibleCentersFormula(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out);
	cv::Point EyeDetector::unscalePoint(cv::Point p, cv::Rect origSize);

public:
	EyeDetector();
	~EyeDetector();

	void ImageProcessAndDetect(cv::Mat& colorImg, cv::Mat& depth_to_color_img, const uint16_t one_meter);
	void DrawDetectedInfo(cv::Mat& colorImg);
	void ClearInfo();

	const size_t getFacesSize();
	const size_t getEyesSize();
	const cv::Rect getEyeLoc(int num);
	const size_t getPupilsSize();
	const cv::Point getPupilLoc(int num);

};

