#pragma once
#include <iostream>
#include <opencv2/core.hpp>			// Mat is in here
#include <opencv2/imgproc.hpp>		// circle is in here
#include <opencv2/objdetect.hpp>	// CascadeClassifier is in here
#include <opencv2/highgui.hpp>		// imshow is in here
#include <queue>

using namespace cv;
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

	const Scalar colors[8] = { CV_RGB(0,0,255),
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

	CascadeClassifier cascade;
	CascadeClassifier nestedCascade;

	vector<Rect> rawFaces;		// rawFaces: saving absolute locations
	vector<Rect> rawEyes;		// rawEyes: before rotate back: saving location offset corresponding to faces
								//          after rotate back: saving absolute locations
	vector<Point> rawPupils;	// rawPupils: before rotate back: saving location offset corresponding to eyes
								//            after rotate back: saving absolute locations
	vector<Rect> resultFaces;	// Faces: saving absolute locations
	vector<Rect> resultEyes;	// Eyes: saving absolute locations
	vector<Point> resultPupils;	// Pupils: pupil center corresponding to eyes, absolute locations.

	CvSize original_Image_size;
	Point roi_lt_point, roi_rb_point;
	Mat gray_Image;
	Mat roi_Image;
	Mat rotated_Image;

	bool IsFaceOverlap(Rect& newFace);
	void CascadeDetection(Mat& inputImg);
	void PupilDetection(Mat& inputImg);

	Point rotateBackPoints(Point srcPoint, Mat& rbMat);
	void rotateBackRawInfo(Mat& rbMat);

	// eyeLike functions - proessing
	Point findEyeCenter(Mat& face, Rect& eye);
	void scaleToFastSize(const Mat &src, Mat &dst);
	Mat computeMatXGradient(const Mat &mat);
	cv::Mat matrixMagnitude(const cv::Mat &matX, const cv::Mat &matY);
	double computeDynamicThreshold(const cv::Mat &mat, double stdDevFactor);
	void EyeDetector::testPossibleCentersFormula(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out);
	cv::Point EyeDetector::unscalePoint(cv::Point p, cv::Rect origSize);

public:
	EyeDetector();
	~EyeDetector();

	void ImageProcessAndDetect(Mat& colorImg, Mat& depth_to_color_img, const uint16_t one_meter);
	void DrawDetectedInfo(Mat& colorImg);
	void ClearInfo();

	const size_t getEyesSize();
	const Rect getEyeLoc(int num);
};

