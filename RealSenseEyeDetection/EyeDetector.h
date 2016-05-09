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
	// Face and eye detection tools.
	cv::CascadeClassifier cascade;
	cv::CascadeClassifier nestedCascade;
	const string casdace_face_xml_location = "D:/Lib/OpenCV/build/etc/haarcascades/haarcascade_frontalface_default.xml";
	const string casdace_eye_xml_location = "D:/Lib/OpenCV/build/etc/haarcascades/haarcascade_eye_tree_eyeglasses.xml";

	// Closest and Farthest face detection distance (in meters).
	const float closest_depth_distance = 0.5;
	const float farthest_depth_distance = 3.0;

	// Different colors to mark different detected faces and eyes.
	// Same people has same mark color of eyes and face.
	const cv::Scalar colors[8] = { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };

	// Try different rotate angles to detect reasonable rolls of face.
	const double rorates[3] = {0.0, -20.0, 20.0};

	// Scale of resizing an image - no scaling.
	const double scale = 1.0;

	// "EyeLike" Algorithm Parameters
	const int kFastEyeWidth = 50;
	const int kWeightBlurSize = 5;
	const bool kEnableWeight = true;
	const float kWeightDivisor = 1.0;
	const double kGradientThreshold = 50.0;

	// Containers saving locations of detected items.
	vector<cv::Rect> rawFaces;		// rawFaces: saving absolute locations
	vector<cv::Rect> rawEyes;		// rawEyes: before rotate back: saving location offset corresponding to faces; after rotate back: saving absolute locations
	vector<cv::Point> rawPupils;	// rawPupils: before rotate back: saving location offset corresponding to eyes; after rotate back: saving absolute locations
	vector<cv::Rect> resultFaces;	// Faces: saving absolute locations
	vector<cv::Rect> resultEyes;	// Eyes: saving absolute locations
	vector<cv::Point> resultPupils;	// Pupils: pupil center corresponding to eyes, absolute locations.

	CvSize original_Image_size;
	cv::Point roi_lt_point, roi_rb_point;
	cv::Mat gray_Image;
	cv::Mat roi_Image;
	cv::Mat rotated_Image;

	// Check if a new detected face overlaps with any already detected faces.
	// @out true if overlaps, false otherwise
	// @param newFace The new detected face
	bool IsFaceOverlap(cv::Rect& newFace);

	// Face and eye detection using OpenCV method.
	// It saves detected faces in rawFaces, and saves detected eyes in rawEyes. -- Only save faces which have exactly 2 eyes. 
	// i.e. rawFaces.size()*2 equals to rawEyes.size(), and rawFaces[i] has eyes in rawEyes[i*2] and rawEyes[i*2+1].
	void CascadeDetection(cv::Mat& inputImg);

	// For each detected eye, detect and save its pupil location to rawPupils.
	// i.e. rawEyes[i] has pupil location in rawPupils[i].
	void PupilDetection(cv::Mat& inputImg);

	// Detected points are on their rotated locations in a rotated image.
	// This function can get their original locations.
	// @out original location of a point in rotated image.
	// @param srcPoint : a point to be rotated back.
	// @param rbMat : rotate back matrix.
	cv::Point rotateBackPoints(cv::Point srcPoint, cv::Mat& rbMat);

	// Rotate back all raw info. 
	// Dealing with rawFaces, rawEyes, and rawPupils. Results saved in the same places.
	// Detected rects(faces and eyes) and points(pupils) are on their rotated locations in a rotated image.
	// This function can get their original locations.
	// @param rbMat : rotate back matrix.
	void rotateBackRawInfo(cv::Mat& rbMat);

	// eyeLike functions
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

	// Process and detect image, all results are saved in resultFaces, resultEyes and resultPupils.
	// @param colorImg
	// @param depth_to_color_img : To get a smaller ROI of color image.
	// @param one_meter : The value equals to one metes in depth image.
	void ImageProcessAndDetect(cv::Mat& colorImg, cv::Mat& depth_to_color_img, const uint16_t one_meter);

	// Draw detected faces, eyes and pupils (from resultFaces, resultEyes and resultPupils) to colorImg.
	void DrawDetectedInfo(cv::Mat& colorImg);

	// Clear all raw and result info for current frame.
	void ClearInfo();

	const size_t getFacesSize();
	const size_t getEyesSize();
	// Get info of an result eye.
	const cv::Rect getEyeLoc(int num);
	const size_t getPupilsSize();
	// Get info of an result pupil.
	const cv::Point getPupilLoc(int num);

};

