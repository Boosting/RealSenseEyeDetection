#include "EyeDetector.h"

#define CASCADE_FACE_XML_LOCATION "D:/Lib/OpenCV/build/etc/haarcascades/haarcascade_frontalface_default.xml"
#define CASCADE_EYE_XML_LOCATION "D:/Lib/OpenCV/build/etc/haarcascades/haarcascade_eye_tree_eyeglasses.xml"

EyeDetector::EyeDetector()
{
	scale = 1.0;

	if (!cascade.load(CASCADE_FACE_XML_LOCATION) || !nestedCascade.load(CASCADE_EYE_XML_LOCATION))
	{
		std::cout << "ERROR: Could not load Face / Eye classifier cascade" << std::endl;
	}
}

EyeDetector::~EyeDetector()
{
}

void EyeDetector::CascadeDetection(cv::Mat& colorImg) {
	int i = 0;
	std::vector<cv::Rect> faces;

	cv::Mat gray, smallImg(cvRound(colorImg.rows / scale), cvRound(colorImg.cols / scale), CV_8UC1);

	cv::cvtColor(colorImg, gray, cv::COLOR_BGR2GRAY);
	cv::resize(gray, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
	cv::equalizeHist(smallImg, smallImg);
	
	cascade.detectMultiScale(gray, faces,
		1.1, 3, 0
		//|CASCADE_DO_CANNY_PRUNING
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| cv::CASCADE_SCALE_IMAGE
		,
		cv::Size(50, 50));
	
	for (std::vector<cv::Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		cv::Mat smallImgROI;
		std::vector<cv::Rect> nestedObjects;

		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(*r);

		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			//|CASCADE_DO_CANNY_PRUNING
			| cv::CASCADE_SCALE_IMAGE
			,
			cv::Size(30, 30));

		if (nestedObjects.size() == 2) {
			rawFaces.push_back(*r);
			for (std::vector<cv::Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++)
			{
				rawEyes.push_back(*nr);
			}
		}
	}

	return;
}

void EyeDetector::ImageProcessAndDetect(cv::Mat& colorImg, cv::Mat& depth_to_color_img, const uint16_t one_meter) {
	double t = (double)cvGetTickCount();

	originalImage = depth_to_color_img.size();
	clearLastFrameInfo();

	// Depth images need to be smoothed to cancel noise, which may dramatically expand ROI
	cv::medianBlur(depth_to_color_img, depth_to_color_img, 5);

	// Resize image ROI, objects only in 50cm~300cm are detected
	for (int w = 0; w < originalImage.width; ++w) {
		for (int h = 0; h < originalImage.height; ++h) {
			uint16_t val = depth_to_color_img.at<uint16_t>(h, w);
			if (val < one_meter/2 || val>3*one_meter) continue;
			if (h < roi_lt_point.y) roi_lt_point.y = h;
			else if (h > roi_rb_point.y) roi_rb_point.y = h;
			if (w < roi_lt_point.x) roi_lt_point.x = w;
			else if (w > roi_rb_point.x) roi_rb_point.x = w;
		}
	}

	// If nothing appears, return
	if (roi_rb_point.x <= roi_lt_point.x || roi_rb_point.y <= roi_lt_point.y) return;

	// Show refined ROI, use this ROI to detect faces
	rectangle(colorImg, roi_lt_point, roi_rb_point, CV_RGB(255, 0, 0), 1, 8, 0);
	roi_Image = cv::Mat(colorImg, cvRect(roi_lt_point.x, roi_lt_point.y, roi_rb_point.x - roi_lt_point.x + 1, roi_rb_point.y - roi_lt_point.y + 1));
	rotate_Image = cv::Mat(roi_Image.size(), CV_8UC3);

	for (int i = 0; i < 5; ++i) {
		// Rotate image in different directions
		double rotate = rorates[i];
		cv::Point2f pt(roi_Image.cols / 2., roi_Image.rows / 2.);
		cv::Mat r = getRotationMatrix2D(pt, rotate, 1.0);
		warpAffine(roi_Image, rotate_Image, r, cv::Size(roi_Image.cols, roi_Image.rows));

		// Detect face and eyes, save information
		CascadeDetection(rotate_Image);
		size_t numFaces = rawFaces.size();
		std::vector<cv::Rect>::const_iterator faceItr = rawFaces.begin();
		std::vector<cv::Rect>::const_iterator eyeItr = rawEyes.begin();

		for (int j = 0; j < numFaces; ++j) {
			cv::Rect tmpFace = cvRect((*faceItr).x + roi_lt_point.x, (*faceItr).y + roi_lt_point.y, (*faceItr).width, (*faceItr).height);
			if (IsFaceOverlap(tmpFace)) {
				++faceItr;
				++eyeItr;
				++eyeItr;
				continue;
			};

			resultFaces.push_back(tmpFace);
			++faceItr;
			resultEyes.push_back(*eyeItr);
			++eyeItr;
			resultEyes.push_back(*eyeItr);
			++eyeItr;
		}

		rawFaces.clear();
		rawEyes.clear();
	}

	// Draw faces and eyes
	DrawFacesAndEyes(colorImg);

	t = (double)cvGetTickCount() - t;
	printf("numFaces = %d, detection time = %g ms\n", resultFaces.size(), t / ((double)cvGetTickFrequency()*1000.));
}

void EyeDetector::clearLastFrameInfo() {
	resultFaces.clear();
	resultEyes.clear();
	roi_lt_point = cvPoint(originalImage.width - 1, originalImage.height - 1);
	roi_rb_point = cvPoint(0, 0);
}


void EyeDetector::DrawFacesAndEyes(cv::Mat& colorImg) {
	size_t numFaces = resultFaces.size();
	size_t numEyes = resultEyes.size();
	if (numFaces * 2 != numEyes) return;
	
	std::vector<cv::Rect>::const_iterator faceItr = resultFaces.begin();
	std::vector<cv::Rect>::const_iterator eyeItr = resultEyes.begin();

	for (int i = 0; i < numFaces; ++i) {
		cv::Point center;
		cv::Scalar color = colors[i % 8];
		int radius;

		center.x = cvRound((faceItr->x + faceItr->width*0.5)*scale);
		center.y = cvRound((faceItr->y + faceItr->height*0.5)*scale);
		radius = cvRound((faceItr->width + faceItr->height)*0.25*scale);
		circle(colorImg, center, radius, color, 3, 8, 0);

		center.x = cvRound((faceItr->x + eyeItr->x + eyeItr->width*0.5)*scale);
		center.y = cvRound((faceItr->y + eyeItr->y + eyeItr->height*0.5)*scale);
		radius = cvRound((eyeItr->width + eyeItr->height)*0.25*scale);
		circle(colorImg, center, radius, color, 3, 8, 0);

		++eyeItr;

		center.x = cvRound((faceItr->x + eyeItr->x + eyeItr->width*0.5)*scale);
		center.y = cvRound((faceItr->y + eyeItr->y + eyeItr->height*0.5)*scale);
		radius = cvRound((eyeItr->width + eyeItr->height)*0.25*scale);
		circle(colorImg, center, radius, color, 3, 8, 0);

		++faceItr;
		++eyeItr;
	}
}

bool EyeDetector::IsFaceOverlap(cv::Rect& newFace) {

	for (std::vector<cv::Rect>::const_iterator faceItr = resultFaces.begin(); faceItr != resultFaces.end(); ++faceItr)
	{
		// If one rectangle is on left side of other
		if (newFace.x > (*faceItr).x + (*faceItr).width || (*faceItr).x > newFace.x + newFace.width) continue;
		// If one rectangle is above other
		if (newFace.y > (*faceItr).y + (*faceItr).height || (*faceItr).y > newFace.y + newFace.height) continue;
		return true;
	}
	return false;
}