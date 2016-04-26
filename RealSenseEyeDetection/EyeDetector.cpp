#include "EyeDetector.h"


EyeDetector::EyeDetector()
{
	if (!cascade.load(CASCADE_FACE_XML_LOCATION) || !nestedCascade.load(CASCADE_EYE_XML_LOCATION))
	{
		cout << "ERROR: Could not load Face / Eye classifier cascade" << endl;
	}
}

EyeDetector::~EyeDetector()
{
}

void EyeDetector::CascadeDetection(Mat& inputImg) {
	int i = 0;
	vector<Rect> faces;

	Mat smallImg(cvRound(inputImg.rows / scale), cvRound(inputImg.cols / scale), CV_8UC1);

	resize(inputImg, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);
	
	cascade.detectMultiScale(inputImg, faces,
		1.1, 3, 0
		//|CASCADE_DO_CANNY_PRUNING
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| CASCADE_SCALE_IMAGE
		,
		Size(50, 50));
	
	for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		Mat smallImgROI;
		vector<Rect> nestedObjects;

		if (nestedCascade.empty())
			continue;
		smallImgROI = smallImg(*r);

		nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
			1.1, 2, 0
			//|CASCADE_FIND_BIGGEST_OBJECT
			//|CASCADE_DO_ROUGH_SEARCH
			//|CASCADE_DO_CANNY_PRUNING
			| CASCADE_SCALE_IMAGE
			,
			Size(15, 15));

		if (nestedObjects.size() == 2) {
			rawFaces.push_back(*r);
			for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++)
			{
				rawEyes.push_back(*nr);
			}
		}
	}

	return;
}

void EyeDetector::PupilDetection(Mat& inputImg){
	if (rawEyes.empty()) return;
	bool findPupil = false;

	for (vector<Rect>::const_iterator eyeItr = rawEyes.begin(); eyeItr != rawEyes.end(); ++eyeItr) {
		findPupil = false;
		Mat grayROI(inputImg, (*eyeItr));

		// Convert to binary image by thresholding it
		threshold(grayROI, grayROI, 220, 255, THRESH_BINARY);

		// Find all contours
		vector<vector<Point>> contours;
		findContours(grayROI.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		// Fill holes in each contour
		drawContours(grayROI, contours, -1, CV_RGB(255, 255, 255), -1);

		for (int i = 0; i < contours.size(); i++)
		{
			double area = contourArea(contours[i]);
			Rect rect = boundingRect(contours[i]);
			int radius = rect.width / 2;

			// If contour is big enough and has round shape
			// Then it is the pupil
			if (area >= 30 &&
				std::abs(1 - ((double)rect.width / (double)rect.height)) <= 0.2 &&
				std::abs(1 - (area / (CV_PI * std::pow(radius, 2)))) <= 0.2)
			{
				rawPupils.push_back(Point(rect.x + radius, rect.y + radius));
				findPupil = true;
				break;
			}
		}

		if (findPupil == false)
			rawPupils.push_back(Point((*eyeItr).x + (*eyeItr).width / 2, (*eyeItr).y + (*eyeItr).height / 2));
	}
}

void EyeDetector::ImageProcessAndDetect(Mat& colorImg, Mat& depth_to_color_img, const uint16_t one_meter) {
	double t = (double)cvGetTickCount();
	double one_meter_d = (double)one_meter;

	original_Image_size = depth_to_color_img.size();

	// Depth images need to be smoothed to cancel noise, which may dramatically expand ROI
	medianBlur(depth_to_color_img, depth_to_color_img, 5);

	// Resize image ROI, objects only in 50cm~300cm are detected
	for (int w = 0; w < original_Image_size.width; ++w) {
		for (int h = 0; h < original_Image_size.height; ++h) {
			double val = (double)depth_to_color_img.at<uint16_t>(h, w);
			if (val < one_meter_d * CLOSEST_DEPTH_DISTANCE || val > one_meter_d * FARTHEST_DEPTH_DISTANCE) continue;
			if (h < roi_lt_point.y) roi_lt_point.y = h;
			else if (h > roi_rb_point.y) roi_rb_point.y = h;
			if (w < roi_lt_point.x) roi_lt_point.x = w;
			else if (w > roi_rb_point.x) roi_rb_point.x = w;
		}
	}

	// If nothing appears, return
	if (roi_rb_point.x <= roi_lt_point.x || roi_rb_point.y <= roi_lt_point.y) return;

	cvtColor(colorImg, gray_Image, COLOR_BGR2GRAY);

	// Show refined ROI, use this ROI to detect faces
	rectangle(colorImg, roi_lt_point, roi_rb_point, CV_RGB(255, 0, 0), 1, 8, 0);
	roi_Image = Mat(gray_Image, Rect(roi_lt_point.x, roi_lt_point.y, roi_rb_point.x - roi_lt_point.x + 1, roi_rb_point.y - roi_lt_point.y + 1));
	rotated_Image = Mat(roi_Image.size(), CV_8UC1);

	for (int i = 0; i < 5; ++i) {
		// Rotate image in different directions, once get the eyes' coordinates, rotate the positions back to original image
		double rotate = rorates[i];
		Point2d pt(roi_Image.cols / 2., roi_Image.rows / 2.);
		Mat rMat = getRotationMatrix2D(pt, rotate, 1.0);
		Mat rbMat = getRotationMatrix2D(pt, -rotate, 1.0);
		warpAffine(roi_Image, rotated_Image, rMat, Size(roi_Image.cols, roi_Image.rows));

		// Detect face and eyes, save information
		CascadeDetection(rotated_Image);
		PupilDetection(rotated_Image);
		size_t numFaces = rawFaces.size();
		if (numFaces == 0) continue;
		if (numFaces * 2 != rawEyes.size()) continue;
		if (numFaces * 2 != rawPupils.size()) continue;

		// Detected faces and eyes are saved in raw info, not rotated back.
		// The following function is to get the source(ROI) coordinate of faces and eyes.
		// To get the global coordinates, simply add roi_lt_point.x and roi_lt_point.y to the points.
		rotateBackRawInfo(rbMat);
		vector<Rect>::const_iterator faceItr = rawFaces.begin();
		vector<Rect>::const_iterator eyeItr = rawEyes.begin();
		vector<Point>::const_iterator pupilItr = rawPupils.begin();

		for (int j = 0; j < numFaces; ++j) {

			Rect tmpFace = Rect((*faceItr).x + roi_lt_point.x, (*faceItr).y + roi_lt_point.y, (*faceItr).width, (*faceItr).height);
			if (IsFaceOverlap(tmpFace)) {
				++faceItr;
				for (int k = 0; k < 2; ++k) {
					++eyeItr;
					++pupilItr;
				}
				continue;
			};

			resultFaces.push_back(tmpFace);
			++faceItr;
			for (int k = 0; k < 2; ++k) {
				resultEyes.push_back(Rect((*eyeItr).x + roi_lt_point.x, (*eyeItr).y + roi_lt_point.y, (*eyeItr).width, (*eyeItr).height));
				++eyeItr;
				resultPupils.push_back(Point((*pupilItr).x + roi_lt_point.x, (*pupilItr).y + roi_lt_point.y));
				++pupilItr;
			}
		}

		rawFaces.clear();
		rawEyes.clear();
		rawPupils.clear();
	}	

	t = (double)cvGetTickCount() - t;
	printf("numFaces = %d, detection time = %g ms\n", resultFaces.size(), t / ((double)cvGetTickFrequency()*1000.));
}

void EyeDetector::ClearInfo() {
	resultFaces.clear();
	resultEyes.clear();
	resultPupils.clear();
	roi_lt_point = Point(INT_MAX, INT_MAX);
	roi_rb_point = Point(0, 0);
}

void EyeDetector::DrawDetectedInfo(Mat& colorImg) {
	size_t numFaces = resultFaces.size();
	size_t numEyes = resultEyes.size();
	
	vector<Rect>::const_iterator faceItr = resultFaces.begin();
	vector<Rect>::const_iterator eyeItr = resultEyes.begin();
	vector<Point>::const_iterator pupilItr = resultPupils.begin();

	for (int i = 0; i < numFaces; ++i) {
		Point center;
		Scalar color = colors[i % 8];
		int radius;

		center.x = cvRound((faceItr->x + faceItr->width*0.5)*scale);
		center.y = cvRound((faceItr->y + faceItr->height*0.5)*scale);
		radius = cvRound((faceItr->width + faceItr->height)*0.25*scale);
		circle(colorImg, center, radius, color, 3, 8, 0);
		++faceItr;

		for (int j = 0; j < 2; ++j) {
			center.x = cvRound((eyeItr->x + eyeItr->width*0.5)*scale);
			center.y = cvRound((eyeItr->y + eyeItr->height*0.5)*scale);
			radius = cvRound((eyeItr->width + eyeItr->height)*0.25*scale);
			circle(colorImg, center, radius, color, 3, 8, 0);
			center.x = pupilItr->x;
			center.y = pupilItr->y;
			radius = 3;
			circle(colorImg, center, radius, CV_RGB(0, 255, 0), 1, 8, 0);
			++pupilItr;
			++eyeItr;
		}
	}
}

bool EyeDetector::IsFaceOverlap(Rect& newFace) {

	for (vector<Rect>::const_iterator faceItr = resultFaces.begin(); faceItr != resultFaces.end(); ++faceItr)
	{
		// If one rectangle is on left side of other
		if (newFace.x > (*faceItr).x + (*faceItr).width || (*faceItr).x > newFace.x + newFace.width) continue;
		// If one rectangle is above other
		if (newFace.y > (*faceItr).y + (*faceItr).height || (*faceItr).y > newFace.y + newFace.height) continue;
		return true;
	}
	return false;
}

void EyeDetector::rotateBackRawInfo(Mat& rbMat) {
	size_t numFaces = rawFaces.size();
	vector<Rect>::iterator faceItr = rawFaces.begin();
	vector<Rect>::iterator eyeItr = rawEyes.begin();
	vector<Point>::iterator pupilItr = rawPupils.begin();

	for (int i = 0; i < numFaces; ++i) {
		Point faceLT((*faceItr).x, (*faceItr).y);
		Point faceRB((*faceItr).x + (*faceItr).width, (*faceItr).y + (*faceItr).height);

		for (int j = 0; j < 2; ++j) {
			Point eyeLT(faceLT.x + (*eyeItr).x, faceLT.y + (*eyeItr).y);
			Point eyeRB(eyeLT.x + (*eyeItr).width, eyeLT.y + (*eyeItr).height);
			eyeLT = rotateBackPoints(eyeLT, rbMat);
			eyeRB = rotateBackPoints(eyeRB, rbMat);
			(*eyeItr).x = eyeLT.x;
			(*eyeItr).y = eyeLT.y;
			(*eyeItr).width = eyeRB.x - eyeLT.x;
			(*eyeItr).height = eyeRB.y - eyeLT.y;
			++eyeItr;

			Point pupil(faceLT.x + (*pupilItr).x, faceLT.y + (*pupilItr).y);
			pupil = rotateBackPoints(pupil, rbMat);
			(*pupilItr).x = pupil.x;
			(*pupilItr).y = pupil.y;
			++pupilItr;
		}

		faceLT = rotateBackPoints(faceLT, rbMat);
		faceRB = rotateBackPoints(faceRB, rbMat);
		(*faceItr).x = faceLT.x;
		(*faceItr).y = faceLT.y;
		(*faceItr).width = faceRB.x - faceLT.x;
		(*faceItr).height = faceRB.y - faceLT.y;
		++faceItr;
	}

	return;
}

Point EyeDetector::rotateBackPoints(Point srcPoint, Mat& rbMat) {
	double np1 = srcPoint.x, np2 = srcPoint.y;
	double p1, p2;
	p1 = np1 * rbMat.at<double>(0, 0) + np2 * rbMat.at<double>(0, 1) + rbMat.at<double>(0, 2);
	p2 = np1 * rbMat.at<double>(1, 0) + np2 * rbMat.at<double>(1, 1) + rbMat.at<double>(1, 2);
	return Point((int)p1, (int)p2);
}

const size_t EyeDetector::getEyesSize() {
	return resultEyes.size();
}

const Rect EyeDetector::getEyeLoc(int num) {
	assert(num <= resultEyes.size());
	return resultEyes[num];
}