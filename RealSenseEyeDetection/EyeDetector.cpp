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

void EyeDetector::CascadeDetection(Mat& colorImg) {
	int i = 0;
	vector<Rect> faces;

	Mat gray, smallImg(cvRound(colorImg.rows / scale), cvRound(colorImg.cols / scale), CV_8UC1);

	cvtColor(colorImg, gray, COLOR_BGR2GRAY);
	resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
	equalizeHist(smallImg, smallImg);
	
	cascade.detectMultiScale(gray, faces,
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

void EyeDetector::HoughPupilDetection(Mat& colorImg){
	// It is not working properly - can't detect pupils as expected.
	if (resultEyes.size() == 0) return;

	Mat gray;
	cvtColor(colorImg, gray, CV_BGR2GRAY);
	// smooth it, otherwise a lot of false circles may be detected
	// GaussianBlur(gray, gray, Size(9, 9), 2, 2);

	for (vector<Rect>::const_iterator eyeItr = resultEyes.begin(); eyeItr != resultEyes.end(); ++eyeItr) {
		Mat roiEye = Mat(gray, (*eyeItr));
		vector<Vec3f> circles;
		HoughCircles(roiEye, circles, CV_HOUGH_GRADIENT,
			2, roiEye.rows / 8);

		for (size_t i = 0; i < circles.size(); i++)
		{
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			// draw the circle center
			circle(colorImg, CvPoint((*eyeItr).x+ center.x, (*eyeItr).y + center.y), 3, Scalar(0, 255, 0), -1, 8, 0);
			// draw the circle outline
			circle(colorImg, CvPoint((*eyeItr).x + center.x, (*eyeItr).y + center.y), radius, Scalar(255, 0, 0), 3, 8, 0);
		}
	}
    return;
}

void EyeDetector::ImageProcessAndDetect(Mat& colorImg, Mat& depth_to_color_img, const uint16_t one_meter) {
	double t = (double)cvGetTickCount();
	double one_meter_d = (double)one_meter;

	originalImage = depth_to_color_img.size();

	// Depth images need to be smoothed to cancel noise, which may dramatically expand ROI
	medianBlur(depth_to_color_img, depth_to_color_img, 5);

	// Resize image ROI, objects only in 50cm~300cm are detected
	for (int w = 0; w < originalImage.width; ++w) {
		for (int h = 0; h < originalImage.height; ++h) {
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

	// Show refined ROI, use this ROI to detect faces
	rectangle(colorImg, roi_lt_point, roi_rb_point, CV_RGB(255, 0, 0), 1, 8, 0);
	roi_Image = Mat(colorImg, cvRect(roi_lt_point.x, roi_lt_point.y, roi_rb_point.x - roi_lt_point.x + 1, roi_rb_point.y - roi_lt_point.y + 1));
	rotate_Image = Mat(roi_Image.size(), CV_8UC3);

	for (int i = 0; i < 5; ++i) {
		// Rotate image in different directions, once get the eyes' coordinates, rotate the positions back to original image
		double rotate = rorates[i];
		Point2d pt(roi_Image.cols / 2., roi_Image.rows / 2.);
		Mat rMat = getRotationMatrix2D(pt, rotate, 1.0);
		Mat rbMat = getRotationMatrix2D(pt, -rotate, 1.0);
		warpAffine(roi_Image, rotate_Image, rMat, Size(roi_Image.cols, roi_Image.rows));

		// Detect face and eyes, save information
		CascadeDetection(rotate_Image);
		size_t numFaces = rawFaces.size();
		if (numFaces == 0) continue;

		// Detected faces and eyes are saved in raw info, not rotated back.
		// The following function is to get the source(ROI) coordinate of faces and eyes.
		// To get the global coordinates, simply add roi_lt_point.x and roi_lt_point.y to the points.
		rotateBackRawInfo(rbMat);
		vector<Rect>::const_iterator faceItr = rawFaces.begin();
		vector<Rect>::const_iterator eyeItr = rawEyes.begin();

		for (int j = 0; j < numFaces; ++j) {

			Rect tmpFace = cvRect((*faceItr).x + roi_lt_point.x, (*faceItr).y + roi_lt_point.y, (*faceItr).width, (*faceItr).height);
			if (IsFaceOverlap(tmpFace)) {
				++faceItr;
				++eyeItr;
				++eyeItr;
				continue;
			};

			resultFaces.push_back(tmpFace);
			++faceItr;
			resultEyes.push_back(cvRect((*eyeItr).x + roi_lt_point.x, (*eyeItr).y + roi_lt_point.y, (*eyeItr).width, (*eyeItr).height));
			++eyeItr;
			resultEyes.push_back(cvRect((*eyeItr).x + roi_lt_point.x, (*eyeItr).y + roi_lt_point.y, (*eyeItr).width, (*eyeItr).height));
			++eyeItr;
		}

		rawFaces.clear();
		rawEyes.clear();
	}

	//Detect pupils after eyes
	//HoughPupilDetection(rotate_Image);
	

	t = (double)cvGetTickCount() - t;
	printf("numFaces = %d, detection time = %g ms\n", resultFaces.size(), t / ((double)cvGetTickFrequency()*1000.));
}

void EyeDetector::ClearInfo() {
	resultFaces.clear();
	resultEyes.clear();
	roi_lt_point = cvPoint(INT_MAX, INT_MAX);
	roi_rb_point = cvPoint(0, 0);
}


void EyeDetector::DrawFacesAndEyes(Mat& colorImg) {
	size_t numFaces = resultFaces.size();
	size_t numEyes = resultEyes.size();
	if (numFaces * 2 != numEyes) return;
	
	vector<Rect>::const_iterator faceItr = resultFaces.begin();
	vector<Rect>::const_iterator eyeItr = resultEyes.begin();

	for (int i = 0; i < numFaces; ++i) {
		Point center;
		Scalar color = colors[i % 8];
		int radius;

		center.x = cvRound((faceItr->x + faceItr->width*0.5)*scale);
		center.y = cvRound((faceItr->y + faceItr->height*0.5)*scale);
		radius = cvRound((faceItr->width + faceItr->height)*0.25*scale);
		circle(colorImg, center, radius, color, 3, 8, 0);
		++faceItr;

		center.x = cvRound((eyeItr->x + eyeItr->width*0.5)*scale);
		center.y = cvRound((eyeItr->y + eyeItr->height*0.5)*scale);
		radius = cvRound((eyeItr->width + eyeItr->height)*0.25*scale);
		circle(colorImg, center, radius, color, 3, 8, 0);
		++eyeItr;

		center.x = cvRound((eyeItr->x + eyeItr->width*0.5)*scale);
		center.y = cvRound((eyeItr->y + eyeItr->height*0.5)*scale);
		radius = cvRound((eyeItr->width + eyeItr->height)*0.25*scale);
		circle(colorImg, center, radius, color, 3, 8, 0);
		++eyeItr;
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

	for (int i = 0; i < numFaces; ++i) {
		CvPoint faceLT((*faceItr).x, (*faceItr).y);
		CvPoint faceRB((*faceItr).x + (*faceItr).width, (*faceItr).y + (*faceItr).height);

		CvPoint eye1LT(faceLT.x + (*eyeItr).x, faceLT.y + (*eyeItr).y);
		CvPoint eye1RB(eye1LT.x + (*eyeItr).width, eye1LT.y + (*eyeItr).height);
		eye1LT = rotateBackPoints(eye1LT, rbMat);
		eye1RB = rotateBackPoints(eye1RB, rbMat);
		(*eyeItr).x = eye1LT.x;
		(*eyeItr).y = eye1LT.y;
		(*eyeItr).width = eye1RB.x - eye1LT.x;
		(*eyeItr).height = eye1RB.y - eye1LT.y;
		++eyeItr;

		CvPoint eye2LT(faceLT.x + (*eyeItr).x, faceLT.y + (*eyeItr).y);
		CvPoint eye2RB(eye2LT.x + (*eyeItr).width, eye2LT.y + (*eyeItr).height);
		eye2LT = rotateBackPoints(eye2LT, rbMat);
		eye2RB = rotateBackPoints(eye2RB, rbMat);
		(*eyeItr).x = eye2LT.x;
		(*eyeItr).y = eye2LT.y;
		(*eyeItr).width = eye2RB.x - eye2LT.x;
		(*eyeItr).height = eye2RB.y - eye2LT.y;
		++eyeItr;

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

CvPoint EyeDetector::rotateBackPoints(CvPoint srcPoint, Mat& rbMat) {
	double np1 = srcPoint.x, np2 = srcPoint.y;
	double p1, p2;
	p1 = np1 * rbMat.at<double>(0, 0) + np2 * rbMat.at<double>(0, 1) + rbMat.at<double>(0, 2);
	p2 = np1 * rbMat.at<double>(1, 0) + np2 * rbMat.at<double>(1, 1) + rbMat.at<double>(1, 2);
	return CvPoint((int)p1, (int)p2);
}

const size_t EyeDetector::getEyesSize() {
	return resultEyes.size();
}

const Rect EyeDetector::getEyeLoc(int num) {
	assert(num <= resultEyes.size());
	return resultEyes[num];
}