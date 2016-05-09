#include "EyeDetector.h"

EyeDetector::EyeDetector()
{
	if (!cascade.load(casdace_face_xml_location) || !nestedCascade.load(casdace_eye_xml_location))
	{
		cout << "ERROR: Could not load Face / Eye classifier cascade" << endl;
	}
}

EyeDetector::~EyeDetector()
{
}

void EyeDetector::CascadeDetection(cv::Mat& inputImg) {
	int i = 0;
	vector<cv::Rect> faces;

	cv::Mat smallImg(cvRound(inputImg.rows / scale), cvRound(inputImg.cols / scale), CV_8UC1);

	resize(inputImg, smallImg, smallImg.size(), 0, 0, cv::INTER_LINEAR);
	equalizeHist(smallImg, smallImg);
	
	cascade.detectMultiScale(inputImg, faces,
		1.1, 3, 0
		//|CASCADE_DO_CANNY_PRUNING
		//|CASCADE_FIND_BIGGEST_OBJECT
		//|CASCADE_DO_ROUGH_SEARCH
		| cv::CASCADE_SCALE_IMAGE
		,
		cv::Size(50, 50));
	
	for (vector<cv::Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++)
	{
		cv::Mat smallImgROI;
		vector<cv::Rect> nestedObjects;

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
			cv::Size(15, 15));

		if (nestedObjects.size() == 2) {
			rawFaces.push_back(*r);
			for (vector<cv::Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++)
			{
				rawEyes.push_back(*nr);
			}
		}
	}

	return;
}

void EyeDetector::PupilDetection(cv::Mat& inputImg){
	if (rawEyes.empty()) return;

	size_t numFaces = rawFaces.size();
	size_t numEyes = rawEyes.size();

	vector<cv::Rect>::const_iterator faceItr = rawFaces.begin();
	vector<cv::Rect>::const_iterator eyeItr = rawEyes.begin();

	for (int i = 0; i < numFaces; ++i) {
		// For each eye of all faces, run findEyeCenter function.
		for (int j = 0; j < 2; ++j) {
			cv::Rect eyeROI = cv::Rect((*faceItr).x + (*eyeItr).x, (*faceItr).y + (*eyeItr).y, (*eyeItr).width, (*eyeItr).height);
			cv::Point pupil = findEyeCenter(inputImg, eyeROI);
			rawPupils.push_back(pupil);
			++eyeItr;
		}
		++faceItr;
	}
}

void EyeDetector::ImageProcessAndDetect(cv::Mat& colorImg, cv::Mat& depth_to_color_img, const uint16_t one_meter) {
	double one_meter_d = (double)one_meter;

	original_Image_size = depth_to_color_img.size();

	// Depth images need to be smoothed to cancel noise, which may dracv::Matically expand ROI
	medianBlur(depth_to_color_img, depth_to_color_img, 5);

	// Resize image ROI, objects only in 50cm~300cm are detected
	for (int w = 0; w < original_Image_size.width; ++w) {
		for (int h = 0; h < original_Image_size.height; ++h) {
			double val = (double)depth_to_color_img.at<uint16_t>(h, w);
			if (val < one_meter_d * closest_depth_distance || val > one_meter_d * farthest_depth_distance) continue;
			if (h < roi_lt_point.y) roi_lt_point.y = h;
			else if (h > roi_rb_point.y) roi_rb_point.y = h;
			if (w < roi_lt_point.x) roi_lt_point.x = w;
			else if (w > roi_rb_point.x) roi_rb_point.x = w;
		}
	}

	// If nothing appears, return
	if (roi_rb_point.x <= roi_lt_point.x || roi_rb_point.y <= roi_lt_point.y) return;

	cvtColor(colorImg, gray_Image, CV_BGR2GRAY);

	// Show refined ROI, use this ROI to detect faces
	rectangle(colorImg, roi_lt_point, roi_rb_point, CV_RGB(255, 0, 0), 1, 8, 0);
	roi_Image = cv::Mat(gray_Image, cv::Rect(roi_lt_point.x, roi_lt_point.y, roi_rb_point.x - roi_lt_point.x + 1, roi_rb_point.y - roi_lt_point.y + 1));
	rotated_Image = cv::Mat(roi_Image.size(), CV_8UC1);

	for (int i = 0; i < 3 ; ++i) {
		// Rotate image in different directions, once get the eyes' coordinates, rotate the positions back to original image
		double rotate = rorates[i];
		cv::Point2d pt(roi_Image.cols / 2., roi_Image.rows / 2.);
		cv::Mat rMat = getRotationMatrix2D(pt, rotate, 1.0);
		cv::Mat rbMat = getRotationMatrix2D(pt, -rotate, 1.0);
		warpAffine(roi_Image, rotated_Image, rMat, cv::Size(roi_Image.cols, roi_Image.rows));

		// Detect face, eyes and pupils, save information
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
		vector<cv::Rect>::const_iterator faceItr = rawFaces.begin();
		vector<cv::Rect>::const_iterator eyeItr = rawEyes.begin();
		vector<cv::Point>::const_iterator pupilItr = rawPupils.begin();

		for (int j = 0; j < numFaces; ++j) {
			cv::Rect tmpFace = cv::Rect((*faceItr).x + roi_lt_point.x, (*faceItr).y + roi_lt_point.y, (*faceItr).width, (*faceItr).height);
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
				resultEyes.push_back(cv::Rect((*eyeItr).x + roi_lt_point.x, (*eyeItr).y + roi_lt_point.y, (*eyeItr).width, (*eyeItr).height));
				++eyeItr;
				resultPupils.push_back(cv::Point((*pupilItr).x + roi_lt_point.x, (*pupilItr).y + roi_lt_point.y));
				++pupilItr;
			}

			// Fill detected faces with black - to avoid the waste time of detecting the same face
			// cv::Mat blackMask = cv::Mat(tmpFace.width, tmpFace.height, CV_8UC1, 0.0);
			// cv::Mat aux = roi_Image.colRange(tmpFace.x, tmpFace.x + tmpFace.width).rowRange(tmpFace.y, tmpFace.y + tmpFace.height);
			// blackMask.copyTo(aux);
		}

		rawFaces.clear();
		rawEyes.clear();
		rawPupils.clear();
	}

	return;
}

void EyeDetector::ClearInfo() {
	resultFaces.clear();
	resultEyes.clear();
	resultPupils.clear();
	roi_lt_point = cv::Point(INT_MAX, INT_MAX);
	roi_rb_point = cv::Point(0, 0);
}

void EyeDetector::DrawDetectedInfo(cv::Mat& colorImg) {
	size_t numFaces = resultFaces.size();
	size_t numEyes = resultEyes.size();
	
	vector<cv::Rect>::const_iterator faceItr = resultFaces.begin();
	vector<cv::Rect>::const_iterator eyeItr = resultEyes.begin();
	vector<cv::Point>::const_iterator pupilItr = resultPupils.begin();

	for (int i = 0; i < numFaces; ++i) {
		cv::Point center;
		cv::Scalar color = colors[i % 8];
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

bool EyeDetector::IsFaceOverlap(cv::Rect& newFace) {

	for (vector<cv::Rect>::const_iterator faceItr = resultFaces.begin(); faceItr != resultFaces.end(); ++faceItr)
	{
		// If one rectangle is on left side of other
		if (newFace.x > (*faceItr).x + (*faceItr).width || (*faceItr).x > newFace.x + newFace.width) continue;
		// If one rectangle is above other
		if (newFace.y > (*faceItr).y + (*faceItr).height || (*faceItr).y > newFace.y + newFace.height) continue;
		return true;
	}
	return false;
}

void EyeDetector::rotateBackRawInfo(cv::Mat& rbMat) {
	size_t numFaces = rawFaces.size();
	vector<cv::Rect>::iterator faceItr = rawFaces.begin();
	vector<cv::Rect>::iterator eyeItr = rawEyes.begin();
	vector<cv::Point>::iterator pupilItr = rawPupils.begin();

	for (int i = 0; i < numFaces; ++i) {
		cv::Point faceLT((*faceItr).x, (*faceItr).y);
		cv::Point faceRB((*faceItr).x + (*faceItr).width, (*faceItr).y + (*faceItr).height);

		for (int j = 0; j < 2; ++j) {
			cv::Point eyeLT(faceLT.x + (*eyeItr).x, faceLT.y + (*eyeItr).y);
			cv::Point eyeRB(eyeLT.x + (*eyeItr).width, eyeLT.y + (*eyeItr).height);

			cv::Point pupil(eyeLT.x + (*pupilItr).x, eyeLT.y + (*pupilItr).y);
			pupil = rotateBackPoints(pupil, rbMat);
			(*pupilItr).x = pupil.x;
			(*pupilItr).y = pupil.y;

			eyeLT = rotateBackPoints(eyeLT, rbMat);
			eyeRB = rotateBackPoints(eyeRB, rbMat);
			(*eyeItr).x = eyeLT.x;
			(*eyeItr).y = eyeLT.y;
			(*eyeItr).width = eyeRB.x - eyeLT.x;
			(*eyeItr).height = eyeRB.y - eyeLT.y;

			++eyeItr;
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

cv::Point EyeDetector::rotateBackPoints(cv::Point srcPoint, cv::Mat& rbMat) {
	double np1 = srcPoint.x, np2 = srcPoint.y;
	double p1, p2;
	p1 = np1 * rbMat.at<double>(0, 0) + np2 * rbMat.at<double>(0, 1) + rbMat.at<double>(0, 2);
	p2 = np1 * rbMat.at<double>(1, 0) + np2 * rbMat.at<double>(1, 1) + rbMat.at<double>(1, 2);
	return cv::Point((int)p1, (int)p2);
}


const size_t EyeDetector::getFacesSize() {
	return resultFaces.size();
}

const size_t EyeDetector::getEyesSize() {
	return resultEyes.size();
}

const cv::Rect EyeDetector::getEyeLoc(int num) {
	assert(num <= resultEyes.size());
	return resultEyes[num];
}

const size_t EyeDetector::getPupilsSize() {
	return resultPupils.size();
}

const cv::Point EyeDetector::getPupilLoc(int num) {
	assert(num <= resultPupils.size());
	return resultPupils[num];
}

//////////////////////// eyeLike functions ////////////////////////////

// It returns eye center location offset, based on location of eyes.
cv::Point EyeDetector::findEyeCenter(cv::Mat& face, cv::Rect& eye) {
	cv::Mat eyeROIUnscaled = face(eye);
	cv::Mat eyeROI;

	scaleToFastSize(eyeROIUnscaled, eyeROI);

	//-- Find the gradient
	cv::Mat gradientX = computeMatXGradient(eyeROI);
	cv::Mat gradientY = computeMatXGradient(eyeROI.t()).t();

	//-- Normalize and threshold the gradient
	// compute all the magnitudes
	cv::Mat mags = matrixMagnitude(gradientX, gradientY);
	//compute the threshold
	double gradientThresh = computeDynamicThreshold(mags, kGradientThreshold);
	//double gradientThresh = kGradientThreshold;
	//double gradientThresh = 0;
	//normalize
	for (int y = 0; y < eyeROI.rows; ++y) {
		double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		const double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < eyeROI.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = Mr[x];
			if (magnitude > gradientThresh) {
				Xr[x] = gX / magnitude;
				Yr[x] = gY / magnitude;
			}
			else {
				Xr[x] = 0.0;
				Yr[x] = 0.0;
			}
		}
	}

	//-- Create a blurred and inverted image for weighting
	cv::Mat weight;
	GaussianBlur(eyeROI, weight, cv::Size(kWeightBlurSize, kWeightBlurSize), 0, 0);
	for (int y = 0; y < weight.rows; ++y) {
		unsigned char *row = weight.ptr<unsigned char>(y);
		for (int x = 0; x < weight.cols; ++x) {
			row[x] = (255 - row[x]);
		}
	}
	//imshow(debugWindow,weight);
	//-- Run the algorithm!
	cv::Mat outSum = cv::Mat::zeros(eyeROI.rows, eyeROI.cols, CV_64F);
	// for each possible gradient location
	// Note: these loops are reversed from the way the paper does them
	// it evaluates every possible center for each gradient location instead of
	// every possible gradient location for every center.
	for (int y = 0; y < weight.rows; ++y) {
		const double *Xr = gradientX.ptr<double>(y), *Yr = gradientY.ptr<double>(y);
		for (int x = 0; x < weight.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			if (gX == 0.0 && gY == 0.0) {
				continue;
			}
			testPossibleCentersFormula(x, y, weight, gX, gY, outSum);
		}
	}
	// scale all the values down, basically averaging them
	double numGradients = (weight.rows*weight.cols);
	cv::Mat out;
	outSum.convertTo(out, CV_32F, 1.0 / numGradients);
	//imshow("debugwindow",out);
	//-- Find the maximum point
	cv::Point maxP;
	double maxVal;
	cv::minMaxLoc(out, NULL, &maxVal, NULL, &maxP);

	cv::Point result = unscalePoint(maxP, eye);
	return result;
}

void EyeDetector::scaleToFastSize(const cv::Mat &src, cv::Mat &dst) {
	resize(src, dst, cv::Size(kFastEyeWidth, (((float)kFastEyeWidth) / src.cols) * src.rows));
}

cv::Mat EyeDetector::computeMatXGradient(const cv::Mat &Mat) {
	cv::Mat out(Mat.rows, Mat.cols, CV_64F);

	for (int y = 0; y < Mat.rows; ++y) {
		const uchar *Mr = Mat.ptr<uchar>(y);
		double *Or = out.ptr<double>(y);

		Or[0] = Mr[1] - Mr[0];
		for (int x = 1; x < Mat.cols - 1; ++x) {
			Or[x] = (Mr[x + 1] - Mr[x - 1]) / 2.0;
		}
		Or[Mat.cols - 1] = Mr[Mat.cols - 1] - Mr[Mat.cols - 2];
	}

	return out;
}

cv::Mat EyeDetector::matrixMagnitude(const cv::Mat &MatX, const cv::Mat &MatY) {
	cv::Mat mags(MatX.rows, MatX.cols, CV_64F);
	for (int y = 0; y < MatX.rows; ++y) {
		const double *Xr = MatX.ptr<double>(y), *Yr = MatY.ptr<double>(y);
		double *Mr = mags.ptr<double>(y);
		for (int x = 0; x < MatX.cols; ++x) {
			double gX = Xr[x], gY = Yr[x];
			double magnitude = sqrt((gX * gX) + (gY * gY));
			Mr[x] = magnitude;
		}
	}
	return mags;
}

double EyeDetector::computeDynamicThreshold(const cv::Mat &Mat, double stdDevFactor) {
	cv::Scalar stdMagnGrad, meanMagnGrad;
	cv::meanStdDev(Mat, meanMagnGrad, stdMagnGrad);
	double stdDev = stdMagnGrad[0] / sqrt(Mat.rows*Mat.cols);
	return stdDevFactor * stdDev + meanMagnGrad[0];
}

void EyeDetector::testPossibleCentersFormula(int x, int y, const cv::Mat &weight, double gx, double gy, cv::Mat &out) {
	// for all possible centers
	for (int cy = 0; cy < out.rows; ++cy) {
		double *Or = out.ptr<double>(cy);
		const unsigned char *Wr = weight.ptr<unsigned char>(cy);
		for (int cx = 0; cx < out.cols; ++cx) {
			if (x == cx && y == cy) {
				continue;
			}
			// create a vector from the possible center to the gradient origin
			double dx = x - cx;
			double dy = y - cy;
			// normalize d
			double magnitude = sqrt((dx * dx) + (dy * dy));
			dx = dx / magnitude;
			dy = dy / magnitude;
			double dotProduct = dx*gx + dy*gy;
			dotProduct = std::max(0.0, dotProduct);
			// square and multiply by the weight
			if (kEnableWeight) {
				Or[cx] += dotProduct * dotProduct * (Wr[cx] / kWeightDivisor);
			}
			else {
				Or[cx] += dotProduct * dotProduct;
			}
		}
	}
}

cv::Point EyeDetector::unscalePoint(cv::Point p, cv::Rect origSize) {
	float ratio = (((float)kFastEyeWidth) / origSize.width);
	int x = round(p.x / ratio);
	int y = round(p.y / ratio);
	return cv::Point(x, y);
}
