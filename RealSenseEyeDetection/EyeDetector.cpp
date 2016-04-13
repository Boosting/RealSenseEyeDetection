#include "EyeDetector.h"

#define CASCADE_FACE_XML_LOCATION "D:/Lib/OpenCV/build/etc/haarcascades/haarcascade_frontalface_default.xml"
#define CASCADE_EYE_XML_LOCATION "D:/Lib/OpenCV/build/etc/haarcascades/haarcascade_eye_tree_eyeglasses.xml"

EyeDetector::EyeDetector()
{
	scale = 1.0;
	tryflip = false;
	// cascade.load(CASCADE_FACE_XML_LOCATION);
	// nestedCascade.load(CASCADE_EYE_XML_LOCATION);

	if (!cascade.load(CASCADE_FACE_XML_LOCATION) || !nestedCascade.load(CASCADE_EYE_XML_LOCATION))
	{
		std::cout << "ERROR: Could not load Face / Eye classifier cascade" << std::endl;
	}
}

EyeDetector::~EyeDetector()
{
}

void EyeDetector::CascadeDetection(cv::Mat& colorImg) {
	double t = (double)cvGetTickCount();

	int i = 0;
	std::vector<cv::Rect> faces;
	const static cv::Scalar colors[] = { CV_RGB(0,0,255),
		CV_RGB(0,128,255),
		CV_RGB(0,255,255),
		CV_RGB(0,255,0),
		CV_RGB(255,128,0),
		CV_RGB(255,255,0),
		CV_RGB(255,0,0),
		CV_RGB(255,0,255) };
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
		cv::Point center;
		cv::Scalar color = colors[i % 8];
		int radius;

		double aspect_ratio = (double)r->width / r->height;
		if (0.75 < aspect_ratio && aspect_ratio < 1.3)
		{
			center.x = cvRound((r->x + r->width*0.5)*scale);
			center.y = cvRound((r->y + r->height*0.5)*scale);
			radius = cvRound((r->width + r->height)*0.25*scale);
			circle(colorImg, center, radius, color, 3, 8, 0);
		}
		else
			rectangle(colorImg, cvPoint(cvRound(r->x*scale), cvRound(r->y*scale)),
				cvPoint(cvRound((r->x + r->width - 1)*scale), cvRound((r->y + r->height - 1)*scale)),
				color, 3, 8, 0);
		
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

		for (std::vector<cv::Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++)
		{
			center.x = cvRound((r->x + nr->x + nr->width*0.5)*scale);
			center.y = cvRound((r->y + nr->y + nr->height*0.5)*scale);
			radius = cvRound((nr->width + nr->height)*0.25*scale);
			circle(colorImg, center, radius, color, 3, 8, 0);
		}
	}
	t = (double)cvGetTickCount() - t;
	printf("detection time = %g ms\n", t / ((double)cvGetTickFrequency()*1000.));

	return;
}

void EyeDetector::CascadeDetection(cv::Mat& colorImg, cv::Mat& depth_to_color_img, const uint16_t one_meter) {
	
	CvSize originalImage, newROI;
	originalImage = depth_to_color_img.size();
	CvPoint pointRB(0, 0), pointLT(originalImage.width-1, originalImage.height-1);

	// Depth images need to be smoothed to cancel noise, which may dramatically expand ROI
	cv::medianBlur(depth_to_color_img, depth_to_color_img, 5);

	// Resize image ROI, objects only in 50cm~300cm are detected
	for (int w = 0; w < originalImage.width; ++w) {
		for (int h = 0; h < originalImage.height; ++h) {
			uint16_t val = depth_to_color_img.at<uint16_t>(h, w);
			if (val < one_meter/2 || val>3*one_meter) continue;
			if (h < pointLT.y) pointLT.y = h;
			else if (h > pointRB.y) pointRB.y = h;
			if (w < pointLT.x) pointLT.x = w;
			else if (w > pointRB.x) pointRB.x = w;
		}
	}

	// If nothing appears, return
	if (pointRB.x <= pointLT.x || pointRB.y <= pointLT.y) return;
	rectangle(colorImg, pointLT, pointRB, CV_RGB(255, 0, 0), 3, 8, 0);

}