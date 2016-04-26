#include "PupilLocator.h"



PupilLocator::PupilLocator()
{
}


PupilLocator::~PupilLocator()
{
}


void PupilLocator::AddEye(Rect detectedEye) {
	detectedEyes.push_back(detectedEye);
}


void PupilLocator::DetectPupils(Mat& depthImg) {

}

void PupilLocator::DrawPupils(Mat& depthImg) {

}

void PupilLocator::ClearInfo() {
	detectedEyes.clear();
}

void PupilLocator::DrawEyes(Mat& depthImg) {
	if (detectedEyes.empty()) return;
	for (vector<Rect>::const_iterator eyeItr = detectedEyes.begin(); eyeItr != detectedEyes.end(); ++eyeItr) {
		Point center;
		Scalar color = colors[0];
		int radius;

		center.x = cvRound((eyeItr->x + eyeItr->width*0.5)*scale);
		center.y = cvRound((eyeItr->y + eyeItr->height*0.5)*scale);
		radius = cvRound((eyeItr->width + eyeItr->height)*0.25*scale);
		circle(depthImg, center, radius, color, 3, 8, 0);
	}
}

