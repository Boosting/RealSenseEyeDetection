#include "PupilLocator.h"



PupilLocator::PupilLocator()
{
}


PupilLocator::~PupilLocator()
{
}


void PupilLocator::AddPupil(Point detectedPupil) {
	detectedPupils.push_back(detectedPupil);
}

void PupilLocator::ClearInfo() {
	detectedPupils.clear();
	pupilDepthPoints.clear();
	pupilReferencePoints.clear();
}

void PupilLocator::DrawPupils(Mat& depthImg) {
	if (pupilDepthPoints.empty()) return;
	
	for (vector<Point>::const_iterator pupilItr = pupilDepthPoints.begin(); pupilItr != pupilDepthPoints.end(); ++pupilItr) {
		Point center = (*pupilItr);
		int radius = 2;
		circle(depthImg, center, radius, color, 3, 8, 0);
	}
}

// return false: couldn't find any reference points for some pupils
bool PupilLocator::FindPupilDepthPoints(Mat& convert_imgX, Mat& convert_imgY) {
	for (vector<Point>::const_iterator pupilItr = detectedPupils.begin(); pupilItr != detectedPupils.end(); ++pupilItr) {
		Point pupil = *pupilItr;
		int dist;
		for (dist = 0; dist < max_reference_search_pixels; ++dist) {
			if (convert_imgX.at<uint16_t>(pupil.y, pupil.x + dist) > 0 && convert_imgX.at<uint16_t>(pupil.y, pupil.x - dist) > 0 && 
				convert_imgY.at<uint16_t>(pupil.y, pupil.x + dist) > 0 && convert_imgY.at<uint16_t>(pupil.y, pupil.x - dist) > 0) {
				Point a(convert_imgX.at<uint16_t>(pupil.y, pupil.x - dist), convert_imgY.at<uint16_t>(pupil.y, pupil.x - dist));
				Point b(convert_imgX.at<uint16_t>(pupil.y, pupil.x + dist), convert_imgY.at<uint16_t>(pupil.y, pupil.x + dist));
				pupilDepthPoints.push_back(Point((a.x + b.x) / 2, (a.y + b.y) / 2));
				pupilReferencePoints.push_back(a);
				pupilReferencePoints.push_back(b);
				break;
			}
			else if (convert_imgX.at<uint16_t>(pupil.y + dist, pupil.x) > 0 && convert_imgX.at<uint16_t>(pupil.y - dist, pupil.x) > 0 &&
				     convert_imgY.at<uint16_t>(pupil.y + dist, pupil.x) > 0 && convert_imgY.at<uint16_t>(pupil.y - dist, pupil.x) > 0) {
				Point a(convert_imgX.at<uint16_t>(pupil.y - dist, pupil.x), convert_imgY.at<uint16_t>(pupil.y - dist, pupil.x));
				Point b(convert_imgX.at<uint16_t>(pupil.y + dist, pupil.x), convert_imgY.at<uint16_t>(pupil.y + dist, pupil.x));
				pupilDepthPoints.push_back(Point((a.x + b.x) / 2, (a.y + b.y) / 2));
				pupilReferencePoints.push_back(a);
				pupilReferencePoints.push_back(b);
				break;
			}
		}
		if (dist == max_reference_search_pixels)
			return false;
	}
	return true;
}

const size_t PupilLocator::getPupilDepthSize() {
	return pupilDepthPoints.size();
}

const Point PupilLocator::getPupilDepthLoc(int num) {
	assert(num <= pupilDepthPoints.size());
	return pupilDepthPoints[num];
}

const size_t PupilLocator::getPupilReferenceSize() {
	return pupilReferencePoints.size();
}

const Point PupilLocator::getPupilReferenceLoc(int num) {
	assert(num <= pupilReferencePoints.size());
	return pupilReferencePoints[num];
}