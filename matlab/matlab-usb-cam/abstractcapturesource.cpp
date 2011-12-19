/*
 * abstractcapturesource.cpp
 *
 *  Created on: 25-mag-2009
 *      Author: seide
 */

#include "abstractcapturesource.h"
#include <iostream>
#include <cv.h>
#include <highgui.h>
AbstractCaptureSource::AbstractCaptureSource() {
}

AbstractCaptureSource::~AbstractCaptureSource() {
    cvReleaseCapture(&(this->capture));
}
int AbstractCaptureSource::getDepth() {return this->depth;}
int AbstractCaptureSource::getHeight() {return this->height;}
int AbstractCaptureSource::getWidth() {return this->width;}

void AbstractCaptureSource::convertFrame(unsigned char* video_ptr, IplImage* img, bool color) {
    if (!color) {
        IplImage* img_gray = cvCreateImage(cvSize(this->getWidth(), this->getHeight()), this->getDepth(), 1);
        IplImage* img_gray_transposed = cvCreateImage(cvSize(this->getHeight(), this->getWidth()), this->getDepth(), 1);
        cvCvtColor(img, img_gray, CV_RGB2GRAY);
        cvTranspose(img_gray, img_gray_transposed);
        //copy 1 frame to video_ptr(output argument of matlab function)
        memcpy((void*)video_ptr, img_gray_transposed->imageData, this->getHeight()*this->getWidth()*sizeof(unsigned char));
        cvReleaseImage(&img_gray);
        cvReleaseImage(&img_gray_transposed);
    } else {
        IplImage* img_transposed = cvCreateImage(cvSize(this->getHeight(), this->getWidth()), this->getDepth(), 3);
        cvTranspose(img, img_transposed);
        //copy 1 frame to video_ptr(output argument of matlab function)
        IplImage* img_r = cvCreateImage(cvSize(this->getHeight(), this->getWidth()), this->getDepth(), 1);
        IplImage* img_g = cvCreateImage(cvSize(this->getHeight(), this->getWidth()), this->getDepth(), 1);
        IplImage* img_b = cvCreateImage(cvSize(this->getHeight(), this->getWidth()), this->getDepth(), 1);

        cvSplit(img_transposed, img_b, img_g, img_r, NULL);
        int size = this->getHeight() * this->getWidth();
        memcpy((void*)(video_ptr), img_r->imageData, this->getHeight()*this->getWidth()*sizeof(unsigned char));
        memcpy((void*)(video_ptr + size), img_g->imageData, this->getHeight()*this->getWidth()*sizeof(unsigned char));
        memcpy((void*)(video_ptr + 2 * size), img_b->imageData, this->getHeight()*this->getWidth()*sizeof(unsigned char));

        cvReleaseImage(&img_transposed);

    }
}
IplImage* AbstractCaptureSource::getNextFrame() {
    if (cvGrabFrame(this->capture))
    { return cvRetrieveFrame(this->capture); }
    else
    { return NULL; }
}
