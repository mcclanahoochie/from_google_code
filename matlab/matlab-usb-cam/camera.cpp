/*
 * camera.cpp
 *
 *  Created on: 25-mag-2009
 *      Author: seide
 */

#include "camera.h"
#include <iostream>
Camera::Camera(int deviceNumber) {
    this->number = deviceNumber;
    this->capture = cvCaptureFromCAM(this->number);
    if (!capture) {
        std::cout << "cannot open device " << this->number << std::endl;
        return;
    }
    IplImage* frame = 0;
    if (cvGrabFrame(capture)) {
        frame = cvRetrieveFrame(capture);
        this->width = frame->width;
        this->height = frame->height;
        this->depth = frame->depth;

    }


}

