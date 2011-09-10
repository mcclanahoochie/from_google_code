#include "video.h"
#include <highgui.h>
#include <iostream>
using namespace std;
Video::Video(const char* filename) {
    IplImage* frame = 0;
    this->name = filename;
    this->capture = cvCreateFileCapture(filename);
    this->frames = 0;

    if (!capture) {
        cout << "cannot open " << filename << endl;
        return;
    }
    while (cvGrabFrame(capture)) {
        frame = cvRetrieveFrame(capture);
        this->width = frame->width;
        this->height = frame->height;
        this->depth = frame->depth;
        this->frames++;
    }
    rewind();
}
Video::Video(const char* filename, int frames) {
    IplImage* frame = 0;
    this->name = filename;
    this->capture = cvCreateFileCapture(filename);
    this->frames = frames;

    if (!capture) {
        cout << "cannot open " << filename << endl;
        return;
    }
    if (cvGrabFrame(capture)) {
        frame = cvRetrieveFrame(capture);
        this->width = frame->width;
        this->height = frame->height;
        this->depth = frame->depth;
    }
    rewind();
}

Video::~Video() {
    cvReleaseCapture(&this->capture);
}
void Video::rewind() {
    cvReleaseCapture(&this->capture);
    this->capture = cvCaptureFromAVI(this->name);
}

