/*
 * abstractcapturesource.h
 *
 *  Created on: 25-mag-2009
 *      Author: seide
 */

#ifndef ABSTRACTCAPTURESOURCE_H_
#define ABSTRACTCAPTURESOURCE_H_
#include <cv.h>
#include <highgui.h>

class AbstractCaptureSource {
public:
    AbstractCaptureSource();
    virtual ~AbstractCaptureSource();
    int getDepth();
    int getHeight();
    int getWidth();
    virtual IplImage* getNextFrame();
    void convertFrame(unsigned char* video_ptr, IplImage* img, bool color);
    CvCapture* capture;
protected:
    int depth, width, height;

};

#endif /* ABSTRACTCAPTURESOURCE_H_ */
