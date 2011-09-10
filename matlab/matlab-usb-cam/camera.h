/*
 * camera.h
 *
 *  Created on: 25-mag-2009
 *      Author: seide
 */

#ifndef CAMERA_H_
#define CAMERA_H_
#include "abstractcapturesource.h"
#include <highgui.h>

class Camera: public AbstractCaptureSource {
public:
    Camera(int deviceNumber);
protected:
    int number;
};

#endif /* CAMERA_H_ */
