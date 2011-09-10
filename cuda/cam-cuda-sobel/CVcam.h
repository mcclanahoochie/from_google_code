/*
   Copyright [2011] [Chris McClanahan]

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef CVCAM_H_
#define CVCAM_H_

/*
 * This file is an interface to a USB camera, using OpenCV
 *   by Chris McClanahan
 *
 */

#include <stdio.h>
#include <highgui.h>
#include <cv.h>




class CVcam {
public:
    // constructor
    CVcam();
    // destructor
    virtual ~CVcam();
    // the connection to the camera
    CvCapture* capture;
    // flag for init capture connection
    int camconnected;

    // returns 1 if connected to camera capture
    int isValid();
    // grabs raw opencv image (into global visCvRaw)
    int GrabCvImage();
    // grabs raw image and converts it into (global) visRaw
    void GrabBuffer2DImage();
    // sets up the camera capture object
    int connect(int deviceID = 0, const char* filename = NULL);
    // starts image grab thread (not currently used)
    void startImageGrabThread(void* obj);

    // loads values/settings for camera
    void loadSettings();

    // testing usb camera functions (not currently used)
    int testwebcamandconvertloop();

    // the image
    IplImage* visCvRaw;


};

#endif /*CVCAM_H_*/
