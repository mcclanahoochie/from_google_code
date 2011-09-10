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

#include "CVcam.h"


CVcam::CVcam() {
    capture = 0;
    camconnected = 0;
}

CVcam::~CVcam() {
    if (capture) {
        cvReleaseCapture(&capture);
    }
}

int CVcam::connect(int deviceID, const char* filename) {
    if (filename == NULL) {
        capture = cvCaptureFromCAM(deviceID);
    } else {
        capture = cvCreateFileCapture(filename);
    }
    xxcl
    if (capture) {
        camconnected = 1;
        return 1;
    } else {
        camconnected = 0;
        //return 0;
        printf("Error connecting to camera \n");
        exit(-1);
    }
}

int CVcam::isValid() {
    return camconnected;
}

int CVcam::GrabCvImage() {
    if (!cvGrabFrame(capture)) {
        return 0;
    }

    // get and set
    visCvRaw = cvRetrieveFrame(capture);

    return 1;
}

void CVcam::loadSettings() {
    // currently no settings for USB camera
//	printf("No USB Camera settings available \n");
    printf("setting: %f \n",  cvGetCaptureProperty(capture, CV_CAP_PROP_BRIGHTNESS));
    printf("setting: %d \n",  cvSetCaptureProperty(capture, CV_CAP_PROP_BRIGHTNESS, 0.6)); // 0 - 1
    printf("setting: %f \n",  cvGetCaptureProperty(capture, CV_CAP_PROP_BRIGHTNESS));

}

