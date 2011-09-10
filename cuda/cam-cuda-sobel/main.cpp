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

//
// My hack to the SobelFilter example by Nvidia
// ~ Chris McClanahan
//

#include "CVcam.h"
#include <stdlib.h>

extern "C"
{
    extern void initCUDA(int w, int h);
    extern void stopCUDA(void);
    extern void runCUDASobel(unsigned char* imageData, float thresh, int iw, int ih);

    extern void initCUDA4(int w, int h);
    extern void stopCUDA4(void);
    extern void runCUDASobel4(unsigned char* imageData, float thresh, int iw, int ih);
}

int main(int argc, char** argv) {
    /*
     *
     */
    int DO_FLOAT4 = -1;
    int a = -1;
    if (argc > 1) {
        a = atoi(argv[1]);;
    }
    if (a == 1 || a == 3) {
        DO_FLOAT4 = (a == 3) ? 1 : 0;
    } else {
        printf("\nUsage: specify wether to use 1 or 3 channels\n\t./cam-cuda [1|3]\n\n");
        exit(0);
    }

    /*
     *
     */
    cvNamedWindow("raw", 0);
    cvNamedWindow("gray", 0);
    cvNamedWindow("cuda", 0);
    CVcam camera_usb;
    IplImage* gray;
    float thresh = 1.0f;
    float incr = 0.50f;
    printf("Press '[' or ']' to change threshold, or 'Esc' to exit\n");

    /*
     *
     */
    if (!camera_usb.connect()) {
        printf("Error getting camera connection \n");
        exit(-1);
    } else if (!camera_usb.GrabCvImage()) {
        printf("Error getting camera frame \n");
        exit(-1);
    }

    /*
     *
     */
    camera_usb.loadSettings();

    /*
     *
     */
    if (DO_FLOAT4) {
        initCUDA4(camera_usb.visCvRaw->width, camera_usb.visCvRaw->height);
        gray = cvCreateImage(cvSize(camera_usb.visCvRaw->width, camera_usb.visCvRaw->height), IPL_DEPTH_8U, 4);
    } else {
        initCUDA(camera_usb.visCvRaw->width, camera_usb.visCvRaw->height);
        gray = cvCreateImage(cvSize(camera_usb.visCvRaw->width, camera_usb.visCvRaw->height), IPL_DEPTH_8U, 1);
    }

    /*
     *
     */
    char key = (char)cvWaitKey(10);
    while (key != 27) {   // Esc to exit
        /*
         *
         */
        if (!camera_usb.GrabCvImage()) {
            printf("Error getting camera frame \n");
        } else {
            cvShowImage("raw", camera_usb.visCvRaw);
        }

        /*
         *
         */
        IplImage* img = camera_usb.visCvRaw;
        if (DO_FLOAT4) {
            cvCvtColor(img, gray, CV_BGR2BGRA);
            runCUDASobel4((unsigned char*)gray->imageData, thresh, img->width, img->height);
        } else {
            cvCvtColor(img, gray, CV_BGR2GRAY);
            cvShowImage("gray", gray);
            runCUDASobel((unsigned char*)gray->imageData, thresh, img->width, img->height);
        }
        cvShowImage("cuda", gray);

        /*
         *
         */
        key = cvWaitKey(10);
        if (key == '[') {
            thresh -= incr;
            printf("thresh: %f \n", thresh);
        }
        if (key == ']') {
            thresh += incr;
            printf("thresh: %f \n", thresh);
        }

    }


    /*
     *
     */
    cvDestroyAllWindows();
    cvReleaseImage(&camera_usb.visCvRaw);
    cvReleaseImage(&gray);
    if (DO_FLOAT4) {
        stopCUDA4();
    } else {
        stopCUDA();
    }

    return 0;
}

