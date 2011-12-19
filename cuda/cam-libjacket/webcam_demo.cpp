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


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <jacket.h>
#include <jacket_gfx.h>
#include "ppm_utils.h"
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

using namespace jkt;
using namespace std;
using namespace cv;

void jacket_img_test_demo(Mat& img) {


    // convolution kernels
    const float h_avg_kernel[] = { 1.0 / 12.0, 2.0 / 12.0, 1.0 / 12.0,
                                   2.0 / 12.0, 0.0       , 2.0 / 12.0,
                                   1.0 / 12.0, 2.0 / 12.0, 1.1 / 12.0
                                 };
    const float h_sobel_kernel[] = { -2.0, -1.0,  0.0,
                                     -1.0,  0.0,  1.0,
                                     0.0,  1.0,  2.0
                                   };
    f32 avg_k   = f32(h_avg_kernel,   3, 3);
    f32 sobel_k = f32(h_sobel_kernel, 3, 3);


    // extract cv image
    Mat mgray(img.rows, img.cols, CV_8UC1);
    cvtColor(img.t(), mgray, CV_BGR2GRAY);
    mgray.convertTo(mgray, CV_32FC1);
    float* fgray = (float*)mgray.data;

    // copy to gpu
    f32 I1 = f32(fgray, img.rows, img.cols);

    // display window
    figure();
    colormap("gray");
    subplot(3, 3, 1); imagesc(I1);                 title("source image");

    // image morphology
    subplot(3, 3, 2); imagesc(erode(I1,  avg_k));  title("erode");
    subplot(3, 3, 3); imagesc(dilate(I1, avg_k));  title("dilate");

    // binary image morphology
    b8 Ib = I1 < 255 / 3;
    subplot(3, 3, 6);  imagesc(f32(bwmorph(Ib, JKT_BWM_Open)));  title("bwmorph-open");
    subplot(3, 3, 9);  imagesc(f32(bwmorph(Ib, JKT_BWM_Close))); title("bwmorph-close");

    // image convolution
    f32 iedge = abs(filter2D(I1, sobel_k));
    subplot(3, 3, 4); imagesc(iedge);               title("edge detection");
    subplot(3, 3, 7); imagesc(iedge > 255 / 3);     title("edge thresh");

    // image histogram
    f32 ihist = hist(I1, 256);
    subplot(3, 3, 8);  plot(f32(seq(255)), ihist(seq(255))); title("image histogram");

    // image histogram equalization
    f32 inorm = histeq(I1, ihist);
    subplot(3, 3, 5); imagesc(inorm);              title("image HistEq");

    // refresh
    drawnow();
}

int main() {

    // camera setup
    Mat cam_img;
    VideoCapture  capture;
    capture.open(0); //try to open
    if (!capture.isOpened()) { //if this fails...
        cerr << "open a video device fail\n" << endl;
        return 0;
    }

    // image setup
    capture >> cam_img;
    if (cam_img.empty()) {
        cout << "load image fail " << endl;
        return 0;
    }
    namedWindow("cam", CV_WINDOW_KEEPRATIO);
    imshow("cam", cam_img);
    printf(" img = %d x %d \n", cam_img.rows, cam_img.cols);

    // process loop
    while (1) {

        // grab frame
        capture >> cam_img;
        imshow("cam", cam_img);

        try {
            // process
            jacket_img_test_demo(cam_img);

        } catch (gexception& e) {
            cout << e.what() << endl;
        }

    }
    return 0;
}
