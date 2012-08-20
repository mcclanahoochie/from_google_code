/*
   Copyright [2012] [Chris McClanahan]

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
#include <arrayfire.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace af;
using namespace std;
using namespace cv;

// trackbar
#define SLIDER_MAX 5
int lce_val = 1;

// mem layout for gpu
void mat_to_array(cv::Mat& input, array& output) {
    input.convertTo(input, CV_32FC3); // floating point
    const unsigned size = input.rows * input.cols;
    const unsigned w = input.cols;
    const unsigned h = input.rows;
    float r[size];
    float g[size];
    float b[size];
    int tmp = 0;
    for (unsigned i = 0; i < h; i++) {
        for (unsigned j = 0; j < w; j++) {
            Vec3f ip = input.at<Vec3f>(i, j);
            tmp = j * h + i; // convert to column major
            r[tmp] = ip[2];
            g[tmp] = ip[1];
            b[tmp] = ip[0];
        }
    }
    output = join(2,
                  array(h, w, r),
                  array(h, w, g),
                  array(h, w, b))/255.f; // merge, set range [0-1]
}

// 5x5 gaussian blur with sigma 3
float h_blur5[] = {
     0.0318,    0.0375,    0.0397,    0.0375,    0.0318
,    0.0375,    0.0443,    0.0469,    0.0443,    0.0375
,    0.0397,    0.0469,    0.0495,    0.0469,    0.0397
,    0.0375,    0.0443,    0.0469,    0.0443,    0.0375
,    0.0318,    0.0375,    0.0397,    0.0375,    0.0318
};
array blur5 = array(5, 5, h_blur5);
void gaussian_5x5(array& in, array& out){
    out = filter(in, blur5);
}

// core
void process_image(Mat& cam_img) {

    // extract cv image, copy to gpu
    array img;
    mat_to_array(cam_img, img);

    // get V of hsv colorspace
    array hsv = rgbtohsv(img);
    array v = hsv(span,span,2); // value

    // LCE v (unsharp v)
    array blurred;
    gaussian_5x5(v, blurred); // unsharp
    const float amount = lce_val;
    array sharp = v * (1+amount) + blurred * (-amount); // enhance!
    sharp = min(1,sharp); // clamp
    sharp = max(0,sharp);
    array lce_v = join(2,
                       hsv(span,span,0),
                       hsv(span,span,1),
                       sharp); // channel merge

    // convert color back to rgb
    array lce_img = hsvtorgb(lce_v);

    // display original & local contrast enhanced
    subfigure(2, 2, 1);  rgbplot(img);           title("Source");
    subfigure(2, 2, 3);  rgbplot(lce_img);       title("LCE");

    // v & vsharp histograms
    array vhist = histogram(v, 256);
    array shist = histogram(sharp, 256);

    // display v & vsharp histogram plots
    subfigure(2, 2, 2);  plot(seq(255), vhist/max<float>(vhist));  title("v hist");
    subfigure(2, 2, 4);  plot(seq(255), shist/max<float>(shist));  title("v sharp hist");

    // refresh
    draw();
}

// start
int main() {

    // camera setup
    Mat cam_img;
    VideoCapture  capture;
    capture.open(0); //try to open
    if (!capture.isOpened()) { //if this fails...
        cerr << "open a video device fail\n" << endl;
        return 0;
    }

    // opencv image setup
    capture >> cam_img;
    if (cam_img.empty()) {
        cout << "load image fail " << endl;
        return 0;
    }
    printf(" img = %d x %d \n", cam_img.cols, cam_img.rows);

    // slider bar
    namedWindow("trackbar", CV_WINDOW_KEEPRATIO); cvMoveWindow("trackbar", 10, 10);
    imshow("trackbar", Mat(40, 340, CV_8UC1));
    createTrackbar("adjustment", "trackbar", &lce_val, SLIDER_MAX, 0);

    // gpu warmup
    randu(1);
    info();
    figure();

    // process loop
    while (1) {

        // timing
        timer start = timer::tic();

        // grab frame
        capture >> cam_img;

        try {
            // process
            process_image(cam_img);

        } catch (af::exception& e) {
            cout << e.what() << endl;
        }

        // timing
        printf("  demo fps:\t %f \n", 1.f / (timer::toc(start)));

        // exit key
        char key;
        key = (char) cvWaitKey(10);
        if (key == 27 || key == 'q' || key == 'Q') { break; }
    }

    return 0;
}
