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
                  array(h, w, b)) / 255.f; // merge, set range [0-1]
}

// edge kernel
const float h_sobel_kernel[] = { -2.0, -1.0,  0.0,
                                 -1.0,  0.0,  1.0,
                                 0.0,  1.0,  2.0
                               };
array sobel_k = array(3, 3, h_sobel_kernel);
void sobel(array& in, array& out) {
    out = medfilt(abs(convolve(in, sobel_k, af_conv_same)));
}

// core
array process_image(Mat& cur_img, array& prev_img) {

    // extract cv image, copy to gpu
    array curr_img;
    mat_to_array(cur_img, curr_img);
    array img = curr_img;

    // get V of hsv colorspace
    array hsv = rgbtohsv(img);
    array v = hsv(span, span, 2) * 255.f; // value

    // histograms
    const int nbins = 256;
    array veq_h = histogram(v, nbins);

    // edges
    array edges;
    sobel(v, edges);

    // fram differences
    array diff = abs(curr_img - prev_img);

    // meanshift
    array m = meanshift(resize(img, 0.5), 4.4, 0.1);

    // display
    subfigure(3, 2, 4);  rgbplot(img);              title("Source");
    subfigure(3, 2, 6);  rgbplot(m);                title("Meanshift");
    subfigure(3, 2, 3);  rgbplot(diff);             title("Motion");
    subfigure(3, 2, 1);  imgplot(edges);            title("Edges");
    subfigure(3, 2, 5);  plot(veq_h(seq(nbins - 1))); title("Histogram");

    // refresh
    draw();

    // previous frame
    return curr_img;
}

// start
int main(int argc, char* argv[]) {
    int device = 0;
    if (argc > 1) { device = atoi(argv[1]); }

    // camera setup
    Mat cam_img, pre_img;
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

    // init
    deviceset(device);
    randu(1);
    info();
    figure();
    palette("orange");

    // logo
    array logo  = loadimage("aflogo.jpg", true) / 255.f; // 3 channel RGB [0-1]
    subfigure(3, 2, 2);   rgbplot(logo);   title("Powered By:");

    // grab frame
    capture >> pre_img;
    array prev_img;
    mat_to_array(pre_img, prev_img);

    // process loop
    while (1) {

        // timing
        timer start = timer::tic();

        // grab frame
        capture >> cam_img;

        // process
        try {
            prev_img = process_image(cam_img, prev_img);
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
