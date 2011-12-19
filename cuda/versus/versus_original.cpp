// place the containing folder in your
//  libjacket/examples/ directory
//
// Chris McClanahan - 2011


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <jacket.h>
#include <jacket_gfx.h>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include "opencv2/gpu/gpu.hpp"
#include "timer.h"


using namespace jkt;
using namespace std;
using namespace cv;
using namespace gpu;


const int ksz = 3;
const float jktmk1[] = {
    1.0, 2.0, 1.0,
};
const float jktmk2[] = {
    1.0, 0.0, -1.0,
};
float cvmk[ksz][ksz] = {
    { -2, -1,  0, },
    { -1,  0,  1, },
    {  0,  1,  2, },
};
Mat cvk = Mat(ksz, ksz, CV_32F, cvmk);

void gen(Mat& mat, int rows, int cols, int type, Scalar low, Scalar high) {
    mat.create(rows, cols, type);
    RNG rng(0);
    rng.fill(mat, RNG::UNIFORM, low, high);
}

const int use_cv_sobel = 1;

int main() {

    // setup
    Mat src, dst;
    GpuMat d_src, d_dst, d_cvk;
    int runs = 100;

    // bench
    for (int size = 512; size < 4000; size += 512) {

        cout  << "size: " << size << "x" << size << endl;
        gen(src, size, size, CV_32FC1, 0, 1);

        try {

            // opencv
            {
                dst.create(size, size, CV_32FC1);
                d_src = src;
                d_dst.create(size, size, CV_32FC1);
                d_cvk = cvk;

                // // cpu
                // Sobel(src, dst, dst.depth(), 1, 1);
                // start_timer(0);
                // for (int i = 0; i < runs; ++i) {
                //     Sobel(src, dst, dst.depth(), 1, 1);
                // }
                // cout  << "  cv-cpu: " << elapsed_time(0) / (float)runs << endl;

                // gpu
                if (use_cv_sobel) {
                    gpu::Sobel(d_src, d_dst, d_dst.depth(), 1, 1, ksz);
                    start_timer(1);
                    for (int i = 0; i < runs; ++i) {
                        gpu::Sobel(d_src, d_dst, d_dst.depth(), 1, 1, ksz);
                    }
                } else {
                    convolve(d_src, d_cvk, d_dst);
                    start_timer(1);
                    for (int i = 0; i < runs; ++i) {
                        convolve(d_src, d_cvk, d_dst);
                    }
                }
                cout << "  cv-gpu: " << elapsed_time(1) / (float)runs << endl;
            }

            // jacket
            {
                // extract cv image
                Mat jimg;
                src.convertTo(jimg, CV_32FC1);
                float* fgray = (float*)jimg.data;
                f32 I1 = f32(fgray, jimg.rows, jimg.cols);
                unsigned dimsb[] = {ksz, ksz};
                // gpu
                //f32 jdst = conv2(I1, sobel_k, jktConvSame);
                f32 jdst = conv2(ksz, jktmk1, ksz, jktmk2, I1, jktConvSame);
                gsync();
                start_timer(2);
                for (int i = 0; i < runs; ++i) {
                    //jdst = conv2(I1, sobel_k, jktConvSame);
                    jdst = conv2(ksz, jktmk1, ksz, jktmk2, I1, jktConvSame);
                    gsync(); // disables smart caching
                }
                cout << "  jacket: " << elapsed_time(2) / (float)runs << endl;
            }

        } catch (gexception& e) {
            cout << e.what() << endl;
        }

    } // loop

    return 0;
}
