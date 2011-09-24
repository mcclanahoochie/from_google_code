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


const float h_sobel_kernel[] = { -2.0, -1.0,  0.0,
                                 -1.0,  0.0,  1.0,
                                 0.0,  1.0,  2.0
                               };
f32 sobel_k = f32(h_sobel_kernel, 3, 3);


void gen(Mat& mat, int rows, int cols, int type, Scalar low, Scalar high) {
    mat.create(rows, cols, type);
    RNG rng(0);
    rng.fill(mat, RNG::UNIFORM, low, high);
}


int main() {

    // setup
    Mat src, dst;
    GpuMat d_src, d_dst;
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

                // cpu
                Sobel(src, dst, dst.depth(), 1, 1);
                start_timer(0);
                for (int i = 0; i < runs; ++i) {
                    Sobel(src, dst, dst.depth(), 1, 1);
                }
                cout  << "  cv-cpu: " << elapsed_time(0) / (float)runs << endl;

                // gpu
                gpu::Sobel(d_src, d_dst, d_dst.depth(), 1, 1);
                start_timer(1);
                for (int i = 0; i < runs; ++i) {
                    gpu::Sobel(d_src, d_dst, d_dst.depth(), 1, 1);
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

                // gpu
                f32 jdst = conv2(I1, sobel_k, jktConvSame);
                gsync();
                start_timer(2);
                for (int i = 0; i < runs; ++i) {
                    jdst = conv2(I1, sobel_k, jktConvSame);
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
