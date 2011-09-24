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


// convolution kernels
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

#define alt_test 0

int main() {

    // image setup
    Mat src, dst;
    GpuMat d_src, d_dst;
    int runs = 100;


    for (int size = 512; size < 4000; size += 512) {

        cout  << "size: " << size << "x" << size << endl;
        gen(src, size, size, CV_32FC1, 0, 1);

        try {

            // opencv
            {
                dst.create(size, size, CV_32FC1);
                d_src = src;
                d_dst.create(size, size, CV_32FC1);
#if alt_test
                // cpu
                cv::log(src, dst);
                start_timer(0);
                for (int i = 0; i < runs; ++i) {
                    cv::log(src, dst);
                }
#else
                // cpu
                Sobel(src, dst, dst.depth(), 1, 1);
                start_timer(0);
                for (int i = 0; i < runs; ++i) {
                    Sobel(src, dst, dst.depth(), 1, 1);
                }
#endif
                cout  << " cv cpu: " << elapsed_time(0) / (float)runs << endl;
#if alt_test
                // gpu
                gpu::log(d_src, d_dst);
                start_timer(1);
                for (int i = 0; i < runs; ++i) {
                    gpu::log(d_src, d_dst);
                }
#else
                // gpu
                gpu::Sobel(d_src, d_dst, d_dst.depth(), 1, 1);
                start_timer(1);
                for (int i = 0; i < runs; ++i) {
                    gpu::Sobel(d_src, d_dst, d_dst.depth(), 1, 1);
                }
#endif
                cout << " cv gpu: " << elapsed_time(1) / (float)runs << endl;
            }

            // jacket
            {
                // extract cv image
                Mat jimg;
                src.convertTo(jimg, CV_32FC1);
                float* fgray = (float*)jimg.data;
                // copy to gpu
                f32 I1 = f32(fgray, jimg.rows, jimg.cols);
#if alt_test
                // gpu
                f32 jdst = jkt::log(I1);
                gsync();
                start_timer(2);
                for (int i = 0; i < runs; ++i) {
                    jdst = jkt::log(I1);
                    gsync();
                }
#else
                // gpu
                f32 jdst = conv2(I1, sobel_k, jktConvSame);
                gsync();
                start_timer(2);
                for (int i = 0; i < runs; ++i) {
                    jdst = conv2(I1, sobel_k, jktConvSame);
                    gsync();
                }
#endif
                cout << " jacket: " << elapsed_time(2) / (float)runs << endl;
            }


        } catch (gexception& e) {
            cout << e.what() << endl;
        }

    }

    //cout << "press space" << endl;
    //cvWaitKey(0);
    return 0;
}
