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

const int t_ocv = 0;
const int t_jkt = 1;

int ksz;
int kszs[] = {3, 5, 9, 13, 32, 64, 128};

void gen(Mat& mat, int rows, int cols, int type, Scalar low, Scalar high) {
    mat.create(rows, cols, type);
    RNG rng(0);
    rng.fill(mat, RNG::UNIFORM, low, high);
}

int main() {

    jkt::info();
    Mat src, dst, ker;
    GpuMat d_src, d_dst, d_ker;
    int runs = 100;

    for (int k = 0; k < 7; ++k) {

        ksz = kszs[k];
        cout  << "kernelsize: " << ksz << "x" << ksz << endl;
        gen(ker, ksz, ksz, CV_32F, -1, 1);

        // bench
        for (int size = 512; size < 4000; size += 512) {

            cout  << "imgsize: " << size << "x" << size << endl;
            gen(src, size, size, CV_32FC1, 0, 1);

            try {

                // opencv
                if (t_ocv) {
                    // setup
                    dst.create(size, size, CV_32FC1);
                    d_src = src;
                    d_dst.create(size, size, CV_32FC1);
                    d_ker = ker;

                    // convolve(d_src, d_ker, d_dst);
                    ConvolveBuf buf;
                    convolve(d_src, d_ker, d_dst, false, buf);
                    start_timer(1);
                    for (int i = 0; i < runs; ++i) {
                        // convolve(d_src, d_ker, d_dst);
                        convolve(d_src, d_ker, d_dst, false, buf);
                    }
                    cout << "cv-gpu: " << elapsed_time(1) / (float)runs << endl;
                }

                // jacket
                if (t_jkt) {
                    // extract cv image
                    Mat jimg;
                    src.convertTo(jimg, CV_32FC1);
                    float* fgray = (float*)jimg.data;
                    f32 I1 = f32(fgray, jimg.rows, jimg.cols);
                    unsigned dimsb[] = {ksz, ksz};

                    // gpu
                    f32 jker = f32((float*)ker.data, ker.rows, ker.cols);
                    f32 jdst = conv2(I1, jker, jktConvValid);
                    gsync();
                    start_timer(2);
                    for (int i = 0; i < runs; ++i) {
                        jdst = conv2(I1, jker, jktConvValid);
                    }
                    gsync();
                    cout << "jacket: " << elapsed_time(2) / (float)runs << endl;
                }

            } catch (gexception& e) {
                cout << e.what() << endl;
            }

        } // image size loop
    } // kernel size loop

    return 0;
}
