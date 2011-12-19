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
// Chris McClanahan - 2011
//
// Adapted from: http://gpu4vision.icg.tugraz.at/index.php?content=downloads.php
//   "An Improved Algorithm for TV-L1 Optical Flow"
//
// More info: http://mcclanahoochie.com/blog/portfolio/gpu-tv-l1-optical-flow-with-libjacket/
//

#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <jacket.h>
#include <jacket_gfx.h>
#include <jacket_timing.h>
#include <string.h>

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

using namespace jkt;
using namespace std;
using namespace cv;

// control
const float pfactor = 0.7;    // scale each pyr level by this amount
const int max_plevels = 9;    // number of pyramid levels
const int max_iters = 6;      // u v w update loop
const float lambda = 40;      // smoothness constraint
const int max_warps = 3;      // warping u v warping
const int min_img_sz = 20;    // min mxn img in pyramid
#define TIMING 0              // warmup, then average multiple runs

// functions
int  grab_frame(Mat& img, char* filename);
void create_pyramids(f32& im1, f32& im2, f32& pyr1, f32& pyr2);
void process_pyramids(f32& pyr1, f32& pyr2, f32& u, f32& v);
void tv_l1_dual(f32& u, f32& v, f32& p, f32& w, f32& I1, f32& I2, int level);
void optical_flow_tvl1(Mat& img1, Mat& img2, Mat& u, Mat& v);
void display_flow(f32& I2, f32& u, f32& v);
void MatToFloat(const Mat& thing, float* thing2);
void FloatToMat(float const* thing, Mat& thing2);

// misc
int plevels = max_plevels;
const int n_dual_vars = 6;
static int cam_init = 0;
static int pyr_init = 0;
VideoCapture  capture;
int pyr_M[max_plevels + 1];
int pyr_N[max_plevels + 1];
f32 pyr1, pyr2;

// macros
#define MSG(msg,...) do {                                   \
	        fprintf(stdout,__FILE__":%d(%s) " msg "\n",     \
	                __LINE__, __FUNCTION__, ##__VA_ARGS__); \
	        fflush(stdout);                                 \
	    } while (0)


// ===== main =====
void optical_flow_tvl1(Mat& img1, Mat& img2, Mat& mu, Mat& mv) {

    // extract cv image 1
    Mat mi1(img1.rows, img1.cols, CV_8UC1);
    cvtColor(img1.t(), mi1, CV_BGR2GRAY);
    mi1.convertTo(mi1, CV_32FC1);
    float* fi1 = (float*)mi1.data;
    f32 I1 = f32(fi1, img1.rows, img1.cols) / 255.0f;

    // extract cv image 2
    Mat mi2(img2.rows, img2.cols, CV_8UC1);
    cvtColor(img2.t(), mi2, CV_BGR2GRAY);
    mi2.convertTo(mi2, CV_32FC1);
    float* fi2 = (float*)mi2.data;
    f32 I2 = f32(fi2, img2.rows, img2.cols) / 255.0f;

#if TIMING
    // runs
    int nruns = 4;
    // warmup
    create_pyramids(I1, I2, pyr1, pyr2);
    f32 ou, ov;
    process_pyramids(pyr1, pyr2, ou, ov);
    // timing
    timer::tic();
    for (int i = 0; i < nruns; ++i) {
        create_pyramids(I1, I2, pyr1, pyr2);
        process_pyramids(pyr1, pyr2, ou, ov);
    }
    MSG("fps: %f", 1.0f / (timer::toc() / (float)nruns));
#else
    // timing
    timer::tic();
    // pyramids
    create_pyramids(I1, I2, pyr1, pyr2);
    // flow
    f32 ou, ov;
    process_pyramids(pyr1, pyr2, ou, ov);
    // timing
    MSG("fps: %f", 1.0f / (timer::toc()));
#endif

    // output
#if 1
    // to opencv
    FloatToMat(ou.T().host(), mu);
    FloatToMat(ov.T().host(), mv);
#else
    // to libjacket
    display_flow(I2, ou, ov);
#endif
}


void MatToFloat(const Mat& thing, float* thing2) {
    int tmp = 0;
    for (int i = 0; i < thing.rows; i++) {
        const float* fptr = thing.ptr<float>(i);
        for (int j = 0; j < thing.cols; j++)
        { thing2[tmp++] = fptr[j]; }
    }
}


void FloatToMat(float const* thing, Mat& thing2) {
    int tmp = 0;
    for (int i = 0; i < thing2.rows; ++i) {
        float* fptr = thing2.ptr<float>(i);
        for (int j = 0; j < thing2.cols; ++j)
        { fptr[j] = thing[tmp++]; }
    }
}


void display_flow(f32& I2, f32& u, f32& v) {
#if 1
    // show in libjacket
    colormap("bone");
    subplot(2, 2, 1); imagesc(I2);                  title("input");
    subplot(2, 2, 2); imagesc(u);                   title("u");
    subplot(2, 2, 3); imagesc(v);                   title("v");
    subplot(2, 2, 4); imagesc((abs(v) + abs(u)));   title("u+v");
    // int M = I2.dims()[0];
    // int N = I2.dims()[1];
    // f32 idx, idy; meshgrid(idx, idy, f32(seq(0,N-1,3)), f32(seq(0,M-1,3)));
    // quiver(idx,idy,u,v);
    drawnow();
#else
    // show in opencv
    int M = I2.dims()[0];
    int N = I2.dims()[1];
    Mat mu(M, N, CV_32FC1);
    Mat mv(M, N, CV_32FC1);
    FloatToMat(u.T().host(), mu);
    FloatToMat(v.T().host(), mv);
    imshow("u", mu);
    imshow("v", mv);
#endif
}


void display_flow(const Mat& u, const Mat& v) {
#if 0
    cv::Mat magnitude, angle, bgr;
    cv::cartToPolar(u, v, magnitude, angle, true);
    double mag_max, mag_min;
    cv::minMaxLoc(magnitude, &mag_min, &mag_max);
    magnitude.convertTo(magnitude, -1, 1.0 / mag_max);
    cv::Mat _hsv[3], hsv_image;
    _hsv[0] = angle;
    _hsv[1] = Mat::ones(angle.size(), CV_32F);
    _hsv[2] = magnitude;
    cv::merge(_hsv, 3, hsv_image);
#else
    cv::Mat magnitude, angle, bgr;
    Mat hsv_image(u.rows, u.cols, CV_8UC3);
    for (int i = 0; i < u.rows; ++i) {
        const float* x_ptr = u.ptr<float>(i);
        const float* y_ptr = v.ptr<float>(i);
        uchar* hsv_ptr = hsv_image.ptr<uchar>(i);
        for (int j = 0; j < u.cols; ++j, hsv_ptr += 3, ++x_ptr, ++y_ptr) {
            hsv_ptr[0] = (uchar)((atan2f(*y_ptr, *x_ptr) / M_PI + 1) * 90);
            hsv_ptr[1] = hsv_ptr[2] = (uchar) std::min<float>(
                                          sqrtf(*y_ptr * *y_ptr + *x_ptr * *x_ptr) * 20, 255.0);
        }
    }
#endif
    cv::cvtColor(hsv_image, bgr, CV_HSV2BGR);
    cv::imshow("optical flow", bgr);
}


int grab_frame(Mat& img, char* filename) {

    // camera/image setup
    if (!cam_init) {
        if (filename != NULL) {
            capture.open(filename);
        } else {
            float rescale = 0.615;
            int w = 640 * rescale;
            int h = 480 * rescale;
            capture.open(0); //try to open
            capture.set(CV_CAP_PROP_FRAME_WIDTH, w);  capture.set(CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if (!capture.isOpened()) { cerr << "open video device fail\n" << endl; return 0; }
        capture >> img; capture >> img;
        if (img.empty()) { cout << "load image fail " << endl; return 0; }
        namedWindow("cam", CV_WINDOW_KEEPRATIO);
        printf(" img = %d x %d \n", img.cols, img.rows);
        cam_init = 1;
    }

    // get frames
    capture.grab();
    capture.retrieve(img);
    imshow("cam", img);

    if (waitKey(10) >= 0) { return 0; }
    else { return 1; }
}


void gen_pyramid_sizes(f32& im1) {
    gdims mnk = im1.dims();
    float sM = mnk[0];
    float sN = mnk[1];
    // store resizing
    for (int level = 0; level <= plevels; ++level) {
        if (level == 0) {
        } else {
            sM *= pfactor;
            sN *= pfactor;
        }
        pyr_M[level] = (int)(sM + 0.5f);
        pyr_N[level] = (int)(sN + 0.5f);
        MSG(" pyr %d: %d x %d ", level, (int)sM, (int)sN);
        if (sM < min_img_sz || sN < min_img_sz) { plevels = level; break; }
    }
}

void create_pyramids(f32& im1, f32& im2, f32& pyr1, f32& pyr2) {

    if (!pyr_init) {
        // list of h,w
        gen_pyramid_sizes(im1);

        // init
        pyr1 = f32::zeros(pyr_M[0], pyr_N[0], plevels);
        pyr2 = f32::zeros(pyr_M[0], pyr_N[0], plevels);
        pyr_init = 1;
    }

    // create
    for (int level = 0; level < plevels; level++) {
        if (level == 0) {
            pyr1(span, span, level) = im1;
            pyr2(span, span, level) = im2;
        } else {
            seq spyi = seq(pyr_M[level - 1]);
            seq spxi = seq(pyr_N[level - 1]);
            f32 small1 = resize(pyr1(spyi, spxi, level - 1), pyr_M[level], pyr_N[level], JKT_RSZ_Bilinear);
            f32 small2 = resize(pyr2(spyi, spxi, level - 1), pyr_M[level], pyr_N[level], JKT_RSZ_Bilinear);
            seq spyo = seq(pyr_M[level]);
            seq spxo = seq(pyr_N[level]);
            pyr1(spyo, spxo, level) = small1;
            pyr2(spyo, spxo, level) = small2;
        }
    }
}


void process_pyramids(f32& pyr1, f32& pyr2, f32& ou, f32& ov) {
    f32 p, u, v, w;

    // pyramid loop
    for (int level = plevels - 1; level >= 0; level--) {
        if (level == plevels - 1) {
            u  = f32::zeros(pyr_M[level], pyr_N[level]);
            v  = f32::zeros(pyr_M[level], pyr_N[level]);
            w  = f32::zeros(pyr_M[level], pyr_N[level]);
            p  = f32::zeros(pyr_M[level], pyr_N[level], n_dual_vars);
        } else {
            float rescale_u =  pyr_N[level + 1] / (float)pyr_N[level];
            float rescale_v =  pyr_M[level + 1] / (float)pyr_M[level];
            // propagate
            f32 u_ =  resize(u, pyr_M[level], pyr_N[level], JKT_RSZ_Bilinear) * rescale_u;
            f32 v_ =  resize(v, pyr_M[level], pyr_N[level], JKT_RSZ_Bilinear) * rescale_v;
            f32 w_ =  resize(w, pyr_M[level], pyr_N[level], JKT_RSZ_Bilinear);
            f32 p_ = f32::zeros(pyr_M[level], pyr_N[level], n_dual_vars);
            gfor(f32 ndv, n_dual_vars) {
                p_(span, span, ndv) = resize(p(span, span, ndv), pyr_M[level], pyr_N[level], JKT_RSZ_Bilinear);
            }
            u = u_;  v = v_;  p = p_;  w = w_;
        }

        // extract
        seq spy = seq(pyr_M[level]);
        seq spx = seq(pyr_N[level]);
        f32 I1 = pyr1(spy, spx, level);
        f32 I2 = pyr2(spy, spx, level);

        // ===== core ====== //
        tv_l1_dual(u, v, p, w, I1, I2, level);
        // ===== ==== ====== //
    }

    // output
    ou = u;
    ov = v;
}


void warping(f32& Ix, f32& Iy, f32& It, f32& I1, f32& I2, f32& u, f32& v) {

    gdims mnk = I2.dims();
    int M = mnk[0];
    int N = mnk[1];
    f32 idx = repmat(f32(seq(N)).T(), M, 1) + 1;
    f32 idy = repmat(f32(seq(M)), 1, N) + 1;
    /* ^ BUG: idx idy should ideally be [0-N); ^ */

    f32 idxx0 = idx + u;
    f32 idyy0 = idy + v;
    f32 idxx = max(1, min(N - 1, idxx0));
    f32 idyy = max(1, min(M - 1, idyy0));

    // interp2 based warp ()
    It = interp2(idy, idx, I2, idyy, idxx) - I1;

    // interp2 based warp ()
    f32 idxm = max(1, min(N - 1, idxx - 1.f));
    f32 idxp = max(1, min(N - 1, idxx + 1.f));
    f32 idym = max(1, min(M - 1, idyy - 1.f));
    f32 idyp = max(1, min(M - 1, idyy + 1.f));
    Ix = interp2(idy, idx, I2, idy, idxp) - interp2(idy, idx, I2, idy, idxm);
    Iy = interp2(idy, idx, I2, idyp, idx) - interp2(idy, idx, I2, idym, idx);
    /* ^ BUG: interp2 should be cubic; that may fix things; ^ */
}


void dxym(f32& Id, f32 I0x, f32 I0y) {
    // divergence
    gdims mnk = I0x.dims();
    int M = mnk[0];
    int N = mnk[1];

    f32 x0 = f32::zeros(M, N);
    f32 x1 = f32::zeros(M, N);
    x0(seq(N - 1), seq(M)) = I0x(seq(N - 1), seq(M));
    x1(seq(1,  N), seq(M)) = I0x(seq(1,  N), seq(M));

    f32 y0 = f32::zeros(M, N);
    f32 y1 = f32::zeros(M, N);
    y0(seq(N), seq(M - 1)) = I0y(seq(N), seq(M - 1));
    y1(seq(N), seq(1,  M)) = I0y(seq(N), seq(1,  M));

    Id = (x0 - x1) + (y0 - y1);
}


void dxyp(f32& Ix, f32& Iy, f32& I0) {
    // shifts
    gdims mnk = I0.dims();
    int M = mnk[0];
    int N = mnk[1];

    f32 y0 = I0;
    f32 y1 = I0;
    y0(seq(0, M - 2), span) = I0(seq(1, M - 1), span);

    f32 x0 = I0;
    f32 x1 = I0;
    x0(span, seq(0, N - 2)) = I0(span, seq(1, N - 1));

    Ix = (x0 - x1);  Iy = (y0 - y1);
}


void tv_l1_dual(f32& u, f32& v, f32& p, f32& w, f32& I1, f32& I2, int level) {

    float L = sqrtf(8.0f);
    float tau   = 1 / L;
    float sigma = 1 / L;

    float eps_u = 0.01f;
    float eps_w = 0.01f;
    float gamma = 0.02f;

    f32 u_ = u;
    f32 v_ = v;
    f32 w_ = w;

    for (int j = 0; j < max_warps; j++) {

        f32 u0 = u;
        f32 v0 = v;

        // warping
        f32 Ix, Iy, It;   warping(Ix, Iy, It, I1, I2, u0, v0);

        // gradients
        f32 I_grad_sqr = jkt::max(float(1e-6), f32(power(Ix, 2) + power(Iy, 2) + gamma * gamma));

        // inner loop
        for (int k = 0; k < max_iters; ++k) {

            // dual =====

            // shifts
            f32 u_x, u_y;    dxyp(u_x, u_y, u_);
            f32 v_x, v_y;    dxyp(v_x, v_y, v_);
            f32 w_x, w_y;    dxyp(w_x, w_y, w_);

            // update dual
            p(span, span, 0) = (p(span, span, 0) + sigma * u_x) / (1 + sigma * eps_u);
            p(span, span, 1) = (p(span, span, 1) + sigma * u_y) / (1 + sigma * eps_u);
            p(span, span, 2) = (p(span, span, 2) + sigma * v_x) / (1 + sigma * eps_u);
            p(span, span, 3) = (p(span, span, 3) + sigma * v_y) / (1 + sigma * eps_u);

            p(span, span, 4) = (p(span, span, 4) + sigma * w_x) / (1 + sigma * eps_w);
            p(span, span, 5) = (p(span, span, 5) + sigma * w_y) / (1 + sigma * eps_w);

            // normalize
            f32 reprojection = max(1, sqrt(power(p(span, span, 0), 2) + power(p(span, span, 1), 2) +
                                           power(p(span, span, 2), 2) + power(p(span, span, 3), 2)));

            p(span, span, 0) = p(span, span, 0) / reprojection;
            p(span, span, 1) = p(span, span, 1) / reprojection;
            p(span, span, 2) = p(span, span, 2) / reprojection;
            p(span, span, 3) = p(span, span, 3) / reprojection;

            reprojection = max(1, sqrt(power(p(span, span, 4), 2) + power(p(span, span, 5), 2)));

            p(span, span, 4) = p(span, span, 4) / reprojection;
            p(span, span, 5) = p(span, span, 5) / reprojection;

            // primal =====

            // divergence
            f32 div_u;   dxym(div_u, p(span, span, 0), p(span, span, 1));
            f32 div_v;   dxym(div_v, p(span, span, 2), p(span, span, 3));
            f32 div_w;   dxym(div_w, p(span, span, 4), p(span, span, 5));

            // old
            u_ = u;
            v_ = v;
            w_ = w;

            // update
            u = u + tau * div_u;
            v = v + tau * div_v;
            w = w + tau * div_w;

            // indexing
            f32 rho  = It + (u - u0) * Ix + (v - v0) * Iy + gamma * w;
            b8 idx1 = rho      <  -tau * lambda * I_grad_sqr;
            b8 idx2 = rho      >   tau * lambda * I_grad_sqr;
            b8 idx3 = abs(rho) <=  tau * lambda * I_grad_sqr;

            u = u + tau * lambda * (Ix * idx1) ;
            v = v + tau * lambda * (Iy * idx1) ;
            w = w + tau * lambda * gamma * idx1;

            u = u - tau * lambda * (Ix * idx2) ;
            v = v - tau * lambda * (Iy * idx2) ;
            w = w - tau * lambda * gamma * idx2;

            u = u - rho * idx3 * Ix / I_grad_sqr;
            v = v - rho * idx3 * Iy / I_grad_sqr;
            w = w - rho * idx3 * gamma / I_grad_sqr;

            // propagate
            u_ = 2 * u - u_;
            v_ = 2 * v - v_;
            w_ = 2 * w - w_;

        }

        // output
        const unsigned hw[] = {3, 3};
        u = medfilt2(u, hw);
        v = medfilt2(v, hw);

    } /* j < warps */
}


// =======================================

int main(int argc, char* argv[]) {

    // video file or usb camera
    Mat cam_img, prev_img, disp_u, disp_v;
    int is_images = 0;
    if (argc == 2) { grab_frame(prev_img, argv[1]); } // video
    else if (argc == 3) {
        prev_img = imread(argv[1]); cam_img = imread(argv[2]);
        is_images = 1;
    } else { grab_frame(prev_img, NULL); } // usb camera

    // results
    int mm = prev_img.rows;  int nn = prev_img.cols;
    disp_u = Mat::zeros(mm, nn, CV_32FC1);
    disp_v = Mat::zeros(mm, nn, CV_32FC1);
    printf("img %d x %d \n", mm, nn);

    // process main
    if (is_images) {
        // show
        imshow("i", cam_img);
        // process files
        optical_flow_tvl1(prev_img, cam_img, disp_u, disp_v);
        // show
        // imshow("u", disp_u);
        // imshow("v", disp_v);
        display_flow(disp_u, disp_v);
        waitKey(0);
        // // write
        // writeFlo(disp_u, disp_v);
    } else {
        // process loop
        while (grab_frame(cam_img, NULL)) {
            try {
                // process
                optical_flow_tvl1(prev_img, cam_img, disp_u, disp_v);
                // frames
                prev_img = cam_img.clone();
                // show
                // imshow("u", disp_u);
                // imshow("v", disp_v);
                display_flow(disp_u, disp_v);
            } catch (gexception& e) {
                cout << e.what() << endl;
                throw;
            }
        }
    }

    return 0;
}

