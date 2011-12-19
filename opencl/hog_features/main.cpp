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
// Chris McClanahan & Brian Hrolenok - 2011
//
// HOG Features
//


#include <iostream>
#include <fstream>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "simpleCL.h"
#include "timer.h"

#include <cv.h>
#include <cxcore.h>
#include <highgui.h>

using namespace std;
using namespace cv;

#define NBINS     8            // CHECK in kernels.cl also
#define CELLDIM  16
#define BLOCK_SIZE_X CELLDIM   // lazy: make block sizes
#define BLOCK_SIZE_Y CELLDIM   //       equal to celldim
#define TIMING    0            // warmup, then average multiple runs
#define DEMO_MODE 1


// functions
int grab_frame(Mat& img, char* filename);
void MatToFloat(const Mat& thing, float* thing2);
void FloatToMat(float const* thing, Mat& thing2);
void cell_hist_test(Mat& img);
void display_cl_buffer(char* name, cl_mem d_img, int rows, int cols, bool convert);
void run_hog(float* h_img, int rows, int cols, float* h_feats, int celldim);
void printFloatMat(const Mat& thing);

// misc
static int cv_init_ = 0;
VideoCapture  capture;
sclHard hardware;
size_t localWorkSize[] = {BLOCK_SIZE_Y, BLOCK_SIZE_X};
cl_mem d_w;
float h_b;
float* h_w_vec;


// kernels
#define NUM_KERNELS 6
static sclSoft software[NUM_KERNELS];
static bool comp_flag[NUM_KERNELS];


// debug cl mem display
void display_cl_buffer(char* name, cl_mem d_img, int rows, int cols, bool convert) {

    size_t mem_size = sizeof(float) * rows * cols;
    Mat disp = Mat::zeros(rows, cols, CV_32FC1);
    float* h_img = (float*)malloc(mem_size);
    sclRead(hardware, mem_size, d_img, h_img);
    FloatToMat(h_img, disp);
    if (convert) { disp.convertTo(disp, CV_8UC1); }
    imshow(name, disp * 1.2); // scale hack for gradient image
    free(h_img);

}

// ... so inefficient!
void display_cl_bins(char* name, cl_mem d_img, int ncells, int nbins, int irows, int icols) {

    Mat hbins = Mat::zeros(ncells, nbins, CV_32FC1);
    size_t mem_size = sizeof(float) * ncells * nbins;
    float* h_img = (float*)malloc(mem_size);
    sclRead(hardware, mem_size, d_img, h_img);
    FloatToMat(h_img, hbins);

    // reshape to ncells_y by ncells_x*nbins rectangle
    hbins = hbins.reshape(0, irows / CELLDIM);

    // place reshaped bins into larger image by
    // replicating rows of each histogram to make a square
    float max = 0;
    Mat large = Mat::zeros(irows / CELLDIM * nbins, icols / CELLDIM * nbins, CV_32FC1);
    // for(int hr = 0; hr<hbins.rows; ++hr){
    //     for(int hc = 0; hc<hbins.cols; ++hc){
    //         float val = hbins.at<float>(hr,hc);
    //         if(val>max) max = val;
    //         for (int j=0; j<nbins; ++j){
    //             large.at<float>(hr*nbins+j,hc) = val;
    //         }
    //     }
    // }
    // // pseudo normalize
    // large = (large/(max/2));

    int sections = 360 / nbins;

    // draw angles
    for (int hr = 0; hr < hbins.rows; ++hr) {
        for (int hc = 0; hc < hbins.cols; hc += nbins) {
            int mid = nbins / 2;
            int idx, val; max = 0;
            for (int bc = 0; bc < nbins; ++bc) {
                val = hbins.at<float>(hr, hc + bc);
                if (val > max) { max = val; idx = bc; }
            }
            //printf("%f ", max);
            int angle = idx * sections;//360 / nbins;
            Point pt; pt.x = hc + mid; pt.y = hr * nbins + mid;
            RotatedRect rr(pt, Size(nbins - 2, 1), angle);
            if (max > 0) { ellipse(large, rr, Scalar(1, 1, 1)); }
        }
    }


    // show cell histo grid
    for (int lr = 0; lr < large.rows; lr += nbins) {
        for (int lc = 0; lc < large.cols; lc += nbins) {
            large.at<float>(lr, lc) = 0.5;
        }
    }

    Mat larger;
    resize(large, larger, Size(), 1.6, 1.6);
    imshow(name, larger);

    free(h_img);

}


void check_feats(cl_mem d_hists, int ncells, int nbins, int wi, int wj, cl_mem d_img) {

    //return;

    // if (!comp_flag[4]) {
    //     comp_flag[4] = true;
    //     software[4] = sclGetCLSoftware("kernels.cl", "classify", hardware);
    // }

    // size_t fargsize = sizeof(float);
    // size_t iargsize = sizeof(int);
    // size_t globalWorkSize[] = {
    //     ((ncells - 1) / localWorkSize[0] + 1)* localWorkSize[0],
    //     ((nbins  - 1) / localWorkSize[1] + 1)* localWorkSize[1]
    // };


    // cl_mem d_response = sclMalloc(hardware, CL_MEM_READ_WRITE,  iargsize * 1);
    // int h_response = 0;

    // sclSetKernelArgs(software[4], " %v %v %v %a ",
    //                  &d_hists, &d_w,
    //                  &d_response,
    //                  iargsize, &ncells
    //                 );
    // sclLaunchKernel(hardware, software[4], globalWorkSize, localWorkSize);

    // sclRead(hardware, iargsize * 1, d_response, &h_response);
    // printf("response %d  class %f \n", h_response, h_response-h_b);

    // if(h_response != 0) waitKey(0);


    // sclReleaseMemObject(d_response);

    size_t mem_hsize = sizeof(float) * ncells * NBINS;
    float* h_feats = (float*)malloc(mem_hsize);
    sclRead(hardware, mem_hsize, d_hists, h_feats);

    float dot  = 0;
    for (int i = 0; i < ncells * nbins; ++i) {
        dot += h_feats[i] * h_w_vec[i];
    }

    float res = dot - h_b;
    printf("response %f \n", res);


    if (abs(res) >= 1) { waitKey(0); }

}


void save_feats(cl_mem d_hists, int ncells, int nbins, int positive) {

    ofstream myfile;
    myfile.open("feats.txt", ios::app);
    int nfeats = 1;

    size_t mem_hsize = sizeof(float) * ncells * NBINS;
    float* h_feats = (float*)malloc(mem_hsize);
    Mat hists  = Mat::zeros(ncells, NBINS, CV_32FC1);

    printf("%d ", positive);
    myfile << positive << " ";

    sclRead(hardware, mem_hsize, d_hists, h_feats);
    FloatToMat(h_feats, hists);
    for (int i = 0; i < hists.rows; ++i) {
        for (int j = 0; j < NBINS; ++j) {
            printf("%d:%d ", nfeats, (int)hists.at<float>(i, j));
            myfile << nfeats << ":" << (int)hists.at<float>(i, j) << " ";
            ++nfeats;
        }
        // printf("\n");
        // myfile << "\n";
    }

    printf("\n");
    myfile << "\n";
    free(h_feats);
    myfile.close();

}


void window_hists(cl_mem d_ang, cl_mem d_mag, cl_mem d_hists, int irows, int icols,
                  int celldim, int wrows, int wcols, cl_mem d_img) {

    if (!comp_flag[0]) {
        comp_flag[0] = true;
        software[0] = sclGetCLSoftware("kernels.cl", "window_hist", hardware);
    }

    size_t iargsize = sizeof(int);
    size_t globalWorkSize[] = {
        ((wrows - 1) / localWorkSize[0] + 1)* localWorkSize[0],
        ((wcols - 1) / localWorkSize[1] + 1)* localWorkSize[1]
    };

    int step = 16;
    for (int i = 0; i < irows - step; i += step) {
        for (int j = 0; j < icols - step; j += step) {

            if (i + wrows > irows || j + wcols > icols) { continue; }
            sclSetKernelArgs(software[0], " %v %v %a %a %v %a %a %a %a ",
                             &d_ang, &d_mag,
                             iargsize, &irows, iargsize, &icols,
                             &d_hists,
                             iargsize, &i, iargsize, &j,
                             iargsize, &wrows, iargsize, &wcols
                            );
            sclLaunchKernel(hardware, software[0], globalWorkSize, localWorkSize);

            // debug
            int ncells = (wcols / celldim) * (wrows / celldim);
#if 1
            //printf("%d %d \n",i,j);
            // display_cl_buffer("ang", d_ang, irows, icols, false);
            display_cl_bins("grid-hists", d_hists, ncells, NBINS, wrows, wcols);
            waitKey(5);
#endif

            int positive_ex = 1;
            // classify
#if DEMO_MODE
            //          check_feats(d_hists, ncells, NBINS, i, j, d_img);
#else
            save_feats(d_hists, ncells, NBINS, positive_ex);
#endif
        }
    }


}


void run_hog(float* h_img, int rows, int cols, float* h_feats, int celldim) {

    // sizes
    size_t fargsize = sizeof(float);
    size_t iargsize = sizeof(int);
    size_t mem_isize = fargsize * rows * cols;
    size_t globalWorkSize[] = {
        ((rows - 1) / localWorkSize[0] + 1)* localWorkSize[0],
        ((cols - 1) / localWorkSize[1] + 1)* localWorkSize[1]
    };

    // mem
    cl_mem d_img   = sclMalloc(hardware, CL_MEM_READ_WRITE,  mem_isize);
    cl_mem d_xfilt = sclMalloc(hardware, CL_MEM_READ_WRITE,  mem_isize);
    cl_mem d_yfilt = sclMalloc(hardware, CL_MEM_READ_WRITE,  mem_isize);
    cl_mem d_ang   = sclMalloc(hardware, CL_MEM_READ_WRITE,  mem_isize);
    cl_mem d_mag   = sclMalloc(hardware, CL_MEM_READ_WRITE,  mem_isize);
    sclWrite(hardware, mem_isize, d_img, h_img);

    // x gradients
    if (!comp_flag[1]) {
        comp_flag[1] = true;
        software[1] = sclGetCLSoftware("1d-gradient-filters.cl", "xfilter", hardware);
    }
    sclSetKernelArgs(software[1],
                     " %v %v %a %a",
                     &d_img, &d_xfilt,
                     iargsize, &rows,
                     iargsize, &cols);
    sclLaunchKernel(hardware, software[1], globalWorkSize, localWorkSize);

    // y gradients
    if (!comp_flag[2]) {
        comp_flag[2] = true;
        software[2] = sclGetCLSoftware("1d-gradient-filters.cl", "yfilter", hardware);
    }
    sclSetKernelArgs(software[2],
                     " %v %v %a %a",
                     &d_img, &d_yfilt,
                     iargsize, &rows,
                     iargsize, &cols);
    sclLaunchKernel(hardware, software[2], globalWorkSize, localWorkSize);

    // angles
    if (!comp_flag[3]) {
        comp_flag[3] = true;
        software[3] = sclGetCLSoftware("cart-to-polar.cl", "cart2polar", hardware);
    }
    sclSetKernelArgs(software[3],
                     " %v %v %v %v %a %a",
                     &d_xfilt, &d_yfilt,
                     &d_ang, &d_mag,
                     iargsize, &rows,
                     iargsize, &cols);
    sclLaunchKernel(hardware, software[3], globalWorkSize, localWorkSize);

    // debug
    // display_cl_buffer("ang", d_ang, rows, cols, false);
    display_cl_buffer("mag", d_mag, rows, cols, false);


    // // histograms (OLD)
    // if (!comp_flag[0]) {
    //     comp_flag[0] = true;
    //     software[0] = sclGetCLSoftware("kernels.cl", "histograms", hardware);
    // }
    // sclSetKernelArgs(software[0], " %v %v %a %a %v ",
    //                  &d_ang, &d_mag,
    //                  iargsize, &rows, iargsize, &cols,
    //                  &d_hists);
    // sclLaunchKernel(hardware, software[0], globalWorkSize, localWorkSize);

    // windowed hists
#if DEMO_MODE
    // entire image
    int w_rows = rows;
    int w_cols = cols;
#else
    // sliding windows
    int w_rows = 468; // from training:
    int w_cols = 352; //  * avg bbox size xy 352.436 468.568
#endif
    int ncells = (w_rows / celldim) * (w_cols / celldim);
    size_t mem_hsize = fargsize * ncells * NBINS;
    cl_mem d_hists = sclMalloc(hardware, CL_MEM_READ_WRITE,  mem_hsize);
    window_hists(d_ang, d_mag, d_hists, rows, cols, celldim, w_rows, w_cols, d_img);

    // // mem output
    // sclRead(hardware, mem_hsize, d_hists, h_feats);

    // cleanup
    sclReleaseMemObject(d_hists);
    sclReleaseMemObject(d_xfilt);
    sclReleaseMemObject(d_yfilt);
    sclReleaseMemObject(d_img);
    sclReleaseMemObject(d_ang);
    sclReleaseMemObject(d_mag);

}


void cell_hist_test(Mat& m_I, Mat& m_F) {

    // extract cv image
    Mat mgray1(m_I.rows, m_I.cols, CV_8UC1);
    cvtColor(m_I, mgray1, CV_BGR2GRAY);
    mgray1.convertTo(mgray1, CV_32FC1);
    mgray1 /= 255.0f;
    float* h_i = (float*)mgray1.data;

    // setup
    unsigned int cols = m_I.cols, rows = m_I.rows;
    float* h_feats = (float*)m_F.data;

#if TIMING
    // runs
    int nruns = 4;
    // warmup
    run_hog(h_i, rows, cols, h_feats, CELLDIM);
    // timing
    start_timer(0);
    for (int i = 0; i < nruns; ++i) {
        // go
        run_hog(h_i, rows, cols, h_feats, CELLDIM);
    }
    printf("fps: %f \n", 1.0f / (elapsed_time(0) / 1000.0f / (float)nruns));
#else
    // timing
    start_timer(0);
    // go
    run_hog(h_i, rows, cols, h_feats, CELLDIM);
    // timing
    printf("fps: %f \n", 1.0f / (elapsed_time(0) / 1000.0f));
#endif

    // // output
    // FloatToMat(h_feats, m_F);
    // // show hist
    // printFloatMat(m_F);
    // Mat temp(m_F);
    // temp.convertTo(temp, CV_8UC1);
    // imshow("hist", temp);
    // waitKey(0);

}


void printFloatMat(const Mat& thing) {
    for (int i = 0; i < thing.rows; i++) {
        const float* fptr = thing.ptr<float>(i);
        for (int j = 0; j < thing.cols; j++)
        { printf("%f ", fptr[j]);}
        { printf("\n"); }
    }
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
    for (int i = 0; i < thing2.rows; i++) {
        float* fptr = thing2.ptr<float>(i);
        for (int j = 0; j < thing2.cols; j++)
        { fptr[j] = thing[tmp++]; }
    }
}


int grab_frame(Mat& img, char* filename) {

    // camera/image setup
    if (!cv_init_) {
        if (filename != NULL) {
            capture.open(filename);
        } else {
            // float scale = 0.615;
            // int w = 640 * scale;
            // int h = 480 * scale;
            int w = 896;
            int h = 592;
            capture.open(0); //try to open
//            capture.set(CV_CAP_PROP_FRAME_WIDTH, w);  capture.set(CV_CAP_PROP_FRAME_HEIGHT, h);
        }
        if (!capture.isOpened()) { cerr << "open video device fail\n" << endl; return 0; }
        capture.grab();
        capture.retrieve(img);
        if (img.empty()) { cout << "load image fail " << endl; return 0; }
        printf(" img = %d x %d \n", img.rows, img.cols);
        cv_init_ = 1;
    }

    // get frames
    capture.grab();
    capture.retrieve(img);
    // img = Mat::eye(img.rows, img.cols, CV_8UC3)*255;
    imshow("cam", img);

    if (waitKey(10) >= 0) { return 0; }
    else { return 1; }
}


// wow, myfile2 just for linecount
void training_loop(ifstream& myfile, ifstream& myfile2) {

    // get length of file:
    int length = 0;
    std::string tmp;
    while (getline(myfile2, tmp)) { ++length; }
    // myfile.seekg(0, ios::beg);
    cout << length << endl;

    length -= 1; // the last last is a newline
    Mat bboxes = Mat::zeros(length, 8, CV_32FC1);
    length = 0;
    double diffX = 0, diffY = 0;
    double tmpPnts[8];
    std::vector<Point> start_pts;
    while (myfile) {
        for (int i = 0; i < 8; ++i) {
            std::string input;
            myfile >> input;
            float val = atof(input.c_str());
            cout << val << " ";
            //bboxes.at<float>(length,i) = val;
            tmpPnts[i] = val;
        }
        diffX += tmpPnts[6] - tmpPnts[2];
        diffY += tmpPnts[7] - tmpPnts[3];
        ++length;
        cout << endl;
        Point pt; pt.x = tmpPnts[2]; pt.y = tmpPnts[3];
        start_pts.push_back(pt);

    }
    length -= 1; // the last last a is newline

    /*
    // 2,3 - 6,7
    float avgx = 0, avgy = 0;
    for (int i=0; i<bboxes.rows; ++i) {
        Point tl; tl.x = bboxes.at<float>(i,2); tl.y = bboxes.at<float>(i,3);
        Point br; br.x = bboxes.at<float>(i,6); br.y = bboxes.at<float>(i,7);
        Point diff = br - tl;
        avgx += diff.x; avgy += diff.y;
    }
    avgx/=(float)length; avgy/=(float)length;
    Point avg; avg.x = avgx; avg.y = avgy;
    cout << "avg bbox size xy " << avg.x << " " << avg.y << " " << endl;
    */

    cout << "avg bbox size xy " << diffX / (double)length  << " " << diffY / (double)length << endl;
    diffX /= (float)length; diffY /= (float)length;
    srand(time(NULL));

    // here be dragons

    for (int i = 0; i < bboxes.rows; ++i) {
        std::string path = "/Users/chris/Downloads/data/image_";
        char name[] = {0, 0, 0, 0};
        sprintf(name, "%04d", i + 1);
        path = path + std::string(name) + std::string(".jpg");
        cout << path.c_str() << endl;
        Mat img_in = imread(path.c_str());
        cout << img_in.rows << endl;

        Point ulrand; ulrand.x = rand() % ((int)(img_in.cols - diffX)); ulrand.y = rand() % ((int)(img_in.rows - diffY));

        Point ul = start_pts.at(i);
        if (ul.x + diffX / 2 > img_in.cols / 2) { ulrand.x = 0; }
        else { ulrand.x = (img_in.cols - diffX) - 1; }

        Point lrt = Point(ul.x + diffX, ul.y + diffY);
        Point ult = ul;

        //ul = ulrand;

        Point lr = Point(ul.x + diffX, ul.y + diffY);
        if (ul.x + diffX >= img_in.cols || ul.y + diffY >= img_in.rows) { continue; }

        //rectangle(img_in, ult, lrt, Scalar(0,255,255));
        rectangle(img_in, ul, lr, Scalar(255, 0, 255));
        imshow("train", img_in);
        Mat cropped(img_in, Rect(ul.x, ul.y, diffX, diffY));
        imshow("cropped", cropped);

        int ncells = (cropped.rows / CELLDIM) * (cropped.cols / CELLDIM);
        Mat hists  = Mat::zeros(ncells, NBINS, CV_32FC1);
        cell_hist_test(cropped, hists);

        waitKey(5);
    }




}

float* load_svm(const char* fname, int& _numFeats, float& _b) {
    ifstream inf;
    char trash[256];
    inf.open(fname);
    string line;
    //First, SVM_Light version string
    inf.getline(trash, 256);
    //kernel type
    inf.getline(trash, 256);
    //kernel parameter -d
    inf.getline(trash, 256);
    //kernel parameter -g
    inf.getline(trash, 256);
    //kernel parameter -s
    inf.getline(trash, 256);
    //kernel parameter -r
    inf.getline(trash, 256);
    //kernel parameter -u
    inf.getline(trash, 256);
    //Num features
    int numFeats;
    inf >> numFeats;
    inf.getline(trash, 256);
    //cout<<"Num features: "<<numFeats<<endl;
    //Num training examples
    inf.getline(trash, 256);
    //Num support vectors
    int numSV;
    inf >> numSV;
    //cout<<"Num SV's: "<<numSV<<endl;
    inf.getline(trash, 256);
    //B
    float b;
    inf >> b;
    //cout<<"B: "<<b<<endl;
    inf.getline(trash, 256);
    float* w = new float[numFeats];
    for (int i = 0; i < numFeats; i++) { w[i] = 0.0; }
    for (int svCnt = 0; svCnt < numSV; svCnt++) {
        double alphaTy, tmp;
        inf >> alphaTy;
        for (int fCnt = 0; fCnt > numFeats; fCnt++) {
            //grab to the first :
            inf.getline(trash, 256, ':');
            inf >> tmp;
            w[fCnt] += (float)tmp * alphaTy;
        }
    }
    //cout<<"W = [ ";
    //for(int i=0;i<numFeats;i++)
    //	cout<<w[i]<<" ";
    //cout<<"]"<<endl;
    _b = b;
    _numFeats = numFeats;
    return w;
}


// =======================================

int main(int argc, char** argv) {

    // image file or usb camera
    Mat cam_img;
    int is_images = 0;
    ifstream myfile;
    ifstream myfile2;
    if (argc == 3) { // image + txt file
        cam_img = imread(argv[1]);
        myfile.open(argv[2]);
        myfile2.open(argv[2]);
        is_images = 1;
    } else {  // usb camera
        grab_frame(cam_img, NULL);
    }

    // simple-opencl
    int found;
    sclHard* allHardware;
    found = sclGetAllHardware(&allHardware);
    hardware = sclGetFastestDevice(allHardware, found);
    for (int i = 0; i < NUM_KERNELS; ++i) { comp_flag[i] = false; }

    // hist
    int ncells = (cam_img.rows / CELLDIM) * (cam_img.cols / CELLDIM);
    Mat hists  = Mat::zeros(ncells, NBINS, CV_32FC1);
    printf("ncells = %d\n", ncells);

    // w
    int numfeats;
    float* h_w = load_svm("../data/learned_hog.txt", numfeats, h_b);
    h_w_vec = h_w;
    // d_w = sclMalloc(hardware, CL_MEM_READ_WRITE,  sizeof(float) * ncells * NBINS);
    // sclWrite(hardware, sizeof(float) * ncells * NBINS, d_w, h_w);


    // process main
    if (is_images) {
        // process file
        // cell_hist_test(cam_img, hists);
        // waitKey(0);
        training_loop(myfile, myfile2);
    } else {
        // process loop
        while (grab_frame(cam_img, NULL)) {
            cell_hist_test(cam_img, hists);
        }
    }

    if (myfile.is_open()) { myfile.close(); }
    if (myfile2.is_open()) { myfile2.close(); }

    free(h_w);
    clReleaseMemObject(d_w);
    return 0;
}

