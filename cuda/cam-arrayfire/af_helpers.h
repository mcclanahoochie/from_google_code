

#ifndef _af_helpers_h_
#define _af_helpers_h_

#include <stdio.h>
#include <arrayfire.h>
#include <opencv2/opencv.hpp>

using namespace af;
using namespace cv;

// mem layout for gpu
void mat_to_array(const cv::Mat& input, array& output) ;
array mat_to_array(const cv::Mat& input) ;

// mem layout for cpu
void array_to_mat(const array& input, cv::Mat& output, int type=CV_32F) ;
Mat array_to_mat(const array& input, int type=CV_32F) ;

#endif
