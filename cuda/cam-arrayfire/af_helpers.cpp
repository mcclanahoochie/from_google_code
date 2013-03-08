#include "af_helpers.hpp"

// ==================================================================

// mem layout for gpu
inline void mat_to_array_(cv::Mat& input, array& output) {
    const unsigned size = input.rows * input.cols;
    const unsigned w = input.cols;
    const unsigned h = input.rows;
    const unsigned channels = input.channels();
    if (channels == 1) {
        // bw
        input.convertTo(input, CV_32F);
        output = array(w, h, input.ptr<float>(0)).T();
    } else if (channels == 3) {
        // color
        input.convertTo(input, CV_32FC3);
        vector<Mat> rgb; split(input, rgb);
        output = array(h, w, 3);
        output(span, span, 0) = array(w, h, rgb[2].ptr<float>(0)).T();
        output(span, span, 1) = array(w, h, rgb[1].ptr<float>(0)).T();
        output(span, span, 2) = array(w, h, rgb[0].ptr<float>(0)).T();
    } else {
        // 2 ch
        input.convertTo(input, CV_32FC2);
        vector<Mat> gb; split(input, gb);
        output = array(h, w, 2);
        output(span, span, 0) = array(w, h, gb[1].ptr<float>(0)).T();
        output(span, span, 1) = array(w, h, gb[0].ptr<float>(0)).T();
    }
}

// mem layout for gpu
void mat_to_array(const cv::Mat& input, array& output) {
    cv::Mat tmp = input.clone();
    mat_to_array_(tmp, output);
}

array mat_to_array(const cv::Mat& input) {
    array output;
    cv::Mat mtmp = input.clone();
    mat_to_array_(mtmp, output);
    return output;
}



// ==================================================================

// mem layout for cpu
void array_to_mat(const array& input_, cv::Mat& output, int type) {
    const int channels = input_.dims(2);
    int ndims = input_.numdims();
    array input;
    if (channels == 1) input = input_.T();
    else {
        input = zero(channels, input_.dims(1), input_.dims(0));
        gfor (array ii, channels) { input(channels - ii - 1, span, span) = input_(span, span, ii).T(); }
    }
    output = cv::Mat(input.dims(ndims - 1), input.dims(ndims - 2), CV_MAKETYPE(type, channels));
    if (type == CV_32F) {
        float* data = output.ptr<float>(0);
        input.host((void*)data);
    } else if (type == CV_64F) {
        double* data = output.ptr<double>(0);
        input.as(af::f64).host((void*)data);
    } else if (type == CV_8U) {
        uchar* data = output.ptr<uchar>(0);
        input.as(af::b8).host((void*)data);
    } else { MSG("array_to_mat: invalid conversion"); }
}

Mat array_to_mat(const array& input, int type) {
    cv::Mat output;
    array_to_mat(input, output, type);
    return output;
}

