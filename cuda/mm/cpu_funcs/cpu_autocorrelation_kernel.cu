/*
 * cpu_autocorrelation_kernel.cu
 *
 *  Created on: Feb 23, 2010
 *      Author: chris
 */

/////////////////////////////////////
// imports
/////////////////////////////////////
#include "cpu_common.cuh"

#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>

/////////////////////////////////////
// data extraction kernel
/////////////////////////////////////
__global__ void cpu_extract_xyz_kernel(int N, void* outAx, void* outAy, void* outAz) {

}

/////////////////////////////////////
// external data extraction kernel manager
/////////////////////////////////////
void cpu_extract_xyz(float* h_xyz, const int validBodies, float* h_x, float* h_y, float* h_z) {

    for (int i = 0; i < validBodies; ++i) {
        h_x[i] = h_xyz[i * 3 + 0];
        h_y[i] = h_xyz[i * 3 + 1];
        h_z[i] = h_xyz[i * 3 + 2];
    }

}

/////////////////////////////////////
// core coviariance vector calculations (float)
/////////////////////////////////////
struct cov_functor_f1 {
    const float u1, u2;

    cov_functor_f1(float _u1, float _u2) : u1(_u1), u2(_u2) {}

    __host__ __host__
    float operator()(const float& t1, const float& t2) const {
        return (t1 - u1) * (t2 - u2);
    }
};

struct cov_functor_f2 {
    cov_functor_f2() {}

    __host__ __host__
    float operator()(const float& t1, const float& t2) const {
        return ((t1 - t2) * (t1 - t2));
    }
};

/////////////////////////////////////
// external compute autocorrelation between datasets at time1 and time2 (float)
/////////////////////////////////////
float cpu_compute_autocorrelation(thrust::host_vector<float>& data_t1, thrust::host_vector<float>& data_t2, int N, int type) {
    // temp
    thrust::host_vector<float> data_r(N);
    float ac = 0.0f;

    switch (type) {
    case 1: {
        // http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient

        // find means
        float u1 = thrust::reduce(data_t1.begin(), data_t1.end(), (float)0, thrust::plus<float>()) / N;
        float u2 = thrust::reduce(data_t2.begin(), data_t2.end(), (float)0, thrust::plus<float>()) / N;

        // r = (t1-u1)*(t2-u2)
        thrust::transform(data_t1.begin(), data_t1.end(), data_t2.begin(), data_r.begin(), cov_functor_f1(u1, u2));

        // cov = sum(r_vector) / n
        float cov = thrust::reduce(data_r.begin(), data_r.end(), (float)0, thrust::plus<float>()) / N;

        // variances
        thrust::transform(data_t1.begin(), data_t1.end(), data_t1.begin(), data_r.begin(), cov_functor_f1(u1, u1));
        float var1 = thrust::reduce(data_r.begin(), data_r.end(), (float)0, thrust::plus<float>()) / N;
        thrust::transform(data_t2.begin(), data_t2.end(), data_t2.begin(), data_r.begin(), cov_functor_f1(u2, u2));
        float var2 = thrust::reduce(data_r.begin(), data_r.end(), (float)0, thrust::plus<float>()) / N;

        // standard deviations
        float std1 = sqrt(var1);
        float std2 = sqrt(var2);

        // autocorrelation
        ac = cov / (std1 * std2);
        break;
    }
    case 2: {
        // http://en.wikipedia.org/wiki/Durbin-Watson_statistic

        // r = (t1-t2)^2
        thrust::transform(data_t1.begin(), data_t1.end(), data_t2.begin(), data_r.begin(), cov_functor_f2());
        float et2 = thrust::reduce(data_r.begin(), data_r.end(), (float)0, thrust::plus<float>()) ;

        // r = t1^2
        thrust::transform(data_t1.begin(), data_t1.end(), data_t1.begin(), data_r.begin(), cov_functor_f1(0, 0));
        float et1 = thrust::reduce(data_r.begin(), data_r.end(), (float)0, thrust::plus<float>()) ;

        // autocorrelation
        ac = 1 - et2 / et1;
        break;
    }

    } // end case

    // out
    return ac;
}

/////////////////////////////////////
// core coviariance vector calculations (int)
/////////////////////////////////////
struct cov_functor_i1 {
    const float u1, u2;

    cov_functor_i1(float _u1, float _u2) : u1(_u1), u2(_u2) {}

    __host__ __host__
    float operator()(const int& t1, const int& t2) const {
        return (t1 - u1) * (t2 - u2);
    }
};

struct cov_functor_i2 {
    cov_functor_i2() {}

    __host__ __host__
    float operator()(const int& t1, const int& t2) const {
        return ((t1 - t2) * (t1 - t2));
    }
};

/////////////////////////////////////
// external compute autocorrelation between datasets at time1 and time2 (int)
/////////////////////////////////////
float cpu_compute_autocorrelation(thrust::host_vector<int>& data_t1, thrust::host_vector<int>& data_t2, int N, int type) {
    // temp
    thrust::host_vector<float> data_r(N);
    float ac = 0.0f;

    switch (type) {
    case 1: {
        // http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient

        // find means
        float u1 = thrust::reduce(data_t1.begin(), data_t1.end(), (int)0, thrust::plus<int>()) / N;
        float u2 = thrust::reduce(data_t2.begin(), data_t2.end(), (int)0, thrust::plus<int>()) / N;

        // r = (t1-u1)*(t2-u2)
        thrust::transform(data_t1.begin(), data_t1.end(), data_t2.begin(), data_r.begin(), cov_functor_i1(u1, u2));

        // cov = sum(r_vector) / n
        float cov = thrust::reduce(data_r.begin(), data_r.end(), (float)0, thrust::plus<float>()) / N;

        // variances
        thrust::transform(data_t1.begin(), data_t1.end(), data_t1.begin(), data_r.begin(), cov_functor_f1(u1, u1));
        float var1 = thrust::reduce(data_r.begin(), data_r.end(), (float)0, thrust::plus<float>()) / N;
        thrust::transform(data_t2.begin(), data_t2.end(), data_t2.begin(), data_r.begin(), cov_functor_f1(u2, u2));
        float var2 = thrust::reduce(data_r.begin(), data_r.end(), (float)0, thrust::plus<float>()) / N;

        // standard deviations
        float std1 = sqrt(var1);
        float std2 = sqrt(var2);

        // autocorrelation
        ac = cov / (std1 * std2);
        break;
    }
    case 2: {
        // http://en.wikipedia.org/wiki/Durbin-Watson_statistic

        // r = (t1-t2)^2
        thrust::transform(data_t1.begin(), data_t1.end(), data_t2.begin(), data_r.begin(), cov_functor_i2());
        float et2 = thrust::reduce(data_r.begin(), data_r.end(), (float)0, thrust::plus<float>()) ;

        // r = t1^2
        thrust::transform(data_t1.begin(), data_t1.end(), data_t1.begin(), data_r.begin(), cov_functor_f1(0, 0));
        float et1 = thrust::reduce(data_r.begin(), data_r.end(), (float)0, thrust::plus<float>()) ;

        // autocorrelation
        ac = 1 - et2 / et1;
        break;
    }

    } // end case

    // out
    return ac;
}

