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


/*
 * gpu_autocorrelation_kernel.cu
 *
 *  Created on: Feb 19, 2010
 *      Author: chris
 */

/////////////////////////////////////
// imports
/////////////////////////////////////
#include "gpu_common.cuh"
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>

/////////////////////////////////////
// data extraction kernel
/////////////////////////////////////
__global__ void gpu_extract_xyz_kernel(int N, void* outAx, void* outAy, void* outAz) {
    float*  global_nbondsx = (float*)outAx;
    float*  global_nbondsy = (float*)outAy;
    float*  global_nbondsz = (float*)outAz;
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gtid < N) {
        // extract xyz array into 3 vectors
        int texidx = gtid * 3; // gpu xyz texture mem
        float3 xyz = { tex1Dfetch(xyz_tex, texidx + 0), tex1Dfetch(xyz_tex, texidx + 1), tex1Dfetch(xyz_tex, texidx + 2) };
        global_nbondsx[gtid] = xyz.x;
        global_nbondsy[gtid] = xyz.y;
        global_nbondsz[gtid] = xyz.z;
    }
}

/////////////////////////////////////
// external data extraction kernel manager
/////////////////////////////////////
void gpu_extract_xyz(float* d_xyz, const int validBodies, float* d_x, float* d_y, float* d_z) {
    // map xyz data to texture
    cudaBindTexture(0, xyz_tex, d_xyz, validBodies * 3 * sizeof(float));

    // setup sizes
    int p = numThreadsPerBlock;
    int val = (int)ceil(validBodies / p);
    dim3 nthreads(p, 1, 1);
    dim3 nblocks(val, 1, 1);

    // run kernel - compute on gpu
    gpu_extract_xyz_kernel <<< nblocks, nthreads >>>(validBodies, d_x, d_y, d_z);

    // unmap texture
    cudaUnbindTexture(xyz_tex);
}

/////////////////////////////////////
// core coviariance vector calculations (float)
/////////////////////////////////////
struct cov_functor_f1 {
    const float u1, u2;

    cov_functor_f1(float _u1, float _u2) : u1(_u1), u2(_u2) {}

    __host__ __device__
    float operator()(const float& t1, const float& t2) const {
        return (t1 - u1) * (t2 - u2);
    }
};

struct cov_functor_f2 {
    cov_functor_f2() {}

    __host__ __device__
    float operator()(const float& t1, const float& t2) const {
        return ((t1 - t2) * (t1 - t2));
    }
};

/////////////////////////////////////
// external compute autocorrelation between datasets at time1 and time2 (float)
/////////////////////////////////////
float gpu_compute_autocorrelation(thrust::device_vector<float>& data_t1, thrust::device_vector<float>& data_t2, int N, int type) {
    // temp
    thrust::device_vector<float> data_r(N);
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

    __host__ __device__
    float operator()(const int& t1, const int& t2) const {
        return (t1 - u1) * (t2 - u2);
    }
};

struct cov_functor_i2 {
    cov_functor_i2() {}

    __host__ __device__
    float operator()(const int& t1, const int& t2) const {
        return ((t1 - t2) * (t1 - t2));
    }
};

/////////////////////////////////////
// external compute autocorrelation between datasets at time1 and time2 (int)
/////////////////////////////////////
float gpu_compute_autocorrelation(thrust::device_vector<int>& data_t1, thrust::device_vector<int>& data_t2, int N, int type) {
    // temp
    thrust::device_vector<float> data_r(N);
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

