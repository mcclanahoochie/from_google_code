/*
 * gpu_autocorrelation_kernel.cuh
 *
 *  Created on: Feb 19, 2010
 *      Author: chris
 */

#ifndef __GPU_AUTOCORR_KERNEL_H__
#define __GPU_AUTOCORR_KERNEL_H__

#include <thrust/device_vector.h>

void  gpu_extract_xyz( float* d_xyz, const int validBodies, float *d_x, float *d_y, float *d_z );
float gpu_compute_autocorrelation(thrust::device_vector<float>& data_t1, thrust::device_vector<float>& data_t2, int N, int type);
float gpu_compute_autocorrelation(thrust::device_vector<int>& data_t1, thrust::device_vector<int>& data_t2, int N, int type);

#endif // #define __GPU_AUTOCORR_KERNEL_H__
