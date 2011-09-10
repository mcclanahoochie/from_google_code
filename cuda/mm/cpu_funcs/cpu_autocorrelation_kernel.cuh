/*
 * gpu_autocorrelation_kernel.cuh
 *
 *  Created on: Feb 23, 2010
 *      Author: chris
 */

#ifndef __CPU_AUTOCORR_KERNEL_H__
#define __CPU_AUTOCORR_KERNEL_H__

#include <thrust/host_vector.h>

void  cpu_extract_xyz( float* d_xyz, const int validBodies, float *d_x, float *d_y, float *d_z );
float cpu_compute_autocorrelation(thrust::host_vector<float>& data_t1, thrust::host_vector<float>& data_t2, int N, int type);
float cpu_compute_autocorrelation(thrust::host_vector<int>& data_t1, thrust::host_vector<int>& data_t2, int N, int type);

#endif // #define __GPU_AUTOCORR_KERNEL_H__
