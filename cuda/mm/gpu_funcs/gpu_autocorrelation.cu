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
 * gpu_autocorrelation.cuh
 *
 *  Created on: Feb 19, 2010
 *      Author: chris
 */

/////////////////////////////////////
// standard imports
/////////////////////////////////////
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "timer.h"
#include "gpu_autocorrelation_kernel.cuh"
#include "gpu_common.cuh"

/////////////////////////////////////
// Thrust imports
/////////////////////////////////////
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/extrema.h>

/////////////////////////////////////
// configuration section
/////////////////////////////////////

//  1 -> autocorr for each adjacent timestep (lag=1)
//  0 -> compare all successive timesteps with first timestep (lag>1)
#define DO_AC_ADJ 0

/////////////////////////////////////
// global variables
//   d_ = device (GPU)
//   h_ = host   (CPU)
/////////////////////////////////////
thrust::device_vector<float> d_gpu_xyz_ac; // gpu storage of xyz data
thrust::device_vector<float> d_gpu_x;   // gpu storage of x data
thrust::device_vector<float> d_gpu_y;   // gpu storage of y data
thrust::device_vector<float> d_gpu_z;   // gpu storage of z data
thrust::device_vector<float> d_gpu_x_old;   // gpu storage of x data
thrust::device_vector<float> d_gpu_y_old;   // gpu storage of y data
thrust::device_vector<float> d_gpu_z_old;   // gpu storage of z data
thrust::device_vector<int> d_gpu_i;   	// gpu storage of int data
thrust::device_vector<int> d_gpu_i_old;   // gpu storage of int data
thrust::host_vector<float> h_gpu_x_old; // cpu storage of old x data
thrust::host_vector<float> h_gpu_y_old; // cpu storage of old y data
thrust::host_vector<float> h_gpu_z_old; // cpu storage of old z data
thrust::host_vector<int  > h_gpu_i_old; // cpu storage of int data

///////////////////////////////////////
// function declarations
///////////////////////////////////////
//#include "gpu_funcs.h" // shared library
void gpu_xyz_ac_computation(int N, float* h_xyz, float& oacx, float& oacy, float& oacz, int type);
void run_xyz_extraction_kernel_gpu(int N, float* h_xyz);
void run_xyz_autocorrelation_kernel_gpu(int N, float& oacx, float& oacy, float& oacz, int type);
void _init_xyz_gpu(int N);
void gpu_int_ac_computation(int N, int* h_i, float& oaci, int type);
void run_int_autocorrelation_kernel_gpu(int validBodies, float& oaci, int type);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// callable external functions
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
extern "C"
{
    /////////////////////////////////////
    // ENTRY POINT:
    //  h_xyz 	= incoming point float array
    //  N 		= size of float array (na3)
    //  oacx    = output autocorrelation for x
    //  oacy    = output autocorrelation for y
    //  oacz    = output autocorrelation for z
    //  type	= choose correlation kernel type
    /////////////////////////////////////
    int compute_xyz_autocorrelation_gpu(float* h_xyz, int N, float& oacx, float& oacy, float& oacz, int type) {
        // timing:
#if TIME_TRIAL
        start_timer(1);
#endif
        // initialize memory
        _init_xyz_gpu(N);

        // gpu computation
        gpu_xyz_ac_computation(N, h_xyz, oacx, oacy, oacz, type);

        // timing:
#if TIME_TRIAL
        printf(" total time (xyz autocorrelation) %.3f\n", elapsed_time(1));
#endif
#if DEBUG_PRINT
        // print results
        printf("~~~~ autocorrelation  x  %0.5f ~~~~ \n", oacx);
        printf("~~~~ autocorrelation  y  %0.5f ~~~~ \n", oacy);
        printf("~~~~ autocorrelation  z  %0.5f ~~~~ \n", oacz);
#endif
        // finish
        return 0;
    }

    /////////////////////////////////////
    // ENTRY POINT:
    //  h_i 	= incoming int array
    //  N 		= size of int array
    //  oaci    = output autocorrelation for int data
    /////////////////////////////////////
    int compute_int_autocorrelation_gpu(int* h_i, int N, float& oaci, int type) {
        // timing:
#if TIME_TRIAL
        start_timer(4);
#endif
        // initialize memory
        d_gpu_i.resize(N);
        //thrust::fill(d_gpu_i.begin(), d_gpu_i.end(), (int)0);
        thrust::copy(h_i, h_i + N, d_gpu_i.begin());

        // gpu computation
        gpu_int_ac_computation(N, h_i, oaci, type);

        // timing:
#if TIME_TRIAL
        printf(" total time (i autocorrelation) %.3f\n", elapsed_time(4));
#endif
#if DEBUG_PRINT
        // print results
        printf("~~~~ autocorrelation  i  %0.5f ~~~~ \n", oaci);
#endif
        // finish
        return 0;
    }

}

/////////////////////////////////////
// autocorr manager
/////////////////////////////////////
void gpu_xyz_ac_computation(int N, float* h_xyz, float& oacx, float& oacy, float& oacz, int type) {
#if DEBUG_PRINT
    printf("gpu...\n");
#endif
    // timing
#if TIME_TRIAL
    start_timer(2);
#endif

    // extract xyz vectors
    run_xyz_extraction_kernel_gpu(N, h_xyz);

    // save first timestamp
    static int first_run = 1;
    if (first_run) {
        first_run = 0;
        d_gpu_x_old = d_gpu_x;
        d_gpu_y_old = d_gpu_y;
        d_gpu_z_old = d_gpu_z;
    }

    // autocorrelation
    run_xyz_autocorrelation_kernel_gpu(N / 3, oacx, oacy, oacz, type);

    // update old timestamp data if doing adjacent comparisons
    if (DO_AC_ADJ) {
        d_gpu_x_old = d_gpu_x;
        d_gpu_y_old = d_gpu_y;
        d_gpu_z_old = d_gpu_z;
    }

    // timing
#if TIME_TRIAL
    printf(" gpu   time (xyz autocorrelation) %.3f\n", elapsed_time(2));
#endif

}

/////////////////////////////////////
// autocorr manager
/////////////////////////////////////
void gpu_int_ac_computation(int N, int* h_i, float& oaci, int type) {
#if DEBUG_PRINT
    printf("gpu...\n");
#endif
    // timing
#if TIME_TRIAL
    start_timer(3);
#endif

    // save first timestamp
    static int first_run = 1;
    if (first_run) {
        first_run = 0;
        d_gpu_i_old = d_gpu_i;
    }

    // autocorrelation
    run_int_autocorrelation_kernel_gpu(N, oaci, type);

    // update old timestamp data if doing adjacent comparisons
    if (DO_AC_ADJ) {
        d_gpu_i_old = d_gpu_i;
    }

    // timing
#if TIME_TRIAL
    printf(" gpu   time (i autocorrelation) %.3f\n", elapsed_time(3));
#endif

}

/////////////////////////////////////
// data extraction kernel manager
/////////////////////////////////////
void run_xyz_extraction_kernel_gpu(int N, float* h_xyz) {
    // copy data to the device
    d_gpu_xyz_ac.resize(N);
    thrust::copy(h_xyz, h_xyz + N, d_gpu_xyz_ac.begin());

    // create device pointers
    float* d_xyz = thrust::raw_pointer_cast(&d_gpu_xyz_ac[0]);
    float* d_x   = thrust::raw_pointer_cast(&d_gpu_x[0]);
    float* d_y   = thrust::raw_pointer_cast(&d_gpu_y[0]);
    float* d_z   = thrust::raw_pointer_cast(&d_gpu_z[0]);

    // perform computation on the gpu
    gpu_extract_xyz(d_xyz, N / 3, d_x, d_y, d_z);
}

/////////////////////////////////////
// autocorr kernel manager
/////////////////////////////////////
void run_xyz_autocorrelation_kernel_gpu(int validBodies, float& oacx, float& oacy, float& oacz, int type) {
    // do the autocorreolation
    float acx = gpu_compute_autocorrelation(d_gpu_x_old, d_gpu_x, validBodies, type);
    float acy = gpu_compute_autocorrelation(d_gpu_y_old, d_gpu_y, validBodies, type);
    float acz = gpu_compute_autocorrelation(d_gpu_z_old, d_gpu_z, validBodies, type);

    // output results
    oacx = acx;
    oacy = acy;
    oacz = acz;
}

/////////////////////////////////////
// autocorr kernel manager
/////////////////////////////////////
void run_int_autocorrelation_kernel_gpu(int validBodies, float& oaci, int type) {
    // do the autocorreolation
    float aci = gpu_compute_autocorrelation(d_gpu_i_old, d_gpu_i, validBodies, type);

    // output results
    oaci = aci;
}

/////////////////////////////////////
// memory initialization
/////////////////////////////////////
void _init_xyz_gpu(int N) {
    // miscellaneous initialization
    int validBodies = N / 3;

    // x y z memory
    d_gpu_x.resize(validBodies);
    thrust::fill(d_gpu_x.begin(), d_gpu_x.end(), (float)0);
    d_gpu_y.resize(validBodies);
    thrust::fill(d_gpu_y.begin(), d_gpu_y.end(), (float)0);
    d_gpu_z.resize(validBodies);
    thrust::fill(d_gpu_z.begin(), d_gpu_z.end(), (float)0);
}

