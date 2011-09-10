/*
 * cpu_autocorrelation.cuh
 *
 *  Created on: Feb 23, 2010
 *      Author: chris
 */

/////////////////////////////////////
// standard imports
/////////////////////////////////////
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "timer.h"
#include "cpu_autocorrelation_kernel.cuh"
#include "cpu_common.cuh"

/////////////////////////////////////
// Thrust imports
/////////////////////////////////////
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
//   h_ = host (cpu)
//   h_ = host   (CPU)
/////////////////////////////////////
thrust::host_vector<float> h_cpu_xyz_ac; // cpu storage of xyz data
thrust::host_vector<float> h_cpu_x;   // cpu storage of x data
thrust::host_vector<float> h_cpu_y;   // cpu storage of y data
thrust::host_vector<float> h_cpu_z;   // cpu storage of z data
thrust::host_vector<int  > h_cpu_i;   // cpu storage of int data
thrust::host_vector<float> h_cpu_x_old; // cpu storage of old x data
thrust::host_vector<float> h_cpu_y_old; // cpu storage of old y data
thrust::host_vector<float> h_cpu_z_old; // cpu storage of old z data
thrust::host_vector<int  > h_cpu_i_old;   // cpu storage of int data

///////////////////////////////////////
// function declarations
///////////////////////////////////////
//#include "cpu_funcs.h" // shared library
void cpu_xyz_ac_computation(int N, float* h_xyz, float& oacx, float& oacy, float& oacz, int type);
void run_xyz_extraction_kernel_cpu(int N, float* h_xyz);
void run_xyz_autocorrelation_kernel_cpu(int N, float& oacx, float& oacy, float& oacz, int type);
void _init_xyz_cpu(int N);
void cpu_int_ac_computation(int N, int* h_i, float& oaci, int type);
void run_int_autocorrelation_kernel_cpu(int validBodies, float& oaci, int type);

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
    int compute_xyz_autocorrelation_cpu(float* h_xyz, int N, float& oacx, float& oacy, float& oacz, int type) {
        // timing:
#if TIME_TRIAL
        start_timer(1);
#endif
        // initialize memory
        _init_xyz_cpu(N);

        // cpu computation
        cpu_xyz_ac_computation(N, h_xyz, oacx, oacy, oacz, type);

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
    int compute_int_autocorrelation_cpu(int* h_i, int N, float& oaci, int type) {
        // timing:
#if TIME_TRIAL
        start_timer(4);
#endif
        // initialize memory
        h_cpu_i.resize(N);
        //thrust::fill(h_cpu_i.begin(), h_cpu_i.end(), (int)0);
        thrust::copy(h_i, h_i + N, h_cpu_i.begin());

        // cpu computation
        cpu_int_ac_computation(N, h_i, oaci, type);

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
void cpu_xyz_ac_computation(int N, float* h_xyz, float& oacx, float& oacy, float& oacz, int type) {
#if DEBUG_PRINT
    printf("cpu...\n");
#endif
    // timing
#if TIME_TRIAL
    start_timer(2);
#endif

    // extract xyz vectors
    run_xyz_extraction_kernel_cpu(N, h_xyz);

    // save first timestamp
    static int first_run = 1;
    if (first_run) {
        first_run = 0;
        h_cpu_x_old = h_cpu_x;
        h_cpu_y_old = h_cpu_y;
        h_cpu_z_old = h_cpu_z;
    }

    // autocorrelation
    run_xyz_autocorrelation_kernel_cpu(N / 3, oacx, oacy, oacz, type);

    // update old timestamp data if doing adjacent comparisons
    if (DO_AC_ADJ) {
        h_cpu_x_old = h_cpu_x;
        h_cpu_y_old = h_cpu_y;
        h_cpu_z_old = h_cpu_z;
    }

    // timing
#if TIME_TRIAL
    printf(" cpu   time (xyz autocorrelation) %.3f\n", elapsed_time(2));
#endif

}

/////////////////////////////////////
// autocorr manager
/////////////////////////////////////
void cpu_int_ac_computation(int N, int* h_i, float& oaci, int type) {
#if DEBUG_PRINT
    printf("cpu...\n");
#endif
    // timing
#if TIME_TRIAL
    start_timer(3);
#endif

    // save first timestamp
    static int first_run = 1;
    if (first_run) {
        first_run = 0;
        h_cpu_i_old = h_cpu_i;
    }

    // autocorrelation
    run_int_autocorrelation_kernel_cpu(N, oaci, type);

    // update old timestamp data if doing adjacent comparisons
    if (DO_AC_ADJ) {
        h_cpu_i_old = h_cpu_i;
    }

    // timing
#if TIME_TRIAL
    printf(" cpu   time (i autocorrelation) %.3f\n", elapsed_time(3));
#endif

}

/////////////////////////////////////
// data extraction kernel manager
/////////////////////////////////////
void run_xyz_extraction_kernel_cpu(int N, float* h_xyz) {
    // copy data to the host
    h_cpu_xyz_ac.resize(N);
    thrust::copy(h_xyz, h_xyz + N, h_cpu_xyz_ac.begin());

    // create host pointers
    float* h_xyz2 = thrust::raw_pointer_cast(&h_cpu_xyz_ac[0]);
    float* h_x    = thrust::raw_pointer_cast(&h_cpu_x[0]);
    float* h_y    = thrust::raw_pointer_cast(&h_cpu_y[0]);
    float* h_z    = thrust::raw_pointer_cast(&h_cpu_z[0]);

    // perform computation on the cpu
    cpu_extract_xyz(h_xyz2, N / 3, h_x, h_y, h_z);
}

/////////////////////////////////////
// autocorr kernel manager
/////////////////////////////////////
void run_xyz_autocorrelation_kernel_cpu(int validBodies, float& oacx, float& oacy, float& oacz, int type) {
    // do the autocorreolation
    float acx = cpu_compute_autocorrelation(h_cpu_x, h_cpu_x_old, validBodies, type);
    float acy = cpu_compute_autocorrelation(h_cpu_y, h_cpu_y_old, validBodies, type);
    float acz = cpu_compute_autocorrelation(h_cpu_z, h_cpu_z_old, validBodies, type);

    // output results
    oacx = acx;
    oacy = acy;
    oacz = acz;
}

/////////////////////////////////////
// autocorr kernel manager
/////////////////////////////////////
void run_int_autocorrelation_kernel_cpu(int validBodies, float& oaci, int type) {
    // do the autocorreolation
    float aci = cpu_compute_autocorrelation(h_cpu_i, h_cpu_i_old, validBodies, type);

    // output results
    oaci = aci;
}

/////////////////////////////////////
// memory initialization
/////////////////////////////////////
void _init_xyz_cpu(int N) {
    // miscellaneous initialization
    int validBodies = N / 3;

    // x y z memory
    h_cpu_x.resize(validBodies);
    thrust::fill(h_cpu_x.begin(), h_cpu_x.end(), (float)0);
    h_cpu_y.resize(validBodies);
    thrust::fill(h_cpu_y.begin(), h_cpu_y.end(), (float)0);
    h_cpu_z.resize(validBodies);
    thrust::fill(h_cpu_z.begin(), h_cpu_z.end(), (float)0);
}

