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
 * gpu_bonds.cu
 *
 *  Created on: Feb 18, 2010
 *      Author: chris
 */

/////////////////////////////////////
// standard imports
/////////////////////////////////////
#include <stdio.h>
#include <math.h>
#include <iostream>
#include "timer.h"
#include "gpu_bonds_kernel.cuh"
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

// neighbor list size per atom
#define NLISTHEIGHT  7

/////////////////////////////////////
// global variables
//   d_ = device (GPU)
//   h_ = host   (CPU)
/////////////////////////////////////
thrust::device_vector<float> d_gpu_xyz;    // gpu storage of xyz data
thrust::device_vector<float> d_gpu_nbonds; // gpu storage of bonds count
thrust::device_vector<int> d_gpu_bins;  // gpu storage of bins
thrust::device_vector<int> d_gpu_nlist; // gpu storage of neighbor list
thrust::host_vector<int> h_gpu_bins;  // host storage of gpu bins
thrust::host_vector<int> h_gpu_nlist; // host storage of gpu neighbor list

///////////////////////////////////////
// function declarations
///////////////////////////////////////
//#include "gpu_funcs.h" // shared library
int gpu_bonds_computation(int N, float* h_xyz, float rmin, float rmax, float radmax, float rdiv, int nbins);
int run_bonds_kernel_gpu(int N, float* h_xyz, float rmin, float rmax, float radmax, float rdiv, int nbins);
void print_results_gpu(int gpu_nbonds_sum, int N);
void _init_gpu(int N, int nbins);
void prepare_output_gpu(int natoms, int nbins, int** nblist_out, int nbonds_sum, int** bins_out);

////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
//
// callable external function
//
////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////
extern "C"
{
    /////////////////////////////////////
    // ENTRY POINT:
    //  h_xyz 	= incoming point float array
    //  N 		= size of float array (na3)
    //  rmin 	= min bond distance
    //  rmax 	= max bond distance
    //  maxrad  = max radial distribution distance for histogram; range[0,radmax]
    //  nbins   = number of radial distribution histogram bins
    //  nblist  = bond neighbor list in pairwise format
    //  bins    = histogram of bond radial distances; see maxrad
    /////////////////////////////////////
    int compute_bonds_gpu(float* h_xyz, int N, float rmin, float rmax, float maxrad, int nbins, int** nblist_out, int** bins_out) {
        // timing
#if TIME_TRIAL
        start_timer(1);
#endif
        // initialization
        _init_gpu(N, nbins);

        // gpu computation
        int gpu_nbonds_sum = gpu_bonds_computation(N, h_xyz, rmin, rmax, maxrad, (maxrad / nbins), nbins);

        // output neighbor list and histogram bins
        prepare_output_gpu((N / 3), nbins, nblist_out, gpu_nbonds_sum, bins_out);

        // timing
#if TIME_TRIAL
        printf(" total time (bonds) %.3f\n", elapsed_time(1));
#endif
#if DEBUG_PRINT
        // print results
        print_results_gpu(gpu_nbonds_sum, N);
#endif
        // return gpu ndongs sum
        return gpu_nbonds_sum;
    }
}

/////////////////////////////////////
// bonds manager
/////////////////////////////////////
int gpu_bonds_computation(int N, float* h_xyz, float rmin, float rmax, float radmax, float rdiv, int nbins) {
#if DEBUG_PRINT
    printf("gpu...\n");
#endif
    // setup
    int gpu_nbonds_sum  = 0;

    // timing
#if TIME_TRIAL
    start_timer(2);
#endif

    // calculate: bonds, neighbor list, neighbor distance histogram
    gpu_nbonds_sum = run_bonds_kernel_gpu(N, h_xyz, rmin, rmax, radmax, rdiv, nbins);

    // get neighbors data from gpu
    thrust::copy(d_gpu_nlist.begin(), d_gpu_nlist.end(), h_gpu_nlist.begin());

    // get histogram results from gpu
    thrust::copy(d_gpu_bins.begin(),  d_gpu_bins.end(),  h_gpu_bins.begin());

    // timing
#if TIME_TRIAL
    printf(" gpu   time (bonds) %.3f\n", elapsed_time(2));
#endif

    // final answer
    return gpu_nbonds_sum;
}

/////////////////////////////////////
// bonds kernel manager
/////////////////////////////////////
int run_bonds_kernel_gpu(int N, float* h_xyz, float rmin, float rmax, float radmax, float rdiv, int nbins) {
    // copy data to the device
    d_gpu_xyz.resize(N);
    thrust::copy(h_xyz, h_xyz + N, d_gpu_xyz.begin());

    // create device pointers
    float* d_xyz    = thrust::raw_pointer_cast(&d_gpu_xyz[0]);
    float* d_nbonds = thrust::raw_pointer_cast(&d_gpu_nbonds[0]);
    int* d_bins  = thrust::raw_pointer_cast(&d_gpu_bins[0]);
    int* d_nlist = thrust::raw_pointer_cast(&d_gpu_nlist[0]);

    // perform computation on the gpu
    gpu_compute_bonds(d_xyz, rmin, rmax, radmax, d_nbonds, rdiv, d_bins, nbins, (N / 3), d_nlist, NLISTHEIGHT);

    // reduce device bond count result to a single number (slow?)
    int gpu_nbonds_sum = (int)thrust::reduce(d_gpu_nbonds.begin(), d_gpu_nbonds.end(), (float)0, thrust::plus<float>());

    // return answer
    return gpu_nbonds_sum;
}

/////////////////////////////////////
// memory initialization
/////////////////////////////////////
void _init_gpu(int N, int nbins) {
    // miscellaneous initialization
    int nbListSize = NLISTHEIGHT * (N / 3);

    // bonds memory
    d_gpu_nbonds.resize(N + 3);
    thrust::fill(d_gpu_nbonds.begin(), d_gpu_nbonds.end(), (float)0);

    // bins memory
    d_gpu_bins.resize(nbins);
    thrust::fill(d_gpu_bins.begin(), d_gpu_bins.end(), (int)0);
    h_gpu_bins.resize(nbins);
    thrust::fill(h_gpu_bins.begin(), h_gpu_bins.end(), (int)0);

    // neighbors memory
    d_gpu_nlist.resize(nbListSize);
    thrust::fill(d_gpu_nlist.begin(), d_gpu_nlist.end(), (int)EMPTY);
    h_gpu_nlist.resize(nbListSize);
    thrust::fill(h_gpu_nlist.begin(), h_gpu_nlist.end(), (int)EMPTY);
}

/////////////////////////////////////
// display results
/////////////////////////////////////
void prepare_output_gpu(int natoms, int nbins, int** nblist_out, int nbonds_sum, int** bins_out) {
    // neighbor list
    int* nblist_tmp = (int*)malloc(nbonds_sum * 2 * sizeof(int));
    int width = natoms;
    int added_bonds = 0;
    int n = 0;
    int h = 0;
    int idx = 0;
    for (n = 0; n < width; ++n) {
        for (h = 0; h < NLISTHEIGHT; ++h) {
            idx = h * width + n;
            if (h_gpu_nlist[idx] != EMPTY) {
                nblist_tmp[added_bonds  ] = n;
                nblist_tmp[added_bonds + 1] = h_gpu_nlist[idx];
                added_bonds += 2;
            }
        }
    }

    // histogram
    int* bins_tmp = (int*)malloc(nbins * sizeof(int));
    if (bins_tmp == NULL) {
        printf(" gpu bonds: bad malloc - bins \n");
        return;
    }
    memcpy(bins_tmp, &h_gpu_bins[0], nbins * sizeof(int));

    // output pointers
    *nblist_out = nblist_tmp;
    *bins_out = bins_tmp;
}

/////////////////////////////////////
// display results
/////////////////////////////////////
void print_results_gpu(int gpu_nbonds_sum, int N) {


#if 0
    // print BINS results
    int mbin = 0;
    int midx = 0;
    int binval = 0;
    std::cout << "\n d_bins (nbins=" << NBINS << " rdiv=" << rdiv << ")" << std::endl;
    for (int i = 0; i < NBINS; ++i) {
        binval = h_gpu_bins[i] ; // / 2;
        printf(" bin[%d](%f-%f)=%d \n", i, i * rdiv, (i + 1)*rdiv, binval);
        // find median bin
        if (binval > mbin) {
            mbin = binval;
            midx = i;
        }
    }
    printf("\n  median: bin[%d]=%d \n", midx, mbin);
#endif // print bins

#if 0
    // print neighbors
    int width = (N / 3);
    printf("\n\nnlist\n\n");
    for (int n = 0; n < width; ++n) {
        printf(" gpu[%d] ", n);
        for (int h = 0; h < NLISTHEIGHT; ++h) {
            int idx = h * width + n;
            if (h_gpu_nlist[idx] != EMPTY) { printf("%d ", h_gpu_nlist[idx]); }
        }
        printf("\n");
    }
#endif // print neighbors

    // print nbonds
    printf("~~~~ bonds  %d ~~~~ \n", gpu_nbonds_sum);
}

