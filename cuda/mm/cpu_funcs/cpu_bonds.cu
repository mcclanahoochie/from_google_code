/*
 * cpu_bonds.cu
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

// neighbor list size per atom
#define NLISTHEIGHT  7

/////////////////////////////////////
// global variables
//   d_ = host (GPU)
//   h_ = host (CPU)
/////////////////////////////////////
thrust::host_vector<float> h_cpu_xyz;    // cpu storage of xyz data
thrust::host_vector<float> h_cpu_nbonds; // cpu storage of bonds count
thrust::host_vector<int> h_cpu_bins;  // cpu storage of bins
thrust::host_vector<int> h_cpu_nlist; // cpu storage of neighbor list

///////////////////////////////////////
// function declarations
///////////////////////////////////////
//#include "cpu_funcs.h" // shared library
int cpu_bonds_computation(int N, float* h_xyz, float rmin, float rmax, float radmax, float rdiv, int nbins);
int run_bonds_kernel_cpu(int N, float* h_xyz, float rmin, float rmax, float radmax, float rdiv, int nbins);
void print_results_cpu(int cpu_nbonds_sum, int N);
void _init_cpu(int N, int nbins);
void prepare_output_cpu(int natoms, int nbins, int** nblist_out, int nbonds_sum, int** bins_out);

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
    int compute_bonds_cpu(float* h_xyz, int N, float rmin, float rmax, float maxrad, int nbins, int** nblist_out, int** bins_out) {
        // timing
#if TIME_TRIAL
        start_timer(1);
#endif
        // initialization
        _init_cpu(N, nbins);

        // cpu computation
        int cpu_nbonds_sum = cpu_bonds_computation(N, h_xyz, rmin, rmax, maxrad, (maxrad / nbins), nbins);

        // output neighbor list and histogram bins
        prepare_output_cpu((N / 3), nbins, nblist_out, cpu_nbonds_sum, bins_out);

        // timing
#if TIME_TRIAL
        printf(" total time (bonds) %.3f\n", elapsed_time(1));
#endif
#if DEBUG_PRINT
        // print results
        print_results_cpu(cpu_nbonds_sum, N);
#endif
        // return cpu ndongs sum
        return cpu_nbonds_sum;
    }
}

/////////////////////////////////////
// bonds manager
/////////////////////////////////////
int cpu_bonds_computation(int N, float* h_xyz, float rmin, float rmax, float radmax, float rdiv, int nbins) {
#if DEBUG_PRINT
    printf("cpu...\n");
#endif

    // setup
    int cpu_nbonds_sum  = 0;

    // timing
#if TIME_TRIAL
    start_timer(2);
#endif

    // calculate: bonds, neighbor list, neighbor distance histogram
    cpu_nbonds_sum = run_bonds_kernel_cpu(N, h_xyz, rmin, rmax, radmax, rdiv, nbins);

    // timing
#if TIME_TRIAL
    printf(" cpu   time (bonds) %.3f\n", elapsed_time(2));
#endif

    // final answer
    return cpu_nbonds_sum;
}

/////////////////////////////////////
// bonds kernel manager
/////////////////////////////////////
int run_bonds_kernel_cpu(int N, float* h_xyz, float rmin, float rmax, float radmax, float rdiv, int nbins) {

    // init
    int nbonds = 0;
    int i, j;
    float dx, dy, dz, dist;
    // neighbors
    int h = 0;
    int width = (N / 3);
    // bins
    int bin;
    for (i = 0; i < nbins; ++i) { h_cpu_bins[i] = 0; }
    // nbody
    for (i = 0; i < N - 2; i += 3) {
        h = 0;
        for (j = i + 3; j < N - 2; j += 3) {
            // dist
            dx = h_xyz[i + 0] - h_xyz[j + 0];
            dy = h_xyz[i + 1] - h_xyz[j + 1];
            dz = h_xyz[i + 2] - h_xyz[j + 2];
            dist = sqrtf(dx * dx + dy * dy + dz * dz);
            // bonds
            if (rmin < dist && dist < rmax) {
                // bond count
                ++nbonds;
                // neighbors
                if (h < NLISTHEIGHT) {
                    h_cpu_nlist[h * width + (i / 3)] = (j / 3);
                    ++h;
                }
            }
            // bins
            if (dist < radmax) {
                bin = (int)floor((dist - rmin) / rdiv);
                bin = (bin >= nbins) ? (nbins) : (bin < 0) ? (nbins) : bin;
                h_cpu_bins[bin] += 1;
            }
        }
    }
    int cpu_nbonds_sum = nbonds;

    // return answer
    return cpu_nbonds_sum;
}

/////////////////////////////////////
// memory initialization
/////////////////////////////////////
void _init_cpu(int N, int nbins) {
    // miscellaneous initialization
    int nbListSize = NLISTHEIGHT * (N / 3);

    // bonds memory
    h_cpu_nbonds.resize(N + 3);
    thrust::fill(h_cpu_nbonds.begin(), h_cpu_nbonds.end(), (float)0);

    // bins memory
    h_cpu_bins.resize(nbins);
    thrust::fill(h_cpu_bins.begin(), h_cpu_bins.end(), (int)0);

    // neighbors memory
    h_cpu_nlist.resize(nbListSize);
    thrust::fill(h_cpu_nlist.begin(), h_cpu_nlist.end(), (int)EMPTY);
}

/////////////////////////////////////
// display results
/////////////////////////////////////
void prepare_output_cpu(int natoms, int nbins, int** nblist_out, int nbonds_sum, int** bins_out) {
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
            if (h_cpu_nlist[idx] != EMPTY) {
                nblist_tmp[added_bonds  ] = n;
                nblist_tmp[added_bonds + 1] = h_cpu_nlist[idx];
                added_bonds += 2;
            }
        }
    }

    // histogram
    int* bins_tmp = (int*)malloc(nbins * sizeof(int));
    if (bins_tmp == NULL) {
        printf(" cpu bonds: bad malloc - bins \n");
        return;
    }
    printf("here 1\n");
    //memcpy(bins_tmp,&h_cpu_bins,nbins*sizeof(int));
    for (n = 0; n < nbins; ++n) { bins_tmp[n] = h_cpu_bins[n]; }
    printf("here 2\n");
    // output pointers
    *nblist_out = nblist_tmp;
    *bins_out = bins_tmp;
}

/////////////////////////////////////
// display results
/////////////////////////////////////
void print_results_cpu(int cpu_nbonds_sum, int N) {


#if 0
    // print BINS results
    int mbin = 0;
    int midx = 0;
    int binval = 0;
    std::cout << "\n d_bins (nbins=" << NBINS << " rdiv=" << rdiv << ")" << std::endl;
    for (int i = 0; i < NBINS; ++i) {
        binval = h_cpu_bins[i] ; // / 2;
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
        printf(" cpu[%d] ", n);
        for (int h = 0; h < NLISTHEIGHT; ++h) {
            int idx = h * width + n;
            if (h_cpu_nlist[idx] != EMPTY) { printf("%d ", h_gpu_nlist[idx]); }
        }
        printf("\n");
    }
#endif // print neighbors

    // print nbonds
    printf("~~~~ bonds  %d ~~~~ \n", cpu_nbonds_sum);
}

