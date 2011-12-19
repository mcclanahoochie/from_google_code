/*
 * main.cpp
 *
 *  Created on: Feb 18, 2010
 *      Author: chris
 */

/////////////////////////////////////
// imports
/////////////////////////////////////
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <math.h>
#include "../natoms/xyz_display_open_gl.h"
#include "gpu_funcs.h" // shared library

/////////////////////////////////////
// global variable declarations
/////////////////////////////////////
int N = (10) * 1024 * 3; // na3
float* xyz0 = NULL;
int* bins = NULL;
int nbins = 512;
int argc0;
char** argv0;

/////////////////////////////////////
// function declarations
/////////////////////////////////////
void TestBondsCUDA();
void TestXyzAutoCorrCUDA();
void TestBinsAutoCorrCUDA();
void CreateRandomXYZData();

/////////////////////////////////////
// main routine
/////////////////////////////////////
int main(int argc, char** argv) {
    argc0 = argc;
    argv0 = argv;

    // command line natoms
    if (argc > 1) {
        int a = atoi(argv[1]);
        printf("\ninput: %d atoms\n\n", a);
        N = a * 3;
    }

    // test loop
    for (int i = 0; i < 5; ++i) {
        // init
        printf("\nRUN %d\n", i);
        CreateRandomXYZData();

        // test bonds
        TestBondsCUDA();

        // test autocorr
        TestXyzAutoCorrCUDA();
        TestBinsAutoCorrCUDA();
    }

    // finish
    if (xyz0) { free(xyz0); }
    if (bins) { free(bins); }
    return 0;
}

/////////////////////////////////////
// data generation
/////////////////////////////////////
void CreateRandomXYZData() {
    // init
    float low = 00.0f;
    float high = 100.0f;
    float temp = 0.0f;

    // allocate
    if (xyz0) {
        free(xyz0);
        xyz0 = NULL;
    }
    xyz0 = (float*)malloc(N * sizeof(float));
    if (xyz0 == NULL) { exit(-1); }

    // generate random float data
    //srand(2010);
    for (int i = 0; i < N - 2; i += 3) {
        temp = sin(i * N / ((double)(N) + 1.0)) * (high - low) + low;
        //temp = sin(rand()/((double)(RAND_MAX)+1.0))*(high-low)+low;
        xyz0[i + 0] = temp;
        temp = cos(i * N / ((double)(N) + 1.0)) * (high - low) + low;
        //temp = cos(rand()/((double)(RAND_MAX)+1.0))*(high-low)+low;
        xyz0[i + 1] = temp;
        temp = tan(i / ((double)(N) + 1.0)) * (high - low) + low;
//		temp = tan(rand()/((double)(RAND_MAX)+1.0))*(high-low)+low;
        xyz0[i + 2] = temp;
    }
}

/////////////////////////////////////
// bonds test
/////////////////////////////////////
void TestBondsCUDA() {
    // configure
    float rmin = 0.00f;	  // minimum distance to be a bond
    float rmax = 4.0f;	  // maximum distance to be a bond
    float maxrad = 7.0f; // max radial distribution distance for histogram; range[0,radmax]
    int* nblist = NULL;  // neighbor list answer
    int nbonds = 0;
    printf("natoms %d \nrmin %.3f \nrmax %.3f \nradmax %.3f \nnbins %d \n", N / 3, rmin, rmax, maxrad, nbins);

    // GO!
    nbonds = compute_bonds_gpu(xyz0, N, rmin, rmax, maxrad, nbins, &nblist, &bins);
    printf(" -> nbonds gpu = %d \n", nbonds);

    // OpenGL display
    if (1) {
        static int once = 1;
        if (once) {
            once = 0;
            showGLbonds(argc0, argv0, xyz0, N / 3, nblist, nbonds);
        } else {
            updateNeighbors(nbonds, nblist, N / 3, xyz0);;
        }

        sleep(2);
    }

    // end
    if (nblist) { free(nblist); }
}

/////////////////////////////////////
// autocorrelation test
/////////////////////////////////////
void TestXyzAutoCorrCUDA() {
    float acx, axy, acz;
    int type = 2;
    compute_xyz_autocorrelation_gpu(xyz0, N, acx, axy, acz, type);
}

/////////////////////////////////////
// autocorrelation test
/////////////////////////////////////
void TestBinsAutoCorrCUDA() {
    float aci;
    int type = 1;
    compute_int_autocorrelation_gpu(bins, nbins, aci, type);
}


