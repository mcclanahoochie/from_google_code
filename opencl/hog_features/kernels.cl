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


#define NBINS    8  // CHECK in main.cpp also
#define CELLDIM  16
#ifndef M_PI
#define M_PI 3.14159265
#endif

//__kernel
/* void histograms(__global float* ang_i, __global float* mag_i, int rows, int cols, __global float* bins) { */
void histograms(__global float* ang_i, __global float* mag_i, int rows, int cols, __global float* bins,
                int row, int col, int i, int j, int cols_i, __local int* sbins, __local int* cumsum) {

    // int row = get_global_id(0);
    // int col = get_global_id(1);
    // if (row >= rows || col >= cols) { return; }

    // int i = get_local_id(0);
    // int j = get_local_id(1);
    // if (i >= CELLDIM || j > 0) { return; }

    // __local int sbins[NBINS];
    // __local int cumsum = 0;

    for (int hh = 0; hh < NBINS; ++hh) { sbins[hh] = 0; } *cumsum = 0;
    mem_fence(CLK_LOCAL_MEM_FENCE); // barrier?

    for (int jj = 0; jj < CELLDIM; ++jj) {
        if ((col + jj) >= cols) { continue; }
        int i_id = (row) * cols_i + (col + jj);
        int h_id = (ang_i[i_id] * (180.f / (float)M_PI)) / (360.f / (float)NBINS);
        int weight = mag_i[i_id] * 10.f;
        h_id = h_id >= NBINS ? NBINS - 1 : h_id;
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
        atom_add(&sbins[h_id], weight);
    }
    mem_fence(CLK_LOCAL_MEM_FENCE); // barrier?

    if (row < NBINS) {
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
        atom_add(cumsum, sbins[row]); // for l2 norm
    }

    mem_fence(CLK_GLOBAL_MEM_FENCE); // barrier?
    if (i == 0) {
#if 1
        // save histo
        int cell_id = (col / CELLDIM) + (row / CELLDIM) * (cols / CELLDIM);
        float denom = sqrt((*cumsum) * (*cumsum) + 0.01f);
        for (int hh = 0; hh < NBINS; ++hh) {
            float normh = (float)sbins[hh] / denom;  // l2 norm
            bins[cell_id * NBINS + hh] = normh;
        }
#endif
#if 0
        // debug
        for (int ii = 0; ii < CELLDIM; ++ii) {
            for (int hh = 0; hh < CELLDIM; ++hh) {
                int xx = (col + hh);
                int yy = (row + ii);
                if (yy >= rows || xx >= cols_i) { continue; }
                int i_id = yy * cols_i + xx;
                ang_i[i_id] = (hh + ii) / (float)CELLDIM / 2.f;
            }
        }
#endif
    }

}


__kernel
void window_hist(__global float* ang_i, __global float* mag_i, int rows_i, int cols_i, __global float* bins,
                 int r_start, int c_start, int w_rows, int w_cols) {


    __global float* ang_i_sub = ang_i + (r_start * cols_i + c_start);
    __global float* mag_i_sub = mag_i + (r_start * cols_i + c_start);


    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= w_rows || col >= w_cols) { return; }

    int i = get_local_id(0);
    int j = get_local_id(1);
    if (i >= CELLDIM || j > 0) { return; }

    if (r_start + row >= rows_i || c_start + col >= cols_i) { return; }

    __local int sbins[NBINS];
    __local int cumsum;

    histograms(ang_i_sub, mag_i_sub, w_rows, w_cols, bins, row, col, i, j, cols_i, sbins, &cumsum);

}



__kernel
void classify(__global float* feats, __global float* w, __global int* res, int ncells) {

    int row = get_global_id(0);
    int col = get_global_id(1);
    if (row >= ncells || col >= NBINS) { return; }

    float vdot[NBINS];
    int idx = row * NBINS + col;

    vdot[col] = (feats[idx] * w[idx]) ;
    mem_fence(CLK_LOCAL_MEM_FENCE); // barrier?

    int j = get_local_id(1);
    __local int sdot = 0;

#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
    atom_add(&sdot, vdot[col]);

    mem_fence(CLK_GLOBAL_MEM_FENCE); // barrier?
    if (col == 0) {
#pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics: enable
        atom_add(res, sdot);
    }


}


