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


#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <cuda.h>
#include <npp.h>


#define CUDA(call) do {                             \
    cudaError_t _e = (call);                        \
    if ((_e = cudaGetLastError()) != cudaSuccess) { \
        return printf( "CUDA runtime error: %s\n",  \
                       cudaGetErrorString(_e));     \
    } cudaThreadSynchronize();                      \
} while (0)
#define NPP(call) do {                              \
    NppStatus _e = (call);                          \
    if( NPP_SUCCESS > _e) {                         \
        return printf( "NPP runtime error: ");      \
    } cudaThreadSynchronize();                      \
} while (0)


int main(int argc, char** args) {

    // args
    int P = 3;
    if (argc < P - 1) { return printf("Usage: %s  <numel>\n", args[0]); }

    // sizes
    int numel = atoi(args[P - 2]);
    const int bins = 256;
    float max = 255;
    float min = 0;

    // input data
    float* h_data = (float*)malloc(numel * sizeof(float));
    for (int i = 0; i < numel ; ++i) { h_data[i] = (i % (max + 1)); }

    // gpu mem
    Npp32f* d_data = nppsMalloc_32f(numel);
    Npp32s* pHist = nppsMalloc_32s(bins);
    Npp32f* pLevels = nppsMalloc_32f(bins + 1);
    CUDA(cudaMemcpy(d_data, h_data, numel * sizeof(float), cudaMemcpyHostToDevice));

    // input data range
    float h_minmax[] = {min, max};
    float* d_minmax = nppsMalloc_32f(2);
    CUDA(cudaMemcpy(d_minmax, &h_minmax, 2 * sizeof(float), cudaMemcpyHostToDevice));

    // bin spacing
    float levels[bins + 1];
    float scalar = (max - min) / (float)(bins);
    for (int i = 0; i < bins + 1 ; ++i) { levels[i] = (i * scalar); }
    levels[bins] = FLT_MAX; // last bin is catch all
    CUDA(cudaMemcpy(pLevels, levels, (bins + 1) * sizeof(float), cudaMemcpyHostToDevice));

    // nppihist config
    int nLevels = bins + 1; // nppihist returns bins-1
    int nSrcStep = numel * sizeof(float); // bytes

    // nppihist scratch buffer
    NppiSize oBuffROI;
    oBuffROI.width = numel;
    oBuffROI.height = 1;
    int buffsize;
    NPP(nppiHistogramRangeGetBufferSize_32f_C1R(oBuffROI, nLevels, &buffsize));
    Npp8u* pBuffer = nppsMalloc_8u(buffsize);

    // nppihist config
    NppiSize oSizeROI;
    oSizeROI.width = numel;
    oSizeROI.height = 1;

    // run gpu histogram
    NPP(nppiHistogramRange_32f_C1R(d_data,
                                   nSrcStep, oSizeROI,
                                   pHist ,
                                   pLevels, nLevels,
                                   pBuffer
                                  ));

    // copy back
    int* h_hist = (int*)malloc(bins * sizeof(int));
    CUDA(cudaMemcpy(h_hist, pHist, bins * sizeof(int), cudaMemcpyDeviceToHost));

    // cpu reference
    int* h_ref = (int*)malloc(bins * sizeof(int));
    for (int i = 0; i < bins ; ++i) { h_ref[i] = 0; }
    for (int i = 0; i < numel ; ++i) {
        int idx = (h_data[i] - min) / ((max - min) / (float)bins);
        idx -= (idx >= bins);
        h_ref[idx]++;
    }

    // compare/print
    for (int i = 0; i < bins ; ++i) {
        printf("%d g %d  c %d\n", i, h_hist[i], h_ref[i]);
    }

    // cleanup
    free(h_ref);
    free(h_data);
    free(h_hist);
    nppsFree(pBuffer);
    nppsFree(pHist);
    nppsFree(pLevels);
    nppsFree(d_data);
    nppsFree(d_minmax);

    return 0;
}
