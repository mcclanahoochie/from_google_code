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

/////////////////////////////////////
//
// Chris McClanahan
//  3/24/2010
//   Modified CUDA SobelFilter example
//
/////////////////////////////////////

/////////////////////////////////////
// standard imports
/////////////////////////////////////
#include <stdio.h>
#include <math.h>

/////////////////////////////////////
// global variables and configuration section
/////////////////////////////////////
static int selectedDevice4 = 0;    // device to use
int _initialized4 = 0;              // safety
texture<uchar4, 2> tex4;    // device texture memory
static cudaArray* d_array4 = NULL;  // storage for texture memory
uchar4* d_data4 = NULL;;    // device data output

/////////////////////////////////////
// kernel sobel function core
/////////////////////////////////////
__device__ uchar4
ComputeSobel(uchar4 ul, // upper left
             uchar4 um, // upper middle
             uchar4 ur, // upper right
             uchar4 ml, // middle left
             uchar4 mm, // middle (unused)
             uchar4 mr, // middle right
             uchar4 ll, // lower left
             uchar4 lm, // lower middle
             uchar4 lr, // lower right
             float fScale) {
    short horz, vert;
    // b
    horz = ur.x + 2 * mr.x + lr.x - ul.x - 2 * ml.x - ll.x;
    vert = ul.x + 2 * um.x + ur.x - ll.x - 2 * lm.x - lr.x;
    short sumb = (short)(fScale * (abs(horz) + abs(vert)));
    if (sumb < 0) { sumb = 0; }
    else if (sumb > 0xff) { sumb = 0xff; }
    // g
    horz = ur.y + 2 * mr.y + lr.y - ul.y - 2 * ml.y - ll.y;
    vert = ul.y + 2 * um.y + ur.y - ll.y - 2 * lm.y - lr.y;
    short sumg = (short)(fScale * (abs(horz) + abs(vert)));
    if (sumg < 0) { sumg = 0; }
    else if (sumg > 0xff) { sumg = 0xff; }
    // r
    horz = ur.z + 2 * mr.z + lr.z - ul.z - 2 * ml.z - ll.z;
    vert = ul.z + 2 * um.z + ur.z - ll.z - 2 * lm.z - lr.z;
    short sumr = (short)(fScale * (abs(horz) + abs(vert)));
    if (sumr < 0) { sumr = 0; }
    else if (sumr > 0xff) { sumr = 0xff; }

    return (uchar4) {
        sumb, sumg, sumr, 1
    };
}

/////////////////////////////////////
// kernel sobel function
/////////////////////////////////////
__global__ void
SobelTex(uchar4* pSobelOriginal, unsigned int pitch,
         int w, int h, float fScale) {
    uchar4* pSobel = (uchar4*)(((char4*) pSobelOriginal) + blockIdx.x * pitch);
    for (int i = threadIdx.x; i < w; i += blockDim.x) {
        uchar4 pix00 = tex2D(tex4, (float) i - 1, (float) blockIdx.x - 1);
        uchar4 pix01 = tex2D(tex4, (float) i + 0, (float) blockIdx.x - 1);
        uchar4 pix02 = tex2D(tex4, (float) i + 1, (float) blockIdx.x - 1);
        uchar4 pix10 = tex2D(tex4, (float) i - 1, (float) blockIdx.x + 0);
        uchar4 pix11 = tex2D(tex4, (float) i + 0, (float) blockIdx.x + 0);
        uchar4 pix12 = tex2D(tex4, (float) i + 1, (float) blockIdx.x + 0);
        uchar4 pix20 = tex2D(tex4, (float) i - 1, (float) blockIdx.x + 1);
        uchar4 pix21 = tex2D(tex4, (float) i + 0, (float) blockIdx.x + 1);
        uchar4 pix22 = tex2D(tex4, (float) i + 1, (float) blockIdx.x + 1);
        pSobel[i] = ComputeSobel(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22,
                                 fScale);
    }
}

/////////////////////////////////////
// error checking routine
/////////////////////////////////////
void checkErrors4(char* label) {
    cudaError_t err;

    err = cudaThreadSynchronize();
    if (err != cudaSuccess) {
        char* e = (char*) cudaGetErrorString(err);
        fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
    }

    err = cudaGetLastError();
    if (err != cudaSuccess) {
        char* e = (char*) cudaGetErrorString(err);
        fprintf(stderr, "CUDA Error: %s (at %s)", e, label);
    }
}


/////////////////////////////////////
// configure and malloc device texture memory
/////////////////////////////////////
void setupTexture4(int iw, int ih) {
    cudaChannelFormatDesc desc;

    desc = cudaCreateChannelDesc<uchar4>();

    cudaMallocArray(&d_array4, &desc, iw, ih);
}


/////////////////////////////////////
// callable external function
/////////////////////////////////////
extern "C"
{

    void initCUDA4(int w, int h) {
        /////////////////////////////////////
        // initialization
        /////////////////////////////////////
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            fprintf(stderr, "Sorry, no CUDA device found");
        }
        if (selectedDevice4 >= deviceCount) {
            fprintf(stderr, "Choose device ID between 0 and %d\n", deviceCount - 1);
        }
        cudaSetDevice(selectedDevice4);
        checkErrors4("initializations");

        /////////////////////////////////////
        // allocate memory
        /////////////////////////////////////
        setupTexture4(w, h);
        cudaMalloc((void**)&d_data4, sizeof(uchar4)*w * h);
        checkErrors4("memory allocation");

        /////////////////////////////////////
        // safety
        /////////////////////////////////////
        _initialized4 = 1;
    }

    void stopCUDA4(void) {
        /////////////////////////////////////
        // safety
        /////////////////////////////////////
        if (!_initialized4) { return; }

        /////////////////////////////////////
        // clean up, free memory
        /////////////////////////////////////
        if (d_data4) { cudaFree(d_data4); }
        cudaFreeArray(d_array4);
    }

    void runCUDASobel4(uchar4* imageData, float thresh, int iw, int ih) {
        /////////////////////////////////////
        // safety
        /////////////////////////////////////
        if (!_initialized4) { return; }

        /////////////////////////////////////
        // copy data to device
        /////////////////////////////////////
        cudaMemcpyToArray(d_array4, 0, 0, imageData, sizeof(uchar4)*iw * ih, cudaMemcpyHostToDevice);
        checkErrors4("copy data to device");

        /////////////////////////////////////
        // perform computation on device
        /////////////////////////////////////
        cudaBindTextureToArray(tex4, d_array4);
        SobelTex <<< ih, 256>>>(d_data4, iw, iw, ih, thresh);
        checkErrors4("compute on device");
        cudaUnbindTexture(tex4);

        /////////////////////////////////////
        // read back result from device
        /////////////////////////////////////
        cudaMemcpy(imageData, d_data4, sizeof(uchar4)*iw* ih, cudaMemcpyDeviceToHost);
        checkErrors4("copy data from device");
    }

}



