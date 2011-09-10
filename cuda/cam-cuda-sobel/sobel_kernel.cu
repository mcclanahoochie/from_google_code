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
static int selectedDevice = 0;    // device to use
int _initialized = 0;              // safety
texture<unsigned char, 2> tex;    // device texture memory
static cudaArray* d_array = NULL;  // storage for texture memory
unsigned char* d_data = NULL;;    // device data output

/////////////////////////////////////
// kernel sobel function core
/////////////////////////////////////
__device__ unsigned char
ComputeSobel(unsigned char ul, // upper left
             unsigned char um, // upper middle
             unsigned char ur, // upper right
             unsigned char ml, // middle left
             unsigned char mm, // middle (unused)
             unsigned char mr, // middle right
             unsigned char ll, // lower left
             unsigned char lm, // lower middle
             unsigned char lr, // lower right
             float fScale) {
    short horz = ur + 2 * mr + lr - ul - 2 * ml - ll;
    short vert = ul + 2 * um + ur - ll - 2 * lm - lr;
    short sum = (short)(fScale * (abs(horz) + abs(vert)));
    if (sum < 0) { return 0; }
    else if (sum > 0xff) { return 0xff; }
    return (unsigned char) sum;
}

/////////////////////////////////////
// kernel sobel function
/////////////////////////////////////
__global__ void
SobelTex(unsigned char* pSobelOriginal, unsigned int pitch,
         int w, int h, float fScale) {
    unsigned char* pSobel = (unsigned char*)(((char*) pSobelOriginal) + blockIdx.x * pitch);
    for (int i = threadIdx.x; i < w; i += blockDim.x) {
        unsigned char pix00 = tex2D(tex, (float) i - 1, (float) blockIdx.x - 1);
        unsigned char pix01 = tex2D(tex, (float) i + 0, (float) blockIdx.x - 1);
        unsigned char pix02 = tex2D(tex, (float) i + 1, (float) blockIdx.x - 1);
        unsigned char pix10 = tex2D(tex, (float) i - 1, (float) blockIdx.x + 0);
        unsigned char pix11 = tex2D(tex, (float) i + 0, (float) blockIdx.x + 0);
        unsigned char pix12 = tex2D(tex, (float) i + 1, (float) blockIdx.x + 0);
        unsigned char pix20 = tex2D(tex, (float) i - 1, (float) blockIdx.x + 1);
        unsigned char pix21 = tex2D(tex, (float) i + 0, (float) blockIdx.x + 1);
        unsigned char pix22 = tex2D(tex, (float) i + 1, (float) blockIdx.x + 1);
        pSobel[i] = ComputeSobel(pix00, pix01, pix02,
                                 pix10, pix11, pix12,
                                 pix20, pix21, pix22,
                                 fScale);
    }
}

/////////////////////////////////////
// error checking routine
/////////////////////////////////////
void checkErrors(char* label) {
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
void setupTexture(int iw, int ih) {
    cudaChannelFormatDesc desc;

    desc = cudaCreateChannelDesc<unsigned char>();

    cudaMallocArray(&d_array, &desc, iw, ih);
}


/////////////////////////////////////
// callable external function
/////////////////////////////////////
extern "C"
{

    void initCUDA(int w, int h) {
        /////////////////////////////////////
        // initialization
        /////////////////////////////////////
        int deviceCount;
        cudaGetDeviceCount(&deviceCount);
        if (deviceCount == 0) {
            fprintf(stderr, "Sorry, no CUDA device found");
        }
        if (selectedDevice >= deviceCount) {
            fprintf(stderr, "Choose device ID between 0 and %d\n", deviceCount - 1);
        }
        cudaSetDevice(selectedDevice);
        checkErrors("initializations");

        /////////////////////////////////////
        // allocate memory
        /////////////////////////////////////
        setupTexture(w, h);
        cudaMalloc((void**)&d_data, sizeof(unsigned char)*w * h);
        checkErrors("memory allocation");

        /////////////////////////////////////
        // safety
        /////////////////////////////////////
        _initialized = 1;
    }

    void stopCUDA(void) {
        /////////////////////////////////////
        // safety
        /////////////////////////////////////
        if (!_initialized) { return; }

        /////////////////////////////////////
        // clean up, free memory
        /////////////////////////////////////
        if (d_data) { cudaFree(d_data); }
        cudaFreeArray(d_array);
    }

    void runCUDASobel(unsigned char* imageData, float thresh, int iw, int ih) {
        /////////////////////////////////////
        // safety
        /////////////////////////////////////
        if (!_initialized) { return; }

        /////////////////////////////////////
        // copy data to device
        /////////////////////////////////////
        cudaMemcpyToArray(d_array, 0, 0, imageData, sizeof(unsigned char)*iw * ih, cudaMemcpyHostToDevice);
        checkErrors("copy data to device");

        /////////////////////////////////////
        // perform computation on device
        /////////////////////////////////////
        cudaBindTextureToArray(tex, d_array);
        SobelTex <<< ih, 384>>>(d_data, iw, iw, ih, thresh);
        checkErrors("compute on device");
        cudaUnbindTexture(tex);

        /////////////////////////////////////
        // read back result from device
        /////////////////////////////////////
        cudaMemcpy(imageData, d_data, sizeof(unsigned char)*iw* ih, cudaMemcpyDeviceToHost);
        checkErrors("copy data from device");
    }

}



