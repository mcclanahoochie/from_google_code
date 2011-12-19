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
 * gpu_bonds_kernel.cu
 *
 *  Created on: Feb 18, 2010
 *      Author: chris
 */

/////////////////////////////////////
// imports
/////////////////////////////////////
#include "gpu_common.cuh"

/////////////////////////////////////
// bonds kernel
/////////////////////////////////////
// 	rmin		= minimum radius to be considered a bond
//	rmax		= maximum radius to be considered a bond
//	radmax		= maximum radius considered for histogram
//	d_nbonds	= device memory storage for bonds count (stored in .x)
//	rdiv		= division size (width) of each bin
//	d_bins		= device memeory storage for bins
//	NBINS		= number of bins
//	validBodies	= actual number of atoms in the current tile (could be less than maxBodies)
//	d_nlist		= device memeory storage for neighbor list
//  maxNeighbors= neighbor list height max
/////////////////////////////////////
__global__ void gpu_compute_bonds_kernel(
    const float rmin, const float rmax, const float radmax,
    void* d_nbonds, const float rdiv, void* d_bins, const int NBINS,
    const int validBodies, int* d_nlist, int maxNeighbors) {
    // identify which atom to handle
    int idx_global = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_global >= validBodies) { return; }

    // setup bonds sum memory
    float* global_nbonds = (float*)d_nbonds;
    int acc = 0;

    // neighbors
    int* nlist = d_nlist;
    int  h = 0;
    int  width = validBodies;

    // read in the position of the current particle.
    int texidx = idx_global * 3; // gpu xyz texture mem
    float3 pos = { tex1Dfetch(xyz_tex, texidx + 0), tex1Dfetch(xyz_tex, texidx + 1), tex1Dfetch(xyz_tex, texidx + 2) };

    // histogram
#if USE_SH_MEM
    extern __shared__ int shbins[];
    for (int i = 0; i < NBINS; ++i) { shbins[i] = 0; }
#endif
    int* gbins = (int*)d_bins;

    // ensure initialization sync before loop
    //__syncthreads();

    // loop over neighbors
    for (int bond_idx = idx_global + 1; bond_idx < validBodies; ++bond_idx) {
        // pos
        texidx = bond_idx * 3; // gpu xyz texture mem
        float3 neigh_pos = { tex1Dfetch(xyz_tex, texidx + 0), tex1Dfetch(xyz_tex, texidx + 1), tex1Dfetch(xyz_tex, texidx + 2) };

        // dist
        float dx = pos.x - neigh_pos.x;
        float dy = pos.y - neigh_pos.y;
        float dz = pos.z - neigh_pos.z;
        float rsq  = dx * dx + dy * dy + dz * dz;
        float dist = sqrtf(rsq);

        // bonds
        if (rmin < dist && dist < rmax) {
            // bond count
            ++acc;

            // neighbors
            if (h < maxNeighbors) {
                nlist[h * width + idx_global] = bond_idx;
                ++h;
            }
        }

        // bins
        if (dist < radmax) {
            int bin = (int) floor((dist - rmin) / rdiv);
            bin = (bin >= NBINS) ? (NBINS) : (bin < 0) ? (NBINS) : bin;
#if USE_SH_MEM
            atomicAdd(shbins + bin, 1);
#else
            gbins[bin] += 1;
#endif
        }
    }

    // write out the result
    global_nbonds[idx_global] = (float)acc;

#if USE_SH_MEM
    // combine histogram results per warp
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int i = 0; i < NBINS; ++i) {
            atomicAdd(gbins + i, shbins[i]);
        }
    }
#endif

}

/////////////////////////////////////
// external bonds kernel manager
/////////////////////////////////////
void gpu_compute_bonds(float* d_xyz,
                       const float rmin, const float rmax, const float radmax,
                       void* d_nbonds, const float rdiv, void* d_bins, const int NBINS,
                       const int validBodies, int* d_nlist, int maxNeighbors) {
    // map xyz data to texture
    cudaBindTexture(0, xyz_tex, d_xyz, validBodies * 3 * sizeof(float));

    // setup sizes
    int p = numThreadsPerBlock;
    int val = (int)ceil(validBodies / p);
    dim3 nthreads(p, 1, 1);
    dim3 nblocks(val, 1, 1);

    // run kernel - compute on gpu
#if USE_SH_MEM
    int sharedmemsize = NBINS * sizeof(int);
    gpu_compute_bonds_kernel <<< nblocks, nthreads, sharedmemsize >>>(
        rmin, rmax, radmax, d_nbonds, rdiv, d_bins, NBINS, validBodies, d_nlist, maxNeighbors);
#else
    gpu_compute_bonds_kernel <<< nblocks, nthreads >>>(
        rmin, rmax, radmax, d_nbonds, rdiv, d_bins, NBINS, validBodies, d_nlist, maxNeighbors);
#endif

    // unmap texture
    cudaUnbindTexture(xyz_tex);
}

