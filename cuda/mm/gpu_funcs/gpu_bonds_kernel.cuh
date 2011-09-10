/*
 * gpu_bonds_kernel.cuh
 *
 *  Created on: Feb 18, 2010
 *      Author: chris
 */

#ifndef __GPU_BONDS_KERNEL_H__
#define __GPU_BONDS_KERNEL_H__

void gpu_compute_bonds( float* d_xyz,
		const float rmin, const float rmax, const float radmax,
		void *d_nbonds, const float rdiv, void *d_bins, const int NBINS,
		const int validBodies, int *d_nlist, int maxNeighbors);

#endif // #define __GPU_BONDS_KERNEL_H__
