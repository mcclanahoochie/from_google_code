/*
 * gpu_common.cuh
 *
 *  Created on: Feb 19, 2010
 *      Author: chris
 */

#ifndef __GPU_COMMON_H__
#define __GPU_COMMON_H__


/////////////////////////////////////
// global variables and configuration section
/////////////////////////////////////

// control timing
#define TIME_TRIAL 1  // 1 for time info

// control debug printing
#define DEBUG_PRINT 1  // 1 for more info

// magic number for zero
#define EMPTY 0xAAAAAAAA

// shared vs. global mem usage
#define USE_SH_MEM 0  // 0 for cards <= 8800gtx (no shared atomics)

// number of gpu threads per computation block
#define numThreadsPerBlock 64  // MUST be a power of 2

// gpu xyz texture mem
texture<float, 1, cudaReadModeElementType> xyz_tex;


#endif // #define __GPU_COMMON_H__
