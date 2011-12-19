// compile:   nvcc -O3  foo.cu -lcublas -I/usr/local/cuda/include -L/usr/local/cuda/lib64 */
// run: ./test_sgemm 128 256   # run for that range

#include <stdio.h>
#include <cuda.h>
#include <cublas.h>
#include <err.h>

#define REPS  7

int main(int argc, char* args[]) {
    if (argc != 3) { errx(-1, "%s <small> <large>", args[0]); }

    int small = atoi(args[1]);
    int large = atoi(args[2]);


    int runtimeVersion;
    cudaRuntimeGetVersion(&runtimeVersion);
    printf("toolkit version %d\n", runtimeVersion);

    float* d_A, *d_B, *d_C;
    int bytes = large * large * sizeof(float);
    cudaMalloc((void**)&d_A, bytes);
    cudaMalloc((void**)&d_B, bytes);
    cudaMalloc((void**)&d_C, bytes);

    /* prepare timers */
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cublasSgemm('N', 'N', small, small, small, 1, d_A, small, d_B, small, 0, d_C, small);
    cudaThreadSynchronize();

    for (int n = small; n <= large; ++n) {
        cudaEventRecord(start, 0);

        for (int reps = 0; reps < REPS; ++reps)
        { cublasSgemm('N', 'N', n, n, n, 1, d_A, n, d_B, n, 0, d_C, n); }

        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        float time;
        cudaEventElapsedTime(&time, start, stop);
        double gflops = pow(n, 3) * 2.0 * REPS * 1e-6 / time;
        printf("%4d  %g\n", n, gflops);
    }

    return 0;
}
