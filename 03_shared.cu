#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define BLOCKSIZE 32

__global__ void sgemm_shared(int M, int N, int K, float alpha, const float *A,
                             const float *B, float beta, float *C) {
    // Which output tile this block is responsible for
    const uint cRow = blockIdx.x;
    const uint cCol = blockIdx.y;

    // Shared memory for the A and B tiles
    __shared__ float As[BLOCKSIZE * BLOCKSIZE];
    __shared__ float Bs[BLOCKSIZE * BLOCKSIZE];

    // This thread's position within the tile (0..31 in each dimension)
    const uint threadCol = threadIdx.x % BLOCKSIZE;
    const uint threadRow = threadIdx.x / BLOCKSIZE;

    // Advance A, B, C pointers to this block's starting position
    A += cRow * BLOCKSIZE * K;         // move to block's row of A
    B += cCol * BLOCKSIZE;              // move to block's column of B
    C += cRow * BLOCKSIZE * N + cCol * BLOCKSIZE;  // move to block's tile of C

    float tmp = 0.0;

    // Sweep through K in BLOCKSIZE-wide chunks
    for (int bkIdx = 0; bkIdx < K; bkIdx += BLOCKSIZE) {
        // Each thread loads one element of A and one element of B into shared memory
        As[threadRow * BLOCKSIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCKSIZE + threadCol] = B[threadRow * N + threadCol];

        // Wait until every thread in the block finishes loading
        __syncthreads();

        // Advance A and B pointers to the next tile along K
        A += BLOCKSIZE;
        B += BLOCKSIZE * N;

        // Do the partial dot product using shared memory
        for (int dotIdx = 0; dotIdx < BLOCKSIZE; ++dotIdx) {
            tmp += As[threadRow * BLOCKSIZE + dotIdx] * Bs[dotIdx * BLOCKSIZE + threadCol];
        }

        // Wait until every thread finishes using the tiles before we overwrite them
        __syncthreads();
    }

    // Write result
    C[threadRow * N + threadCol] = alpha * tmp + beta * C[threadRow * N + threadCol];
}

int main() {
    const int M = 4096;
    const int N = 4096;
    const int K = 4096;
    const float alpha = 1.0f;
    const float beta = 0.0f;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    float *h_A = (float*)malloc(bytes_A);
    float *h_B = (float*)malloc(bytes_B);
    float *h_C_kernel = (float*)malloc(bytes_C);
    float *h_C_cublas = (float*)malloc(bytes_C);

    srand(42);
    for (int i = 0; i < M * K; i++) h_A[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = (float)rand() / RAND_MAX * 2.0f - 1.0f;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C, bytes_C);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    dim3 gridDim(CEIL_DIV(M, BLOCKSIZE), CEIL_DIV(N, BLOCKSIZE), 1);
    dim3 blockDim(BLOCKSIZE * BLOCKSIZE);

    // Warmup
    cudaMemset(d_C, 0, bytes_C);
    sgemm_shared<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();

    const int num_runs = 5;
    float total_ms_kernel = 0.0f;
    for (int run = 0; run < num_runs; run++) {
        cudaMemset(d_C, 0, bytes_C);
        cudaEventRecord(start);
        sgemm_shared<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms_kernel += ms;
    }
    float avg_ms_kernel = total_ms_kernel / num_runs;

    cudaMemcpy(h_C_kernel, d_C, bytes_C, cudaMemcpyDeviceToHost);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaMemset(d_C, 0, bytes_C);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
    cudaDeviceSynchronize();

    float total_ms_cublas = 0.0f;
    for (int run = 0; run < num_runs; run++) {
        cudaMemset(d_C, 0, bytes_C);
        cudaEventRecord(start);
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        total_ms_cublas += ms;
    }
    float avg_ms_cublas = total_ms_cublas / num_runs;

    cudaMemcpy(h_C_cublas, d_C, bytes_C, cudaMemcpyDeviceToHost);

    float max_diff = 0.0f;
    int num_mismatches = 0;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(h_C_kernel[i] - h_C_cublas[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > 1e-2f) num_mismatches++;
    }

    double flops = 2.0 * M * N * K;
    double gflops_kernel = (flops / 1e9) / (avg_ms_kernel / 1000.0);
    double gflops_cublas = (flops / 1e9) / (avg_ms_cublas / 1000.0);

    printf("Matrix size: %d x %d x %d\n", M, N, K);
    printf("Shared kernel: %.2f ms, %.1f GFLOPS\n", avg_ms_kernel, gflops_kernel);
    printf("cuBLAS:        %.2f ms, %.1f GFLOPS\n", avg_ms_cublas, gflops_cublas);
    printf("Shared is %.1f%% of cuBLAS\n", 100.0 * gflops_kernel / gflops_cublas);
    printf("Max diff between shared and cuBLAS: %e\n", max_diff);
    printf("Mismatches (> 1e-2): %d / %d\n", num_mismatches, M * N);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_kernel); free(h_C_cublas);
    return 0;
}
