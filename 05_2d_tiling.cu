#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

#define CEIL_DIV(a, b) (((a) + (b) - 1) / (b))
#define BM 128
#define BN 128
#define BK 8
#define TM 8
#define TN 8

__global__ void sgemm_2d_tiling(int M, int N, int K, float alpha, const float *A,
                                const float *B, float beta, float *C) {
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / (TM * TN);

    const uint threadCol = threadIdx.x % (BN / TN);
    const uint threadRow = threadIdx.x / (BN / TN);

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / BK;
    const uint innerColA = threadIdx.x % BK;
    const uint strideA = numThreadsBlocktile / BK;
    const uint innerRowB = threadIdx.x / BN;
    const uint innerColB = threadIdx.x % BN;
    const uint strideB = numThreadsBlocktile / BN;

    float threadResults[TM * TN] = {0.0};
    float regM[TM] = {0.0};
    float regN[TN] = {0.0};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint loadOffset = 0; loadOffset < BM; loadOffset += strideA) {
            As[(innerRowA + loadOffset) * BK + innerColA] =
                A[(innerRowA + loadOffset) * K + innerColA];
        }
        for (uint loadOffset = 0; loadOffset < BK; loadOffset += strideB) {
            Bs[(innerRowB + loadOffset) * BN + innerColB] =
                B[(innerRowB + loadOffset) * N + innerColB];
        }

        __syncthreads();

        A += BK;
        B += BK * N;

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; ++i) {
                regN[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                    threadResults[resIdxM * TN + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }

        __syncthreads();
    }

    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
            C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN] =
                alpha * threadResults[resIdxM * TN + resIdxN] +
                beta * C[(threadRow * TM + resIdxM) * N + threadCol * TN + resIdxN];
        }
    }
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

    dim3 gridDim(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 blockDim((BM * BN) / (TM * TN));

    cudaMemset(d_C, 0, bytes_C);
    sgemm_2d_tiling<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
    cudaDeviceSynchronize();

    const int num_runs = 5;
    float total_ms_kernel = 0.0f;
    for (int run = 0; run < num_runs; run++) {
        cudaMemset(d_C, 0, bytes_C);
        cudaEventRecord(start);
        sgemm_2d_tiling<<<gridDim, blockDim>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
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
    printf("2D tiling kernel: %.2f ms, %.1f GFLOPS\n", avg_ms_kernel, gflops_kernel);
    printf("cuBLAS:           %.2f ms, %.1f GFLOPS\n", avg_ms_cublas, gflops_cublas);
    printf("2D tiling is %.1f%% of cuBLAS\n", 100.0 * gflops_kernel / gflops_cublas);
    printf("Max diff: %e\n", max_diff);
    printf("Mismatches (> 1e-2): %d / %d\n", num_mismatches, M * N);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C_kernel); free(h_C_cublas);
    return 0;
}
