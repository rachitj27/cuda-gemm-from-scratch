# CUDA GEMM From Scratch

Custom CUDA GEMM kernels progressively optimized from naive to warp-tiled, benchmarked against cuBLAS on NVIDIA T4.

## Results (M=N=K=4096, FP32)

| Kernel | GFLOPS | ms | % of cuBLAS |
| --- | --- | --- | --- |
| Naive | 61.9 | 2221.68 | 1.4% |
| cuBLAS | 4279.7 | 32.11 | 100% |

## Kernels

- `01_naive.cu` — one thread per output element, pure global memory access
