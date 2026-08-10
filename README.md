# CUDA GEMM From Scratch

Custom CUDA GEMM kernels progressively optimized from naive to warp-tiled, benchmarked against cuBLAS on NVIDIA T4.

## Results (M=N=K=4096, FP32)

| Kernel | GFLOPS | ms | % of cuBLAS |
| --- | --- | --- | --- |
| Naive | 61.9 | 2221.68 | 1.4% |
| Coalesced | 541.4 | 253.88 | 13.4% |
| cuBLAS | 4032.6 | 34.08 | 100% |

## Kernels

- `01_naive.cu` — one thread per output element, pure global memory access
- `02_coalesced.cu` — 1D block layout so threadIdx.x maps to column, enabling coalesced reads of B
