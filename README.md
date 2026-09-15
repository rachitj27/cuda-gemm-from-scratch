# CUDA GEMM From Scratch

Custom CUDA GEMM kernels progressively optimized from naive to 2D register tiling, benchmarked against cuBLAS on NVIDIA T4.

## Results (M=N=K=4096, FP32)

| Kernel | GFLOPS | ms | % of cuBLAS |
| --- | --- | --- | --- |
| Naive | 61.9 | 2221.68 | 1.4% |
| Coalesced | 541.4 | 253.88 | 13.4% |
| Shared | 842.1 | 163.21 | 22.5% |
| 1D tiling | 1838.5 | 74.76 | 45.3% |
| 2D tiling | 3220.3 | 42.68 | 76.3% |
| cuBLAS | 4222.2 | 32.55 | 100% |

All kernels verified bit-identical to cuBLAS across all 16.7M output elements.

## Kernels

- `01_naive.cu` — one thread per output element, pure global memory access
- `02_coalesced.cu` — 1D block layout so threadIdx.x maps to column, enabling coalesced reads of B
- `03_shared.cu` — cooperative shared memory tiling, block loads 32×32 tiles of A and B once and reuses each value 32× across the block
- `04_1d_tiling.cu` — each thread computes an 8×1 strip of C, hoisting the shared B load out of the inner loop so one B value is reused across 8 multiply-adds
- `05_2d_tiling.cu` — each thread computes an 8×8 block of C via outer product in registers, 64 FMAs per 16 shared memory loads
