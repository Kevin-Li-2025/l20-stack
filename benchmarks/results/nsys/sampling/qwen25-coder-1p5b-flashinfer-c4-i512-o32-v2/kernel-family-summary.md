# Nsight Systems Kernel Family Summary

Source: `benchmarks/results/nsys/sampling/qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2/stats`

## GPU Kernel Families

| Family | Time share | Total GPU time | Instances | Unique rows |
| --- | ---: | ---: | ---: | ---: |
| `cutlass_or_cublas_gemm` | 42.99% | 550.092 ms | 6010 | 13 |
| `pytorch_fill` | 41.72% | 533.943 ms | 4466 | 10 |
| `triton_generated` | 9.59% | 122.690 ms | 7240 | 12 |
| `flashinfer_attention` | 1.96% | 25.066 ms | 2296 | 4 |
| `pytorch_elementwise` | 0.82% | 10.526 ms | 1683 | 33 |
| `pytorch_softmax` | 0.74% | 9.503 ms | 136 | 1 |
| `flashinfer_sampling` | 0.69% | 8.867 ms | 268 | 2 |
| `sampler_other` | 0.66% | 8.484 ms | 2 | 1 |
| `cublas_gemv` | 0.49% | 6.309 ms | 116 | 2 |
| `other` | 0.33% | 4.233 ms | 2047 | 5 |

## CUDA API Families

| Family | Time share | Total API time | Calls | Unique rows |
| --- | ---: | ---: | ---: | ---: |
| `sync` | 43.76% | 906.897 ms | 1062 | 3 |
| `memcpy` | 13.98% | 289.824 ms | 7368 | 2 |
| `launch` | 13.51% | 279.984 ms | 36285 | 4 |
| `library_load` | 9.17% | 190.012 ms | 58 | 3 |
| `alloc_free` | 8.55% | 177.143 ms | 346 | 3 |
| `memory_info` | 4.91% | 101.764 ms | 9 | 1 |
| `graph` | 3.82% | 79.099 ms | 237 | 5 |
| `other` | 2.31% | 47.826 ms | 50007 | 20 |

## Interpretation

Use this as a ceiling estimate. A family with a small time share cannot produce a large end-to-end win unless the change also removes launches, synchronization, or adjacent work.
