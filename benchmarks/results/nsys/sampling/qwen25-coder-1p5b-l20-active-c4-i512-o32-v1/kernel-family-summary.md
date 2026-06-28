# Nsight Systems Kernel Family Summary

Source: `benchmarks/results/nsys/sampling/qwen25-coder-1p5b-l20-active-c4-i512-o32-v1/stats`

## GPU Kernel Families

| Family | Time share | Total GPU time | Instances | Unique rows |
| --- | ---: | ---: | ---: | ---: |
| `cutlass_or_cublas_gemm` | 79.23% | 551.510 ms | 6011 | 14 |
| `pytorch_fill` | 5.95% | 41.394 ms | 752 | 11 |
| `triton_generated` | 4.63% | 32.255 ms | 5424 | 11 |
| `flashinfer_attention` | 3.52% | 24.509 ms | 2268 | 4 |
| `custom_l20` | 1.98% | 13.793 ms | 264 | 2 |
| `pytorch_elementwise` | 1.40% | 9.738 ms | 2053 | 37 |
| `sampler_other` | 1.22% | 8.482 ms | 2 | 1 |
| `cublas_gemv` | 0.81% | 5.673 ms | 115 | 2 |
| `other` | 0.62% | 4.305 ms | 2047 | 5 |
| `pytorch_softmax` | 0.37% | 2.606 ms | 4 | 1 |
| `flashinfer_sampling` | 0.26% | 1.779 ms | 4 | 2 |

## CUDA API Families

| Family | Time share | Total API time | Calls | Unique rows |
| --- | ---: | ---: | ---: | ---: |
| `sync` | 39.61% | 759.287 ms | 1526 | 3 |
| `memcpy` | 15.16% | 290.562 ms | 7934 | 2 |
| `launch` | 14.03% | 268.949 ms | 30965 | 4 |
| `library_load` | 10.68% | 204.797 ms | 53 | 3 |
| `alloc_free` | 9.74% | 186.627 ms | 361 | 4 |
| `memory_info` | 5.38% | 103.042 ms | 9 | 1 |
| `graph` | 4.08% | 78.175 ms | 237 | 5 |
| `other` | 1.33% | 25.568 ms | 26760 | 20 |

## Interpretation

Use this as a ceiling estimate. A family with a small time share cannot produce a large end-to-end win unless the change also removes launches, synchronization, or adjacent work.
