# Nsight Systems Kernel Family Summary

Source: `benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1/stats`

## GPU Kernel Families

| Family | Time share | Total GPU time | Instances | Unique rows |
| --- | ---: | ---: | ---: | ---: |
| `cutlass_or_cublas_gemm` | 44.30% | 140.983 ms | 6080 | 15 |
| `cublas_gemv` | 17.80% | 56.656 ms | 240 | 1 |
| `pytorch_fill` | 14.66% | 46.668 ms | 1360 | 10 |
| `flashinfer_attention` | 13.22% | 42.086 ms | 2436 | 4 |
| `triton_generated` | 2.90% | 9.245 ms | 3910 | 10 |
| `sampler_other` | 2.57% | 8.182 ms | 177 | 13 |
| `custom_l20` | 1.58% | 5.038 ms | 1260 | 1 |
| `pytorch_elementwise` | 1.06% | 3.373 ms | 1275 | 25 |
| `other` | 1.05% | 3.328 ms | 1251 | 19 |
| `flashinfer_sampling` | 0.52% | 1.642 ms | 4 | 2 |
| `pytorch_softmax` | 0.33% | 1.047 ms | 2 | 1 |

## CUDA API Families

| Family | Time share | Total API time | Calls | Unique rows |
| --- | ---: | ---: | ---: | ---: |
| `launch` | 21.79% | 208.366 ms | 28866 | 4 |
| `memcpy` | 20.19% | 193.029 ms | 3842 | 2 |
| `library_load` | 17.68% | 169.096 ms | 68 | 3 |
| `alloc_free` | 16.22% | 155.126 ms | 1094 | 3 |
| `memory_info` | 11.14% | 106.509 ms | 5 | 1 |
| `graph` | 7.11% | 67.991 ms | 226 | 4 |
| `other` | 3.00% | 28.675 ms | 32207 | 19 |
| `sync` | 2.87% | 27.465 ms | 901 | 3 |

## Interpretation

Use this as a ceiling estimate. A family with a small time share cannot produce a large end-to-end win unless the change also removes launches, synchronization, or adjacent work.
