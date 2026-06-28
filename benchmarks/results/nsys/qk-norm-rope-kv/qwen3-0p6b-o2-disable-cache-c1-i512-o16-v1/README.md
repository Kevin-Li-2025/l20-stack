# Qwen3-0.6B O2 i512/o16 Nsight Timeline

This run is the first full decode-shape Nsight Systems proof that the custom
L20 Q/K norm + Q/K RoPE + KV-cache write kernel executes inside vLLM O2.

## Setup

| Field | Value |
| --- | --- |
| GPU | NVIDIA L20 |
| Model | `/home/hhai/models/Qwen3-0.6B` |
| vLLM source | `/home/hhai/vllm-l20-rfc` |
| Execution mode | O2 |
| Compile cache | Disabled |
| Attention backend | FlashInfer |
| Sampling backend | FlashInfer |
| Input/output | 512 / 16 tokens |
| Prompts | 8 |
| Max concurrency | 1 |
| Request rate | `inf` |

## Serving Result

| Metric | Value |
| --- | ---: |
| Output throughput | 202.930 tok/s |
| Total token throughput | 6,696.683 tok/s |
| Mean TTFT | 39.511 ms |
| Median TTFT | 36.068 ms |
| P99 TTFT | 66.420 ms |
| Mean ITL | 2.609 ms |
| Median ITL | 2.746 ms |
| P99 ITL | 3.514 ms |

## Timeline Counts

| Metric | Value |
| --- | ---: |
| CUDA GPU kernel instances | 17,995 |
| Unique CUDA GPU kernel names | 101 |
| CUDA API calls | 67,209 |
| Kernel launch API calls | 28,987 |
| CUDA graph launches | 121 |
| Custom `_l20_qk_norm_rope_kv_kernel` instances | 1,260 |
| Custom `_l20_qk_norm_rope_kv_kernel` average time | 3.998 us |
| Custom `_l20_qk_norm_rope_kv_kernel` total GPU time | 5.038 ms |
| Custom `_l20_qk_norm_rope_kv_kernel` time share | 1.6% |

## Top GPU Kernels

| Rank | Kernel family | Instances | Time share |
| ---: | --- | ---: | ---: |
| 1 | CUTLASS FP16 GEMM 64x64 | 1,988 | 20.3% |
| 2 | cuBLAS GEMV | 240 | 17.8% |
| 3 | PyTorch `FillFunctor<signed char>` vectorized kernel | 28 | 14.0% |
| 4 | FlashInfer `BatchPrefillWithPagedKVCacheKernel`, mask 0 | 980 | 8.0% |
| 5 | Ampere FP16 GEMM 128x64 sliced | 336 | 7.6% |
| 11 | Custom `_l20_qk_norm_rope_kv_kernel` | 1,260 | 1.6% |

## Family Attribution

`kernel-family-summary.{json,md}` groups the raw Nsight Systems CSV rows by
serving boundary:

| Family | GPU time share |
| --- | ---: |
| CUTLASS/cuBLAS GEMM | 44.30% |
| cuBLAS GEMV | 17.80% |
| PyTorch fill/bookkeeping kernels | 14.66% |
| FlashInfer attention | 13.22% |
| Triton-generated model kernels | 2.90% |
| vLLM sampler/logits processor kernels | 2.57% |
| Custom L20 Q/K norm + RoPE + KV write | 1.58% |

The corresponding CUDA API table is dominated by launch, memcpy, library-load,
allocation, and memory-info calls. This run is therefore an integration proof
and a bottleneck map, not a large end-to-end win claim.

## Interpretation

The custom kernel is present in the compiled O2 serving graph, but it accounts
for only 1.6% of GPU kernel time on this shape. This is positive integration
evidence, not proof of a large end-to-end win. The paired off/on serving run in
`benchmarks/results/l20-qk-norm-rope-kv-serving/qwen3-0p6b-o2-disable-cache-c1-i512-o16-r3-v1/`
is the corresponding latency comparison.

The raw `.nsys-rep`, `.sqlite`, and logs remain on the L20 host. The committed
artifact keeps only `run-config.json`, `serving.json`, `timeline-summary.json`,
`kernel-family-summary.{json,md}`, and exported stats CSV files.
