# L20 Q/K Norm + RoPE + KV Write Nsight Systems Timeline

This directory tracks serving-level Nsight Systems profiles for the Qwen3
Q/K-norm + RoPE + KV-cache boundary on one NVIDIA L20.

The key result is now a corrected integration finding: the first O2 timeline
captured a stale/non-custom graph and had zero custom kernel instances; the
follow-up O2 timelines disable vLLM compile cache during capture and do execute
the custom L20 three-way kernel.

## Runs

| Run | Model | Mode | Shape | Result |
| --- | --- | --- | --- | --- |
| `qwen3-0p6b-o2-c1-i512-v1/` | Qwen3-0.6B | vLLM O2, FlashInfer | c1, input 512, output 16, 8 prompts | Complete checked-in stats. Custom kernel instances: 0. |
| `qwen3-0p6b-o2-disable-cache-i16-v1/` | Qwen3-0.6B | vLLM O2, FlashInfer, compile cache disabled | c1, input 16, output 1, 1 prompt | Positive O2 integration proof. Custom kernel instances: 1,064. |
| `qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1/` | Qwen3-0.6B | vLLM O2, FlashInfer, compile cache disabled | c1, input 512, output 16, 8 prompts | Full decode-shape O2 proof. Custom kernel instances: 1,260. |

The raw `.nsys-rep` and `.sqlite` files are intentionally not checked in. They
remain on the L20 host at:

- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-c1-i512-v1/vllm-qk-rope-kv.nsys-rep`
- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-c1-i512-v1/vllm-qk-rope-kv.sqlite`
- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-i16-v1/vllm-qk-rope-kv.nsys-rep`
- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-i16-v1/vllm-qk-rope-kv.sqlite`
- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1/vllm-qk-rope-kv.nsys-rep`
- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1/vllm-qk-rope-kv.sqlite`

## Main Counts

From `qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1/timeline-summary.json`:

| Metric | Value |
| --- | ---: |
| CUDA GPU kernel instances | 17,995 |
| Unique CUDA GPU kernel names | 101 |
| CUDA API calls | 67,209 |
| Kernel launch API calls | 28,987 |
| CUDA graph launches | 121 |
| Custom `_l20_qk_norm_rope_kv_kernel` instances | 1,260 |
| Custom `_l20_qk_norm_rope_kv_kernel` avg time | 3.998 us |
| Custom `_l20_qk_norm_rope_kv_kernel` total GPU time | 5.038 ms |
| Custom `_l20_qk_norm_rope_kv_kernel` time share | 1.6% |
| NVTX summary rows | 2 |

The full decode-shape positive run completed 8/8 requests with mean TTFT
39.511 ms, median ITL 2.746 ms, and output throughput 202.930 tok/s.

## Top CUDA Kernels

The O2 serving timeline is dominated by existing vLLM/FlashInfer/PyTorch paths:

| Rank | Kernel family | Instances | Time share |
| ---: | --- | ---: | ---: |
| 1 | CUTLASS FP16 GEMM 64x64 | 1,988 | 20.3% |
| 2 | cuBLAS GEMV | 240 | 17.8% |
| 3 | PyTorch `FillFunctor<signed char>` vectorized kernel | 28 | 14.0% |
| 4 | FlashInfer `BatchPrefillWithPagedKVCacheKernel`, mask 0 | 980 | 8.0% |
| 5 | Ampere FP16 GEMM 128x64 sliced | 336 | 7.6% |
| 11 | Custom `_l20_qk_norm_rope_kv_kernel` | 1,260 | 1.6% |

`qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1/kernel-family-summary.{json,md}`
adds a coarser serving-boundary view:

| Family | GPU time share |
| --- | ---: |
| CUTLASS/cuBLAS GEMM | 44.30% |
| cuBLAS GEMV | 17.80% |
| PyTorch fill/bookkeeping kernels | 14.66% |
| FlashInfer attention | 13.22% |
| vLLM sampler/logits processor kernels | 2.57% |
| Custom L20 Q/K norm + RoPE + KV write | 1.58% |
| FlashInfer sampling | 0.52% |

This is the system-level reason the custom kernel is valuable as integration
proof but cannot by itself produce a decisive serving win on this shape. The
next higher-ceiling boundaries are GEMM/GEMV epilogues, attention/KV layout,
and launch/memcpy reduction.

The zero-hit run remains useful as a failure artifact: Python trace files are
not a reliable O2 gate because compiled graph execution bypasses the Python
trace writer. The Nsight timeline gate is now nonzero
`_l20_qk_norm_rope_kv_kernel` instances.

## NVTX Finding

The run captures `--trace=cuda,nvtx,osrt`, but `nvtx_sum` only contains two CUB
DeviceScan ranges. A second remote run passed
`--enable-layerwise-nvtx-tracing` and the server log showed
`enable_layerwise_nvtx_tracing=True`; `nvtx_sum` still only reported the same
CUB ranges. Layerwise vLLM ranges therefore need direct sqlite inspection or a
different capture/export path before they can be used as a profiling artifact.
