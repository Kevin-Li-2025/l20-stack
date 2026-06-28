# L20 Q/K Norm + RoPE + KV Write Nsight Systems Timeline

This directory tracks serving-level Nsight Systems profiles for the Qwen3
Q/K-norm + RoPE + KV-cache boundary on one NVIDIA L20.

The key result is now a corrected integration finding: the first O2 timeline
captured a stale/non-custom graph and had zero custom kernel instances; the
follow-up O2 timeline disables vLLM compile cache during capture and does
execute the custom L20 three-way kernel.

## Runs

| Run | Model | Mode | Shape | Result |
| --- | --- | --- | --- | --- |
| `qwen3-0p6b-o2-c1-i512-v1/` | Qwen3-0.6B | vLLM O2, FlashInfer | c1, input 512, output 16, 8 prompts | Complete checked-in stats. Custom kernel instances: 0. |
| `qwen3-0p6b-o2-disable-cache-i16-v1/` | Qwen3-0.6B | vLLM O2, FlashInfer, compile cache disabled | c1, input 16, output 1, 1 prompt | Positive O2 integration proof. Custom kernel instances: 1,064. |

The raw `.nsys-rep` and `.sqlite` files are intentionally not checked in. They
remain on the L20 host at:

- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-c1-i512-v1/vllm-qk-rope-kv.nsys-rep`
- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-c1-i512-v1/vllm-qk-rope-kv.sqlite`
- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-i16-v1/vllm-qk-rope-kv.nsys-rep`
- `/home/hhai/l20-stack/benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-i16-v1/vllm-qk-rope-kv.sqlite`

## Main Counts

From `qwen3-0p6b-o2-disable-cache-i16-v1/timeline-summary.json`:

| Metric | Value |
| --- | ---: |
| CUDA GPU kernel instances | 14,961 |
| Unique CUDA GPU kernel names | 103 |
| CUDA API calls | 61,252 |
| Kernel launch API calls | 25,833 |
| CUDA graph launches | 1 |
| Custom `_l20_qk_norm_rope_kv_kernel` instances | 1,064 |
| Custom `_l20_qk_norm_rope_kv_kernel` avg time | 3.248 us |
| Custom `_l20_qk_norm_rope_kv_kernel` total GPU time | 3.455 ms |
| NVTX summary rows | 2 |

The positive run is intentionally a one-output-token smoke profile, so ITL is
not meaningful. It completed 1/1 request with mean TTFT 40.631 ms and output
throughput 24.187 tok/s.

## Top CUDA Kernels

The O2 serving timeline is dominated by existing vLLM/FlashInfer/PyTorch paths:

| Rank | Kernel family | Instances | Time share |
| ---: | --- | ---: | ---: |
| 1 | PyTorch `FillFunctor<int>` vectorized kernel | 2,494 | 39.7% |
| 2 | CUTLASS FP16 GEMM 64x64 | 1,988 | 10.8% |
| 3 | cuBLAS GEMV | 240 | 9.6% |
| 4 | PyTorch `FillFunctor<signed char>` vectorized kernel | 28 | 7.5% |
| 5 | Triton `triton_red_fused_1` | 2,242 | 6.8% |
| 6 | FlashInfer `BatchPrefillWithPagedKVCacheKernel`, mask 0 | 980 | 4.3% |
| 7 | Ampere FP16 GEMM 128x64 sliced | 336 | 4.1% |

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
