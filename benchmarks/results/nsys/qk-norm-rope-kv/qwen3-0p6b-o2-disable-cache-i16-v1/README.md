# Qwen3-0.6B O2 L20 Custom Kernel Timeline

Real vLLM serving profile on one NVIDIA L20. This run is a positive
integration proof for the experimental L20 Q/K norm + Q/K RoPE + KV-cache
write kernel in O2 mode.

## Command Shape

- Model: `/home/hhai/models/Qwen3-0.6B`
- vLLM source: `/home/hhai/vllm-l20-rfc`
- Mode: O2 / CUDA graph, FlashInfer attention
- Request shape: input 16, output 1, 1 prompt, max concurrency 1,
  `REQUEST_RATE=inf`
- Native `enable_qk_norm_rope_fusion`: off
- Native `fuse_rope_kvcache`: off
- `VLLM_L20_QK_ROPE_KV`: on
- `VLLM_DISABLE_COMPILE_CACHE`: 1

`VLLM_DISABLE_COMPILE_CACHE=1` is intentional for this artifact: the goal is to
force a fresh compiled graph so stale AOT cache entries cannot hide the
experimental path.

## Serving Result

This is a one-output-token smoke shape, so ITL is not meaningful.

| Metric | Value |
| --- | ---: |
| Completed requests | 1 |
| Failed requests | 0 |
| Output throughput | 24.187 tok/s |
| Total token throughput | 411.176 tok/s |
| Mean TTFT | 40.631 ms |
| Median TTFT | 40.631 ms |
| P99 TTFT | 40.631 ms |

## Timeline Result

| Metric | Value |
| --- | ---: |
| CUDA GPU kernel instances | 14,961 |
| Unique CUDA GPU kernel names | 103 |
| Kernel launch API calls | 25,833 |
| CUDA graph launches | 1 |
| Custom `_l20_qk_norm_rope_kv_kernel` instances | 1,064 |
| Custom `_l20_qk_norm_rope_kv_kernel` avg time | 3.248 us |
| Custom `_l20_qk_norm_rope_kv_kernel` total GPU time | 3.455 ms |
| Custom `_l20_qk_norm_rope_kv_kernel` GPU time share | 1.1% |

This artifact supersedes the earlier zero-hit O2 timeline as an integration
check. In O2 mode, the Python trace file can remain empty because the compiled
graph bypasses the Python trace writer; Nsight kernel rows are the correct gate.

## Checked-In Artifacts

- `run-config.json`
- `serving.json`
- `timeline-summary.json`
- `stats/cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv`
- `stats/cuda_kern_exec_sum_cuda_kern_exec_sum.csv`
- `stats/cuda_api_sum_cuda_api_sum.csv`
- `stats/nvtx_sum_nvtx_sum.csv`
- `stats/cuda_gpu_trace_cuda_gpu_trace.csv`

The raw logs, full `.nsys-rep`, and exported sqlite remain on the L20 host and
are not committed because logs are ignored and the profile files are
binary/large.
