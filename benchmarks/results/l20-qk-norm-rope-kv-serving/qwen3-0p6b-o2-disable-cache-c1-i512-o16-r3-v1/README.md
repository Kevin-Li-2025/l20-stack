# Qwen3-0.6B O2 i512/o16 Serving R3

This run compares vLLM O2 serving with the custom L20 Q/K norm + Q/K RoPE +
KV-cache write path disabled versus enabled. vLLM native
`enable_qk_norm_rope_fusion` and `fuse_rope_kvcache` are disabled in both
variants, so this is not the upstream native QK fusion path.

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
| Prompts per run | 16 |
| Runs per variant | 3 |
| Max concurrency | 1 |
| Request rate | `inf` |

## Median-of-3 Summary

| Variant | Output tok/s | Mean ITL | Median ITL | P99 ITL | Mean TTFT | Median TTFT | P99 TTFT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Custom off | 237.246 | 2.657 ms | 2.747 ms | 3.082 ms | 27.282 ms | 27.845 ms | 31.793 ms |
| Custom on | 239.579 | 2.534 ms | 2.623 ms | 3.099 ms | 28.348 ms | 28.463 ms | 33.856 ms |
| Change | +0.983% | -4.649% | -4.516% | +0.557% | +3.907% | +2.220% | +6.487% |

## Interpretation

The stable signal is a low-single-digit decode latency improvement: mean ITL
improves by 4.649% and median ITL improves by 4.516%. Output throughput
improves by 0.983%. TTFT and p99 latency regress in this small matrix, so they
should not be claimed as wins.

This result should be read with the companion Nsight Systems run in
`benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1/`.
That timeline proves 1,260 custom kernel launches in O2 and shows the custom
kernel is only 1.6% of GPU kernel time, which explains the Amdahl-limited
serving impact.

## Artifacts

- `qk-rope-kv-serving-summary.json`
- `qk-kv-off/c1-i512-r1.json`
- `qk-kv-off/c1-i512-r2.json`
- `qk-kv-off/c1-i512-r3.json`
- `qk-kv-on/c1-i512-r1.json`
- `qk-kv-on/c1-i512-r2.json`
- `qk-kv-on/c1-i512-r3.json`
