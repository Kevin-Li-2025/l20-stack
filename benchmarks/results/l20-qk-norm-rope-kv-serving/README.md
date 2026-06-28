# L20 Q/K Norm + RoPE + KV-Cache Serving

This directory tracks the first real vLLM serving integration for the custom
L20 three-way Q/K norm + Q/K RoPE + KV-cache write path.

This is not vLLM's native `enable_qk_norm_rope_fusion` result.  Both variants
force the native pass off:

```json
{"pass_config":{"enable_qk_norm_rope_fusion":false,"fuse_rope_kvcache":false}}
```

The custom-on variant sets `VLLM_L20_QK_ROPE_KV=1` and uses an experimental
Qwen3 hook installed by `integrations/vllm/install_l20_qk_norm_rope_kv.py`.
The hook mutates packed QKV in place, writes vLLM's paged KV cache through the
L20 Triton kernel, and calls attention with `skip_kv_cache_update=True` to
avoid a duplicate vLLM cache write.

## Smoke

`qwen3-0p6b-strict-smoke-o2-local/` is the strict O2 smoke.  It uses local model
path `/home/hhai/models/Qwen3-0.6B`, FlashInfer attention, FlashInfer sampling,
and CUDA graph decode.  It completed 1/1 requests with `failed=0`.  Earlier
failed attempts found and fixed three integration issues:

- Python trace file writes inside TorchDynamo capture caused graph breaks.
- Dynamic `qkv.shape[0]` logging specialized the dynamic token dimension.
- Passing `key=None,value=None` into FlashInfer attention failed; the working
  contract passes fused Q/K/V and only skips the duplicate KV-cache update.

## O2 Compile-Cache-Disabled R3

`qwen3-0p6b-o2-disable-cache-c1-i512-o16-r3-v1/` is the strongest current
serving artifact because it is paired with a positive Nsight Systems timeline.
It uses one NVIDIA L20, vLLM O2, FlashInfer attention and sampling, compile
cache disabled, input length 512, output length 16, `REQUEST_RATE=inf`, max
concurrency 1, three runs per variant, and 16 prompts per run.

| Variant | Output tok/s | Mean ITL | Median ITL | P99 ITL | Mean TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Custom off | 237.246 | 2.657 ms | 2.747 ms | 3.082 ms | 27.282 ms |
| Custom on | 239.579 | 2.534 ms | 2.623 ms | 3.099 ms | 28.348 ms |
| Change | +0.983% | -4.649% | -4.516% | +0.557% | +3.907% |

The stable signal is low-single-digit decode latency improvement. TTFT and p99
latency do not improve in this small matrix, so they should not be described as
wins. The companion timeline under
`benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1/`
shows 1,260 custom kernel instances and a 1.6% GPU-kernel-time share, explaining
why the serving gain is Amdahl-limited.

## Earlier Mini Matrix

These mini matrices use one NVIDIA L20, vLLM O2, FlashInfer attention and
sampling, input length 512, output length 32, `REQUEST_RATE=inf`, two runs per
shape, and 16 prompts per run.

| Model | Shapes | Output throughput | Mean ITL | Median ITL | P99 ITL | Mean TTFT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | c1/c4, i512 | +0.986% | -0.993% | -1.787% | +25.024% | -3.572% |
| Qwen3-1.7B | c1/c4, i512 | +2.775% | -0.044% | -0.210% | -0.928% | -9.792% |

Per-shape notes:

- Qwen3-0.6B c1/i512 was the best short-batch case: throughput +7.514%,
  median ITL -1.729%, mean TTFT -21.853%.
- Qwen3-0.6B c4/i512 regressed throughput by -1.813% and TTFT by +9.532%,
  while still improving mean/median ITL slightly.
- Qwen3-1.7B was more stable: throughput improved +2.240% to +2.475% across
  c1/c4, while mean/median ITL stayed essentially flat.

The result is an env-gated O2 serving comparison, but it is not sufficient proof
that the custom L20 three-way kernel executed in the production graph. A later
compile-cache-disabled timeline fixed that evidence gap for Qwen3-0.6B
i512/o16, so the older mini matrix should be treated as directional rather than
the primary proof artifact.

## Artifacts

- `qwen3-0p6b-o2-mini-v1/qk-rope-kv-serving-summary.json`
- `qwen3-1p7b-o2-mini-v1/qk-rope-kv-serving-summary.json`
- `qwen3-0p6b-o2-disable-cache-c1-i512-o16-r3-v1/qk-rope-kv-serving-summary.json`
- Raw per-run serving reports live under each `qk-kv-off/` and `qk-kv-on/`
  directory.

## Profiling Status

Nsight Compute is available on the L20 host outside the default `PATH`, and
deterministic kernel-counter profiles are checked in under
`benchmarks/results/ncu/qk-norm-rope-kv/`.

Nsight Systems is also available at
`/opt/nvidia/nsight-compute/2025.3.1/host/target-linux-x64/nsys`.  The first
serving-level zero-hit timeline is checked in under
`benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-c1-i512-v1/`.  It
captured 23,379 CUDA GPU kernel instances, 36,331 kernel launch API calls, 121
CUDA graph launches, and 0 custom QK/RoPE/KV kernel instances.

The corrected compile-cache-disabled timeline is checked in under
`benchmarks/results/nsys/qk-norm-rope-kv/qwen3-0p6b-o2-disable-cache-c1-i512-o16-v1/`.
It captured 17,995 CUDA GPU kernel instances, 28,987 kernel launch API calls,
121 CUDA graph launches, and 1,260 custom QK/RoPE/KV kernel instances. This is
the current serving-level path proof.
