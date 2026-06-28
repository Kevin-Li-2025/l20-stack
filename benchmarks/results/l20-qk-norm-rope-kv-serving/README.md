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

## Mini Matrix

Both mini matrices use one NVIDIA L20, vLLM O2, FlashInfer attention and
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

The result is a real serving integration of the custom L20 three-way path, but
it is still a low-single-digit system win.  It should not be described as an
industry-leading serving kernel yet.

## Artifacts

- `qwen3-0p6b-o2-mini-v1/qk-rope-kv-serving-summary.json`
- `qwen3-1p7b-o2-mini-v1/qk-rope-kv-serving-summary.json`
- Raw per-run serving reports live under each `qk-kv-off/` and `qk-kv-on/`
  directory.

## Profiling Gap

Nsight Compute/Systems profiling is still open.  The tested L20 host does not
currently expose `ncu` or `nsys` in `PATH`, so there is no honest kernel-count,
occupancy, warp-stall, L2, DRAM, or timeline artifact for this serving path yet.
Use `scripts/profile_kernel.sh` once Nsight Compute is installed on the L20
host.
