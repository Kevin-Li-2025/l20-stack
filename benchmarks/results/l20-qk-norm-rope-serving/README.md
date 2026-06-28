# L20 Q/K Norm + RoPE Serving Smoke

This directory tracks a small vLLM O2 serving smoke for Qwen3-style Q/K norm
models on the NVIDIA L20.

The current artifact compares vLLM with `enable_qk_norm_rope_fusion=false`
against `enable_qk_norm_rope_fusion=true` while keeping FlashInfer attention,
FlashInfer sampling, and CUDA graph capture enabled. It is a gate for the larger
L20 fused boundary, not a claim that `integrations/vllm/l20_qk_norm_rope_kv.py`
is already wired into production serving.

Current smoke:

| Model | Mode | Output throughput change | Mean ITL change | Median ITL change | P99 ITL change | Mean TTFT change |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B | O2 QK fusion on vs off | +0.007% | -3.339% | -2.765% | -6.884% | +6.983% |

The ITL direction is useful, but this run has only one paired serving sample.
The result should be treated as a smoke signal until regenerated with more runs
and raw per-run JSON artifacts.

Regenerate:

```bash
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
PYTHONPATH=/home/hhai/vllm-l20-rfc:/home/hhai/l20-stack \
RUNS=2 NUM_PROMPTS=24 OUTPUT_TOKENS=64 INPUTS=512 CONCURRENCIES=1 PORT=8011 \
scripts/run_vllm_l20_qk_norm_rope_serving_smoke.sh \
  /home/hhai/models/Qwen3-0.6B \
  qwen3-0p6b \
  benchmarks/results/l20-qk-norm-rope-serving/qwen3-0p6b-o2-rerun \
  /home/hhai/vllm-l20-rfc
```
