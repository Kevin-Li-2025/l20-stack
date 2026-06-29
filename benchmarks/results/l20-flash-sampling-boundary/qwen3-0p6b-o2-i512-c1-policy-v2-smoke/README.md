# Qwen3-0.6B FlashSampling Policy V2 Serving Smoke

This is the serving check for the `tile-policy-v2` standalone FlashSampling
candidate policy. It is a negative result for serving throughput.

## Setup

- Hardware: NVIDIA L20
- Model: Qwen3-0.6B
- vLLM mode: O2, FlashInfer attention
- Shape: input 512, output 32, concurrency 1, 16 prompts
- Candidate policy: `VLLM_L20_FLASHSAMPLING_CANDIDATE_MAX_BATCH=1`,
  `BLOCK_VOCAB=32`, `BLOCK_HIDDEN=256`

## Result

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Output throughput | 298.58 tok/s | 289.78 tok/s | -2.95% |
| Median ITL | 2.759 ms | 2.728 ms | -1.11% |
| P95 ITL | 2.934 ms | 2.948 ms | +0.45% |
| Median TTFT | 22.79 ms | 25.92 ms | +13.76% |

Candidate trace: 610 events, 608 eligible events.

## Decision

Do not claim a serving win. The tile policy repair removes most of the
standalone batch-one kernel regression, but the real vLLM path still loses
throughput and TTFT. This keeps the next target unchanged: a true LM-head GEMM
epilogue integration rather than a separate replacement kernel.

Raw reports are in `baseline/`, `candidate/`, and `summary.json`.
