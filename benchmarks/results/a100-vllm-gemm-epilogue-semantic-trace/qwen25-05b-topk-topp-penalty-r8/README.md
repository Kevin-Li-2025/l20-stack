# A100 GEMM Epilogue Semantic Trace

This is a trace-only vLLM serving run for the upstream-shaped LM-head/GEMM
epilogue boundary. It does not mutate outputs and does not claim a latency win.

## Setup

- GPU: NVIDIA A100-SXM4-80GB
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- vLLM: `0.10.2`
- Workload: `sample_topk_topp_penalty`
- Sampling: temperature `0.8`, top-k `50`, top-p `0.9`, frequency penalty
  `0.1`, presence penalty `0.1`, repetition penalty `1.05`
- FlashInfer sampler: enabled and CUDA 13 JIT prewarmed
- Output-changing epilogue path: disabled

## Result

| Metric | Value |
| --- | ---: |
| Trace events | 320 |
| Decode-safe hook events | 310 / 320 |
| Semantic P0 target events | 320 / 320 |
| Decode-safe semantic P0 events | 310 / 320 |
| Decode-side avoidable logits materialization | 179.67 MiB FP32 |
| History source | `input_batch_token_ids_cpu` |

All events are classified as
`fused_topk_topp_sparse_penalty_lm_head_epilogue`. Ten events are prefill/profile
events and are rejected by the runtime event gate with
`not_single_token_decode` / `scheduled_tokens_mismatch`.

## Decision

This proves the next producer-side boundary is real in serving traffic:
top-k/top-p plus sparse token-history penalties can be recognized before
`compute_logits`, and the hook can see a history source. The next step is still
not a speed claim; it is an output-changing shadow/correctness pass that compares
candidate sampled tokens against the baseline sampler while continuing to fall
back for unsupported semantics.
