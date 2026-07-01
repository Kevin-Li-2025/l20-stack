# A100 vLLM GEMM Epilogue Candidate

This artifact records the first output-changing LM-head greedy epilogue
candidate wired into real vLLM serving.

## Scope

| Item | Value |
| --- | --- |
| GPU | A100 |
| vLLM | 0.10.2 |
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Mode | batch-1, greedy, no penalties, no logprobs |
| Output path | `VLLM_L20_GEMM_EPILOGUE_ENABLE=1` |
| Candidate | no-full-logits Triton greedy argmax from LM-head weight |

This is a portability and boundary test for the L20 logits/LM-head research
line. It is not an L20 speed claim.

## Result

The candidate path is functionally live in serving, but it does not improve
no-trace ITL versus the same-session baseline.

| Run | Median ITL | Mean ITL | Median TTFT | Median total |
| --- | ---: | ---: | ---: | ---: |
| Baseline no trace | 6.727 ms | 6.260 ms | 12.425 ms | 435.464 ms |
| Candidate no trace | 6.733 ms | 6.133 ms | 11.686 ms | 434.504 ms |

Median ITL delta is +0.005 ms, or +0.08%. Treat this as equal within run noise.

## Path Proof

The trace-enabled candidate run shows the replacement path was actually used:

| Signal | Value |
| --- | ---: |
| Trace events | 384 |
| Eligible decode events | 378 |
| Candidate attempted | 378 |
| Candidate returned `SamplerOutput` | 378 |
| Events mutating output path | 378 |

Trace mode adds overhead and is not used for the ITL comparison.

## Interpretation

The output-changing path works, but the serving-level result rejects the simple
standalone greedy epilogue as a performance win on this A100/Qwen shape. The
earlier larger gap versus a default Qwen run came from sampling/penalty
configuration, not from removing logits materialization alone.

The next boundary should not be another standalone batch-1 greedy replacement.
It should either:

- integrate at the real LM-head producer epilogue without losing the optimized
  matmul path; or
- target sampling/penalty/logprob configurations where vLLM still pays a larger
  logits/sampler tax.

## Files

- `baseline-notrace/serving_itl_nopenalty_summary.json`
- `baseline-notrace/serving_itl_nopenalty_raw.jsonl`
- `candidate-notrace/serving_itl_nopenalty_summary.json`
- `candidate-notrace/serving_itl_nopenalty_raw.jsonl`
- `candidate-trace/gemm_epilogue_candidate_trace_summary.json`
- `candidate-trace/gemm_epilogue_trace.jsonl`
- `candidate-trace/serving_itl_nopenalty_summary.json`
- `candidate-trace/serving_itl_nopenalty_raw.jsonl`
