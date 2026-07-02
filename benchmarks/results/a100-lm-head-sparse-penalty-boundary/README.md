# A100 LM-Head Sparse-Penalty Boundary

This artifact tests a stricter version of the logits-boundary hypothesis:
apply sparse token-history penalties inside LM-head vocab tiles, then reduce to
the winning token, instead of materializing full logits and applying penalties
afterward.

The path is correct, but it is not profitable as a standalone Triton LM-head
replacement on A100. This keeps the next step narrow: a real win needs a
producer-side GEMM epilogue or upstream-shaped LM-head integration, not another
external GEMM rewrite.

## Runs

| Shape | Baseline full logits + sparse penalty + argmax | Producer-side tile path | Ratio |
| --- | ---: | ---: | ---: |
| `b1 h256 v8192 history16` | 0.142 ms | 0.188 ms | 1.32x slower |
| `b1 h1536 v151936 history32` | 0.320 ms | 0.444 ms | 1.39x slower |

## Reproduce

```bash
PYTHONPATH=src python scripts/benchmark_lm_head_sparse_penalty_boundary.py \
  --batch 1 --hidden 1536 --vocab 151936 --max-history 32 \
  --rounds 6 --warmup 3 \
  --output /tmp/sgi-lmhead-sparse-qwenish.json
```

## Decision

- Keep the in-tile sparse penalty implementation as a correctness/profiling
  prototype.
- Do not claim a serving win from this boundary.
- Continue with a true LM-head/GEMM epilogue or an upstream vLLM
  `ParallelLMHead`/`LogitsProcessor` boundary where logits are never produced as
  a standalone `[batch, vocab]` tensor.
