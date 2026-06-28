# L20 vLLM Logits Boundary Trace Matrix

This directory contains raw vLLM serving reports and trace-only logits-boundary
gate evidence from a real NVIDIA L20 run. The hook is behavior-preserving: it
records whether a future LM-head epilogue / fused sampling path would be safe,
but it does not change logits, sampling, or generated tokens.

## Run Setup

- GPU: NVIDIA L20 / SM89
- vLLM source: `/home/hhai/vllm-l20-upstream`
- Execution mode: `o2`
- Attention backend: `FLASHINFER`
- Sampling: `temperature=0.8`, `top_p=0.9`, `top_k=50`, `min_p=0.0`
- Inputs: `128`, `512`, `2048`
- Concurrency: `1`, `4`, `16`
- Output tokens: `32`
- Request rate: `inf`

The Qwen3-0.6B campaign used two runs per shape. The Qwen3-1.7B and
Qwen2.5-Coder-1.5B campaigns used one run per shape. The trace hook writes Python
JSONL from the serving process, so the serving latency numbers here are path
sanity checks and workload context, not a clean performance comparison.

## Campaign Summary

| Campaign | Serving Reports | Trace Events | Eligible Events | Eligible Fraction | Fallback Pattern |
| --- | ---: | ---: | ---: | ---: | --- |
| `qwen3-0p6b-o2-v1` | 18 | 7242 | 6859 | 94.71% | Mostly prefill / non-single-token decode |
| `qwen3-1p7b-o2-v1` | 9 | 3619 | 3428 | 94.72% | Mostly prefill / non-single-token decode |
| `qwen25-coder-1p5b-o2-v2` | 9 | 3621 | 3431 | 94.75% | Prefill / non-single-token decode only |

## Median ITL and Throughput

| Campaign | Concurrency | Input Tokens | Median ITL ms | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| `qwen3-0p6b-o2-v1` | 1 | 128 | 2.86487 | 285.78821 |
| `qwen3-0p6b-o2-v1` | 1 | 512 | 2.87274 | 276.73222 |
| `qwen3-0p6b-o2-v1` | 1 | 2048 | 3.15962 | 214.11408 |
| `qwen3-0p6b-o2-v1` | 4 | 128 | 3.08796 | 968.90457 |
| `qwen3-0p6b-o2-v1` | 4 | 512 | 3.33956 | 849.44076 |
| `qwen3-0p6b-o2-v1` | 4 | 2048 | 4.30920 | 461.04382 |
| `qwen3-0p6b-o2-v1` | 16 | 128 | 3.43457 | 2322.10985 |
| `qwen3-0p6b-o2-v1` | 16 | 512 | 4.37970 | 1639.83311 |
| `qwen3-0p6b-o2-v1` | 16 | 2048 | 7.90310 | 643.76239 |
| `qwen3-1p7b-o2-v1` | 1 | 128 | 5.99221 | 153.57569 |
| `qwen3-1p7b-o2-v1` | 1 | 512 | 6.01166 | 144.37445 |
| `qwen3-1p7b-o2-v1` | 1 | 2048 | 6.29030 | 112.88957 |
| `qwen3-1p7b-o2-v1` | 4 | 128 | 6.24571 | 548.99946 |
| `qwen3-1p7b-o2-v1` | 4 | 512 | 6.49777 | 452.48553 |
| `qwen3-1p7b-o2-v1` | 4 | 2048 | 7.44457 | 245.41024 |
| `qwen3-1p7b-o2-v1` | 16 | 128 | 6.52928 | 1339.47220 |
| `qwen3-1p7b-o2-v1` | 16 | 512 | 7.41236 | 871.12117 |
| `qwen3-1p7b-o2-v1` | 16 | 2048 | 11.37594 | 337.86929 |
| `qwen25-coder-1p5b-o2-v2` | 1 | 128 | 5.03767 | 181.54460 |
| `qwen25-coder-1p5b-o2-v2` | 1 | 512 | 5.04969 | 171.01171 |
| `qwen25-coder-1p5b-o2-v2` | 1 | 2048 | 5.08813 | 131.80505 |
| `qwen25-coder-1p5b-o2-v2` | 4 | 128 | 5.36554 | 627.70014 |
| `qwen25-coder-1p5b-o2-v2` | 4 | 512 | 5.39989 | 517.99724 |
| `qwen25-coder-1p5b-o2-v2` | 4 | 2048 | 5.65542 | 290.92349 |
| `qwen25-coder-1p5b-o2-v2` | 16 | 128 | 5.39974 | 1551.71152 |
| `qwen25-coder-1p5b-o2-v2` | 16 | 512 | 5.59083 | 1031.49487 |
| `qwen25-coder-1p5b-o2-v2` | 16 | 2048 | 6.69029 | 405.35458 |

## Conclusion

Across three L20 serving campaigns, normal decode steps are eligible for a
future fused logits-boundary sampling path about 94.7% of the time. The dominant
fallback is expected prefill / multi-token scheduling; for Qwen2.5-Coder after
the v2 runner trace fix, no extra sampling-policy blockers appeared.

This makes fused GPU-side top-k / top-p / multinomial sampling the next highest
leverage implementation target. It has a real vLLM boundary, broad decode
coverage, and can collapse logits post-processing and sampling work without
touching prefill.

The c16/i2048 rows show tail-latency pressure, especially on larger models. That
is useful evidence for scheduler and KV-pressure work, but it is not solved by a
logits-boundary sampling kernel alone.
