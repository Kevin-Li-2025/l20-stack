# A100 Sparse Top-k/Top-p + Penalty Microbenchmark

This artifact tests the serving-shaped successor to the dense-count penalty
prototype. Instead of assuming a dense `[batch, vocab]` token-count matrix, the
new path consumes sparse token history:

```text
history_tokens[batch, max_history] + history_lengths[batch]
```

The kernel boundary is:

```text
copy logits to FP32 workspace
-> sparse token-history scatter penalties
-> existing two-stage top-k/top-p sampler
```

This is still a microbenchmark. It is not a vLLM serving ITL win yet.

## Results

Hardware: NVIDIA A100-SXM4-80GB

Shape: Qwen vocab 151936, top-k 50, top-p 0.9, temperature 0.8,
128 history tokens, FP16 logits.

| Batch | Sparse history path | Dense apply then sample | Speedup vs apply | Dense-count fused | Sparse / dense-count |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.1531 ms | 0.1944 ms | 1.27x | 0.1442 ms | 0.94x |
| 4 | 0.1800 ms | 0.2365 ms | 1.31x | 0.1661 ms | 0.92x |

## Interpretation

The result is useful because it removes the unrealistic dense-count assumption
while staying faster than a separate penalty-then-sampling path. The remaining
gap to dense-count fused is expected: sparse v1 adds a copy kernel and a scatter
kernel before top-k/top-p reduction.

The next step is not to claim serving speedup. It is to wire this path into
vLLM with an explicit opt-in gate and run paired ITL A/B on requests that use
top-k/top-p plus penalties.

The current vLLM installer now carries that opt-in boundary:

```text
VLLM_L20_TOPK_TOPP_DEFER_PENALTIES=1
```

When enabled, the patched vLLM path builds a bounded sparse history window from
request token IDs and defers penalties into the custom sampler. If the custom
sampler is not eligible, it fails fast instead of falling back to unpenalized
sampling. The default path leaves vLLM penalty handling unchanged.

## Files

- `qwen-vocab-b1-h128.json`
- `qwen-vocab-b4-h128.json`
