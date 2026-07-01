# Sampler Semantics Target Plan

| Case | Median ITL | Delta vs greedy | Target | Priority |
| --- | ---: | ---: | --- | --- |
| `greedy_no_penalty` | 6.720 ms | +0.00% | `greedy_no_penalty_control` | `control` |
| `greedy_default_repetition` | 9.224 ms | +37.27% | `fused_repetition_penalty` | `p1` |
| `sample_topk_topp` | 9.544 ms | +42.03% | `fused_topk_topp` | `p0` |
| `sample_topk_topp_penalty` | 9.562 ms | +42.29% | `fused_topk_topp+penalty` | `p0` |
| `greedy_token_logprobs` | 9.336 ms | +38.94% | `fused_token_logprobs` | `p0` |

## Recommendation

Start with `fused_topk_topp+penalty` because it is a P0 semantics path with the largest observed ITL delta in the probe.
