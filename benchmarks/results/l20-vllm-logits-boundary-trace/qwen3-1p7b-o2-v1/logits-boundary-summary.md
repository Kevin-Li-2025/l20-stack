# L20 vLLM Logits Boundary Trace Summary

- Trace: `/home/hhai/l20-stack/benchmarks/results/l20-vllm-logits-boundary-trace/qwen3-1p7b-o2-v1/logits-boundary-trace.jsonl`
- Total events: `3619`
- Eligible events: `3428`
- Fallback events: `191`
- Eligible fraction: `0.9472`

## Fallback Reasons

| Reason | Count |
| --- | ---: |
| `prefill` | 190 |
| `not_single_token_decode` | 190 |
| `token_logprobs` | 2 |
| `min_p` | 2 |
| `penalties` | 2 |
| `logit_bias_or_min_tokens` | 2 |
| `bad_words` | 2 |
| `grammar_or_structured_output` | 1 |

## Logits Shapes

| Shape | Count |
| --- | ---: |
| `1x151936` | 2630 |
| `4x151936` | 550 |
| `3x151936` | 216 |
| `8x151936` | 83 |
| `16x151936` | 76 |
| `2x151936` | 24 |
| `14x151936` | 10 |
| `15x151936` | 5 |
| `7x151936` | 4 |
| `5x151936` | 4 |
| `9x151936` | 3 |
| `13x151936` | 3 |
| `12x151936` | 3 |
| `256x151936` | 2 |
| `6x151936` | 2 |
| `10x151936` | 2 |
| `11x151936` | 2 |
