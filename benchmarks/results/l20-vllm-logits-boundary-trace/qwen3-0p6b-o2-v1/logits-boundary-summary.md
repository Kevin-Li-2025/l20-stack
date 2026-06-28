# L20 vLLM Logits Boundary Trace Summary

- Trace: `/home/hhai/l20-stack/benchmarks/results/l20-vllm-logits-boundary-trace/qwen3-0p6b-o2-v1/logits-boundary-trace.jsonl`
- Total events: `7242`
- Eligible events: `6859`
- Fallback events: `383`
- Eligible fraction: `0.9471`

## Fallback Reasons

| Reason | Count |
| --- | ---: |
| `prefill` | 382 |
| `not_single_token_decode` | 382 |
| `token_logprobs` | 2 |
| `min_p` | 2 |
| `penalties` | 2 |
| `logit_bias_or_min_tokens` | 2 |
| `bad_words` | 2 |
| `grammar_or_structured_output` | 1 |

## Logits Shapes

| Shape | Count |
| --- | ---: |
| `1x151936` | 5263 |
| `4x151936` | 1101 |
| `3x151936` | 433 |
| `8x151936` | 167 |
| `16x151936` | 152 |
| `2x151936` | 49 |
| `13x151936` | 22 |
| `15x151936` | 10 |
| `7x151936` | 8 |
| `5x151936` | 6 |
| `9x151936` | 6 |
| `12x151936` | 6 |
| `6x151936` | 5 |
| `10x151936` | 4 |
| `11x151936` | 4 |
| `14x151936` | 4 |
| `256x151936` | 2 |
