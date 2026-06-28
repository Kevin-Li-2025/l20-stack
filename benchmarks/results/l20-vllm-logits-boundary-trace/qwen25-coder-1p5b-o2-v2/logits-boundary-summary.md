# L20 vLLM Logits Boundary Trace Summary

- Trace: `/home/hhai/l20-stack/benchmarks/results/l20-vllm-logits-boundary-trace/qwen25-coder-1p5b-o2-v2/logits-boundary-trace.jsonl`
- Total events: `3621`
- Eligible events: `3431`
- Fallback events: `190`
- Eligible fraction: `0.9475`

## Fallback Reasons

| Reason | Count |
| --- | ---: |
| `prefill` | 190 |
| `not_single_token_decode` | 190 |

## Logits Shapes

| Shape | Count |
| --- | ---: |
| `1x151936` | 2630 |
| `4x151936` | 549 |
| `3x151936` | 218 |
| `8x151936` | 83 |
| `16x151936` | 76 |
| `2x151936` | 25 |
| `12x151936` | 11 |
| `15x151936` | 5 |
| `7x151936` | 4 |
| `5x151936` | 4 |
| `6x151936` | 4 |
| `9x151936` | 3 |
| `13x151936` | 3 |
| `10x151936` | 2 |
| `11x151936` | 2 |
| `14x151936` | 2 |
