# L20 vLLM Logits Boundary Trace Summary

- Trace: `/home/hhai/l20-stack/benchmarks/results/l20-vllm-logits-boundary-rfc-shadow/qwen3-0p6b-o2-v1/logits-boundary-trace.jsonl`
- Total events: `775`
- Eligible events: `744`
- Fallback events: `31`
- Eligible fraction: `0.9600`
- Eligible logits materialization: `339.93 MiB`
- Total logits materialization: `500.77 MiB`
- Events without logits byte estimate: `0`
- Shadow epilogue events: `775`
- Shadow epilogue eligible: `744`
- Shadow avoidable logits materialization: `339.93 MiB`

## Fallback Reasons

| Reason | Count |
| --- | ---: |
| `prefill` | 30 |
| `not_single_token_decode` | 30 |
| `token_logprobs` | 2 |
| `min_p` | 2 |
| `penalties` | 2 |
| `logit_bias_or_min_tokens` | 2 |
| `bad_words` | 2 |
| `grammar_or_structured_output` | 1 |

## Shadow Epilogue Fallback Reasons

| Reason | Count |
| --- | ---: |
| `prefill` | 30 |
| `not_single_token_decode` | 30 |
| `token_logprobs` | 2 |
| `min_p` | 2 |
| `penalties` | 2 |
| `logit_bias_or_min_tokens` | 2 |
| `bad_words` | 2 |
| `grammar_or_structured_output` | 1 |

## Logits Materialization Budget

| Logits shape | Events | Eligible | Eligible logits MiB | Total logits MiB |
| --- | ---: | ---: | ---: | ---: |
| `1x151936` | 613 | 589 | 170.69 | 177.64 |
| `4x151936` | 124 | 120 | 139.10 | 143.74 |
| `3x151936` | 35 | 34 | 29.56 | 30.43 |
| `2x151936` | 1 | 1 | 0.58 | 0.58 |
| `256x151936` | 2 | 0 | 0.00 | 148.38 |

## Logits Shapes

| Shape | Count |
| --- | ---: |
| `1x151936` | 613 |
| `4x151936` | 124 |
| `3x151936` | 35 |
| `256x151936` | 2 |
| `2x151936` | 1 |
