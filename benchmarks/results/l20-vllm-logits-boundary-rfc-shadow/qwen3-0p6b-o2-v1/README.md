# L20 vLLM Logits Boundary Campaign

- Campaign: `benchmarks/results/l20-vllm-logits-boundary-rfc-shadow/qwen3-0p6b-o2-v1`
- Serving reports: `2`
- Trace events: `775`
- Eligible fraction: `0.9600`
- Eligible events: `744`
- Fallback events: `31`
- Eligible logits materialization: `339.93 MiB`
- Total logits materialization: `500.77 MiB`
- Events without logits byte estimate: `0`
- Shadow epilogue events: `775`
- Shadow epilogue eligible: `744`
- Shadow avoidable logits materialization: `339.93 MiB`

## Serving Shapes

| Concurrency | Input Tokens | Runs | Median TTFT ms | Median ITL ms | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 1 | 33.19063 | 2.82927 | 273.15172 |
| 4 | 512 | 1 | 50.65035 | 3.27361 | 834.28947 |

## Logits Materialization Budget

| Logits shape | Events | Eligible | Eligible logits MiB | Total logits MiB |
| --- | ---: | ---: | ---: | ---: |
| `1x151936` | 613 | 589 | 170.69 | 177.64 |
| `4x151936` | 124 | 120 | 139.10 | 143.74 |
| `3x151936` | 35 | 34 | 29.56 | 30.43 |
| `2x151936` | 1 | 1 | 0.58 | 0.58 |
| `256x151936` | 2 | 0 | 0.00 | 148.38 |

## Shadow Epilogue Fallback Reasons

| Reason | Count |
| --- | ---: |
| `bad_words` | 2 |
| `grammar_or_structured_output` | 1 |
| `logit_bias_or_min_tokens` | 2 |
| `min_p` | 2 |
| `not_single_token_decode` | 30 |
| `penalties` | 2 |
| `prefill` | 30 |
| `token_logprobs` | 2 |

## Fallback Reasons

| Reason | Count |
| --- | ---: |
| `bad_words` | 2 |
| `grammar_or_structured_output` | 1 |
| `logit_bias_or_min_tokens` | 2 |
| `min_p` | 2 |
| `not_single_token_decode` | 30 |
| `penalties` | 2 |
| `prefill` | 30 |
| `token_logprobs` | 2 |
