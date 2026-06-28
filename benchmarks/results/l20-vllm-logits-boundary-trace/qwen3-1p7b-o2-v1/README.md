# L20 vLLM Logits Boundary Campaign

- Campaign: `/home/hhai/l20-stack/benchmarks/results/l20-vllm-logits-boundary-trace/qwen3-1p7b-o2-v1`
- Serving reports: `9`
- Trace events: `3619`
- Eligible fraction: `0.9472`
- Eligible events: `3428`
- Fallback events: `191`

## Serving Shapes

| Concurrency | Input Tokens | Runs | Median TTFT ms | Median ITL ms | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1 | 23.23046 | 5.99221 | 153.57569 |
| 1 | 512 | 1 | 36.54330 | 6.01166 | 144.37445 |
| 1 | 2048 | 1 | 91.99581 | 6.29030 | 112.88957 |
| 4 | 128 | 1 | 38.38918 | 6.24571 | 548.99946 |
| 4 | 512 | 1 | 81.89473 | 6.49777 | 452.48553 |
| 4 | 2048 | 1 | 261.36215 | 7.44457 | 245.41024 |
| 16 | 128 | 1 | 111.23196 | 6.52928 | 1339.47220 |
| 16 | 512 | 1 | 172.00654 | 7.41236 | 871.12117 |
| 16 | 2048 | 1 | 434.55115 | 11.37594 | 337.86929 |

## Fallback Reasons

| Reason | Count |
| --- | ---: |
| `bad_words` | 2 |
| `grammar_or_structured_output` | 1 |
| `logit_bias_or_min_tokens` | 2 |
| `min_p` | 2 |
| `not_single_token_decode` | 190 |
| `penalties` | 2 |
| `prefill` | 190 |
| `token_logprobs` | 2 |
