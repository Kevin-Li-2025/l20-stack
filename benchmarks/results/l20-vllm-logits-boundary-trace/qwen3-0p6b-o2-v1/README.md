# L20 vLLM Logits Boundary Campaign

- Campaign: `/home/hhai/l20-stack/benchmarks/results/l20-vllm-logits-boundary-trace/qwen3-0p6b-o2-v1`
- Serving reports: `18`
- Trace events: `7242`
- Eligible fraction: `0.9471`
- Eligible events: `6859`
- Fallback events: `383`

## Serving Shapes

| Concurrency | Input Tokens | Runs | Median TTFT ms | Median ITL ms | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 2 | 23.06256 | 2.86487 | 285.78821 |
| 1 | 512 | 2 | 28.38485 | 2.87274 | 276.73222 |
| 1 | 2048 | 2 | 55.98072 | 3.15962 | 214.11408 |
| 4 | 128 | 2 | 34.50687 | 3.08796 | 968.90457 |
| 4 | 512 | 2 | 45.66319 | 3.33956 | 849.44076 |
| 4 | 2048 | 2 | 137.17668 | 4.30920 | 461.04382 |
| 16 | 128 | 2 | 76.48418 | 3.43457 | 2322.10985 |
| 16 | 512 | 2 | 101.42831 | 4.37970 | 1639.83311 |
| 16 | 2048 | 2 | 217.29865 | 7.90310 | 643.76239 |

## Fallback Reasons

| Reason | Count |
| --- | ---: |
| `bad_words` | 2 |
| `grammar_or_structured_output` | 1 |
| `logit_bias_or_min_tokens` | 2 |
| `min_p` | 2 |
| `not_single_token_decode` | 382 |
| `penalties` | 2 |
| `prefill` | 382 |
| `token_logprobs` | 2 |
