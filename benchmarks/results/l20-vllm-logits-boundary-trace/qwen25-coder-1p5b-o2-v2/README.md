# L20 vLLM Logits Boundary Campaign

- Campaign: `/home/hhai/l20-stack/benchmarks/results/l20-vllm-logits-boundary-trace/qwen25-coder-1p5b-o2-v2`
- Serving reports: `9`
- Trace events: `3621`
- Eligible fraction: `0.9475`
- Eligible events: `3431`
- Fallback events: `190`

## Serving Shapes

| Concurrency | Input Tokens | Runs | Median TTFT ms | Median ITL ms | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 128 | 1 | 20.29497 | 5.03767 | 181.54460 |
| 1 | 512 | 1 | 29.87899 | 5.04969 | 171.01171 |
| 1 | 2048 | 1 | 88.33696 | 5.08813 | 131.80505 |
| 4 | 128 | 1 | 34.01377 | 5.36554 | 627.70014 |
| 4 | 512 | 1 | 75.15761 | 5.39989 | 517.99724 |
| 4 | 2048 | 1 | 241.93513 | 5.65542 | 290.92349 |
| 16 | 128 | 1 | 99.25313 | 5.39974 | 1551.71152 |
| 16 | 512 | 1 | 150.99576 | 5.59083 | 1031.49487 |
| 16 | 2048 | 1 | 399.05519 | 6.69029 | 405.35458 |

## Fallback Reasons

| Reason | Count |
| --- | ---: |
| `not_single_token_decode` | 190 |
| `prefill` | 190 |
