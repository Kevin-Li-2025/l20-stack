# L20 vLLM Logits Boundary Campaign

- Campaign: `/home/hhai/l20-stack/benchmarks/results/l20-logits-boundary-ab-smoke/qwen25-coder-1p5b-c1c4-i512-r1/baseline-trace-only`
- Serving reports: `2`
- Trace events: `773`
- Eligible fraction: `0.9625`
- Eligible events: `744`
- Fallback events: `29`
- Eligible logits materialization: `339.93 MiB`
- Total logits materialization: `352.39 MiB`
- Events without logits byte estimate: `0`
- Shadow epilogue events: `773`
- Shadow epilogue eligible: `744`
- Shadow avoidable logits materialization: `339.93 MiB`

## Serving Shapes

| Concurrency | Input Tokens | Runs | Median TTFT ms | Median ITL ms | Output tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 512 | 1 | 33.27057 | 5.05075 | 168.68319 |
| 4 | 512 | 1 | 71.53187 | 5.40789 | 525.82546 |

## Logits Materialization Budget

| Logits shape | Events | Eligible | Eligible logits MiB | Total logits MiB |
| --- | ---: | ---: | ---: | ---: |
| `1x151936` | 613 | 589 | 170.69 | 177.64 |
| `4x151936` | 124 | 120 | 139.10 | 143.74 |
| `3x151936` | 35 | 34 | 29.56 | 30.43 |
| `2x151936` | 1 | 1 | 0.58 | 0.58 |

## Shadow Epilogue Fallback Reasons

| Reason | Count |
| --- | ---: |
| `not_single_token_decode` | 29 |
| `prefill` | 29 |

## Fallback Reasons

| Reason | Count |
| --- | ---: |
| `not_single_token_decode` | 29 |
| `prefill` | 29 |
