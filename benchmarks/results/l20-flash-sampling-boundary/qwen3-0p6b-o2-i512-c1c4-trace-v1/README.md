# Qwen3-0.6B FlashSampling Shadow Trace, L20 O2

This is a behavior-preserving vLLM serving trace run on a single NVIDIA L20. The
hook runs after `compute_logits`, so it proves path eligibility and logits bytes
that a future LM-head FlashSampling epilogue could avoid. It is not a serving
speedup claim yet.

## Setup

- model: `/home/hhai/models/Qwen3-0.6B`
- served name: `qwen3-0p6b`
- vLLM source: `/home/hhai/vllm-l20-rfc`
- execution mode: `o2`
- attention backend: `FLASHINFER`
- input tokens: `512`
- concurrencies: `1, 4`
- output tokens: `32`
- sampling: temperature `0.8`, top-k `-1`, top-p `1.0`
- commit: `0c9ab6f`

## FlashSampling Gate

- total events: 775
- eligible events: 744
- eligible fraction: 96.00%
- avoidable logits materialization: 339.93 MiB
- total traced logits materialization: 500.77 MiB

Eligible shape counts:

- `b1-h1024-v151936-gumbel`: 589
- `b2-h1024-v151936-gumbel`: 1
- `b3-h1024-v151936-gumbel`: 34
- `b4-h1024-v151936-gumbel`: 120


Fallback reasons are dominated by prefill/not-single-token decode and two warmup
or initialization events that still used top-k/top-p defaults. See
`flashsampling-summary.json` and `logits-boundary-summary.json` for full counts.

## Serving Smoke Metrics

| max concurrency | completed | failed | output tok/s | p50 TTFT ms | p95 TTFT ms | p50 ITL ms | p95 ITL ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 0 | 297.35 | 24.31 | 29.58 | 2.74 | 3.04 |
| 4 | 16 | 0 | 866.30 | 47.00 | 65.75 | 3.18 | 3.89 |

## Artifacts

- `run-config.json`
- `c1-i512-r1.json`
- `c4-i512-r1.json`
- `flashsampling-summary.json`
- `flashsampling-summary.md`
- `logits-boundary-summary.json`
- `logits-boundary-summary.md`

Raw trace JSONL and `server.log` are intentionally not checked in; they remain on
the L20 host under `/home/hhai/tmp/l20-flashsampling-qwen3-0p6b-i512-c1c4-20260629-150517`.
