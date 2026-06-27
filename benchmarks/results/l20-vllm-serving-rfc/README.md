# L20 vLLM Serving RFC Result

Date: 2026-06-28

This artifact compares vLLM FlashInfer serving with and without the experimental
L20 SM89 paged decode path. It is a real HTTP serving benchmark, not only a
standalone kernel benchmark.

## Setup

- GPU: NVIDIA L20
- vLLM source: `/home/hhai/vllm-l20-rfc`
- vLLM branch: `l20-sm89-paged-decode-rfc`
- vLLM commit used locally: `b81980aa5`
- Model: `Qwen/Qwen3-1.7B`
- Dtype: FP16
- Attention backend: `--attention-backend FLASHINFER`
- Execution mode: `--enforce-eager`
- Prompt/output shape: random 1024 input tokens, 64 output tokens
- Serving load: 24 prompts, request rate 1 RPS
- Endpoint: OpenAI `/v1/completions`

Runtime notes:

- FlashInfer sampling JIT needed CUDA 13 `nvcc`.
- FlashInfer sampling JIT also needed the venv `ninja` binary in `PATH`.
- The CLI flag `--attention-backend FLASHINFER` was required. The environment
  variable alone selected FlashAttention on this vLLM branch.
- The downloaded model directory was deleted after the run.

Path proof:

- `server-*.log` confirms `Using AttentionBackendEnum.FLASHINFER backend`.
- `l20-trace-smoke.txt` confirms the L20 path hit real vLLM decode calls before
  the clean benchmark run.

## Offline Latency

Common command shape: `vllm bench latency`, batch 1, input 1024, output 64,
FlashInfer backend, FP16, eager mode.

| Variant | Avg latency |
| --- | ---: |
| FlashInfer baseline | 0.873720658 s |
| L20 paged decode | 0.866071789 s |
| Delta | -0.875% |

## HTTP Serving

Two independent serving runs were collected for each variant.

| Run | Variant | Output tok/s | Mean TTFT | Median TTFT | P99 TTFT | Mean ITL | Median ITL | P99 ITL |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | Baseline | 61.696 | 74.764 ms | 73.655 ms | 96.490 ms | 13.693 ms | 13.536 ms | 29.029 ms |
| 1 | L20 path | 61.737 | 75.164 ms | 73.218 ms | 109.846 ms | 13.445 ms | 13.027 ms | 20.855 ms |
| 2 | Baseline | 61.688 | 75.991 ms | 75.158 ms | 99.339 ms | 13.548 ms | 13.346 ms | 28.345 ms |
| 2 | L20 path | 61.750 | 75.146 ms | 73.645 ms | 98.033 ms | 13.754 ms | 13.392 ms | 23.558 ms |

Mean of two serving runs:

| Metric | Baseline | L20 path | Delta |
| --- | ---: | ---: | ---: |
| Request throughput | 0.9639 req/s | 0.9647 req/s | +0.084% |
| Output throughput | 61.6918 tok/s | 61.7436 tok/s | +0.084% |
| Total token throughput | 1048.761 tok/s | 1049.641 tok/s | +0.084% |
| Mean TTFT | 75.378 ms | 75.155 ms | -0.295% |
| Median TTFT | 74.406 ms | 73.431 ms | -1.311% |
| P99 TTFT | 97.915 ms | 103.940 ms | +6.153% |
| Mean TPOT | 13.594 ms | 13.550 ms | -0.319% |
| Median TPOT | 13.623 ms | 13.676 ms | +0.388% |
| P99 TPOT | 14.677 ms | 15.213 ms | +3.655% |
| Mean ITL | 13.621 ms | 13.600 ms | -0.154% |
| Median ITL | 13.441 ms | 13.209 ms | -1.727% |
| P99 ITL | 28.687 ms | 22.206 ms | -22.591% |

## Conclusion

The L20 paged decode path is correctly wired into real vLLM FlashInfer serving
under eager execution, but the end-to-end gain is small on Qwen3-1.7B at this
load. Throughput improves only about 0.08% on average, mean ITL is essentially
flat, and median ITL improves about 1.7%. P99 ITL improved in both runs, but the
sample size is too small to claim stable tail-latency control. P99 TTFT is mixed.

The honest conclusion is that this path is useful as an upstream-shaped SM89
integration proof, not yet a strong production serving win. The next required
step is to make the path CUDA-graph-safe or move to a larger fused boundary,
because the current benchmark must run with `--enforce-eager` to exercise the
custom path.

