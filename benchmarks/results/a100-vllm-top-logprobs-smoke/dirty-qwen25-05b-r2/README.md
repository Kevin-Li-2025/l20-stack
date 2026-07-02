# Dirty A100 vLLM Top-Logprobs Serving Smoke

This is a path-proof smoke for the fused top-logprobs vLLM hook. It is not a
clean performance artifact: the A100 had an unrelated process using about
14.5 GiB VRAM at 100% GPU utilization before the run.

## Setup

- GPU: NVIDIA A100-SXM4-80GB
- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- vLLM: `0.10.2`
- Torch: `2.8.0+cu128`
- Workload: `sample_topk_topp_penalty_logprobs`
- Output length: 16 tokens
- Measured requests: 2, with 1 warmup

Both baseline and candidate keep FlashInfer top-k/top-p sampling enabled. The
candidate only changes generated-token logprobs gathering.

## Dirty Latency Signal

| Metric | Native logprobs median | Fused top-logprobs median | Delta |
| --- | ---: | ---: | ---: |
| ITL | 13.376 ms | 12.485 ms | -6.66% |
| ms/output token | 13.963 ms | 13.196 ms | -5.50% |
| Total request time | 223.412 ms | 211.131 ms | -5.50% |
| TTFT | 20.523 ms | 18.909 ms | -7.86% |

## Path Proof

Trace run:

| Trace metric | Value |
| --- | ---: |
| Total events | 8 |
| Eligible fused events | 8 |
| Fallback events | 0 |
| Eligible fraction | 100.00% |

## Claim Boundary

- This proves the fused top-logprobs hook reaches real vLLM HTTP serving.
- This does not prove a clean serving speedup because the GPU was busy.
- A clean A100/L20 run with `REQUIRE_IDLE=1`, more requests, and repeated pairs
  is still required before updating the public README with a serving win.
