# L20 vLLM Sampling ITL Results

This directory contains real vLLM serving results for the experimental L20
top-k/top-p sampler hook.

Hardware and serving shape:

- GPU: NVIDIA L20
- Model: Qwen2.5-Coder-1.5B-Instruct
- vLLM: local `/home/hhai/vllm-l20-rfc`
- Attention backend: FlashInfer
- Sampling: `temperature=0.8`, `top_k=50`, `top_p=0.9`
- Shape: random input 512, output 32, 32 prompts, 3 runs
- Limits: `max_model_len=2048`, `gpu_memory_utilization=0.70`

## Result

Median of 3 runs:

| Mode | Concurrency | Median ITL ms | Mean ITL ms | Output tok/s | Median TTFT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| FlashInfer clean | 1 | 5.060 | 5.037 | 171.0 | 29.0 |
| L20 no-trace | 1 | 6.697 | 6.707 | 133.9 | 36.0 |
| L20 trace | 1 | 6.913 | 6.925 | 129.0 | 37.9 |
| FlashInfer clean | 4 | 5.721 | 6.042 | 512.4 | 69.7 |
| L20 no-trace | 4 | 7.555 | 7.593 | 400.0 | 89.0 |
| L20 trace | 4 | 7.700 | 7.724 | 387.8 | 96.9 |

Deltas against clean FlashInfer:

| Mode | Concurrency | Median ITL delta | Mean ITL delta | Output throughput delta |
| --- | ---: | ---: | ---: | ---: |
| L20 no-trace | 1 | +32.36% | +33.17% | -21.70% |
| L20 no-trace | 4 | +32.06% | +25.68% | -21.94% |
| L20 trace | 1 | +36.63% | +37.50% | -24.57% |
| L20 trace | 4 | +34.59% | +27.84% | -24.30% |

Conclusion: the standalone two-stage Triton top-k/top-p sampler wins the
microbenchmark but loses real vLLM serving. The current hook should stay
experimental and disabled by default.

## Path Evidence

The trace run records `4251 / 4253` eligible events, so the negative result is
not a fallback artifact. The custom path really ran for nearly all decode
sampling calls. The two fallback events were large `256 x 151936` logits shapes
outside the measured profitability gate.

The main gap is integration overhead:

- one extra random-uniform generation on the vLLM hot path;
- Python gate and scalar top-k/top-p checks;
- two standalone Triton kernels outside vLLM's compiled sampler/CUDA graph;
- no fusion with the logits producer or FlashInfer's seed/offset sampling path.

## Artifacts

- `qwen25-coder-1p5b-summary.json`: aggregate summary and deltas
- `qwen25-coder-1p5b-flashinfer-clean-c1c4-i512-o32-r3/`: clean FlashInfer baseline
- `qwen25-coder-1p5b-l20-notrace-c1c4-i512-o32-r3/`: L20 hook performance run
- `qwen25-coder-1p5b-l20-c1c4-i512-o32-r3/`: trace-enabled proof run

## Reproduce

```bash
PYTHON=/home/hhai/venvs/vllm-l20/bin/python \
VLLM_SOURCE_TREE=/home/hhai/vllm-l20-rfc \
PORT=8101 \
INPUTS="512" \
CONCURRENCIES="1 4" \
RUNS=3 \
NUM_PROMPTS=32 \
OUTPUT_TOKENS=32 \
MAX_MODEL_LEN=2048 \
GPU_MEMORY_UTILIZATION=0.70 \
TEMPERATURE=0.8 \
TOP_P=0.9 \
TOP_K=50 \
scripts/run_vllm_l20_sampling_campaign.sh \
  /home/hhai/models/Qwen2.5-Coder-1.5B-Instruct \
  qwen25-coder-1p5b \
  flashinfer \
  benchmarks/results/l20-vllm-sampling-itl/qwen25-coder-1p5b-flashinfer-clean-c1c4-i512-o32-r3
```

Use `SAMPLER_MODE=l20` for the custom hook and set `L20_TRACE=1` only for path
proof runs, not for performance runs.
