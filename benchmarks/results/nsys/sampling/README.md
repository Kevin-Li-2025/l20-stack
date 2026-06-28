# L20 vLLM Sampling Nsight Systems Timelines

This directory tracks serving-level Nsight Systems profiles for stochastic
sampling paths on one NVIDIA L20.

## Runs

| Run | Model | Sampler | Shape | Result |
| --- | --- | --- | --- | --- |
| `qwen25-coder-1p5b-flashinfer-c4-i512-o32-v2/` | Qwen2.5-Coder-1.5B-Instruct | FlashInfer top-k/top-p | c4, input 512, output 32, 16 prompts | Positive GPU-sampler path proof. Matched sampler kernel instances: 270. |
| `qwen25-coder-1p5b-l20-active-c4-i512-o32-v1/` | Qwen2.5-Coder-1.5B-Instruct | Experimental L20 top-k/top-p hook | c4, input 512, output 32, 16 prompts | Active hook path proof, but median ITL regresses to 7.879 ms versus 5.426 ms for FlashInfer. |

The raw `.nsys-rep`, `.sqlite`, and server logs are intentionally not checked
in. They remain on the L20 host under the matching result directory.

## Current Finding

The v2 FlashInfer run uses `--generation-config vllm`, prewarms FlashInfer
sampling with CUDA 13 nvcc, and records the expected server-log branch:
`Using FlashInfer for top-p & top-k sampling`.

The timeline confirms real GPU sampler kernels:

| Kernel | Instances | Avg time | Time share |
| --- | ---: | ---: | ---: |
| `_topk_topp_kernel` | 2 | 4.242 ms | 0.7% |
| `flashinfer::sampling::TopPSamplingFromProbKernel` | 134 | 38.420 us | 0.4% |
| `flashinfer::sampling::RadixTopKMaskLogitsKernel_MultiCTA` | 134 | 27.755 us | 0.3% |

The family attribution artifact for the same run shows why this should stay a
system-boundary project rather than a standalone sampler project: CUTLASS/cuBLAS
GEMM is 42.99% of GPU time, PyTorch fill/bookkeeping kernels are 41.72%,
FlashInfer attention is 1.96%, FlashInfer sampling is 0.69%, and vLLM's native
`_topk_topp_kernel` is 0.66%. On the CUDA API side, sync/memcpy/launch account
for 43.76%, 13.98%, and 13.51% of API time.

The active L20 hook run proves the custom sampler enters real serving: 132 of
134 sampling events were L20-eligible, and Nsight Systems captured 132
instances each of `_topk_topp_partial_kernel` and
`_topk_topp_reduce_sample_seed_kernel`. That run is not a win: median ITL
regressed from 5.426 ms to 7.879 ms, and the custom L20 kernels account for only
1.98% of GPU time.

Together these runs support the existing conclusion from the paired serving
matrix: the production FlashInfer sampler route is real and modestly useful,
but a standalone replacement sampler is unlikely to be the next large win. The
more valuable next boundary is fusing sampling with the logits producer or
LM-head epilogue so the full logits tensor does not need a separate
postprocessing pipeline.
