# L20 Q/K Norm + RoPE + KV Write Nsight Compute Profile

This directory contains the first Nsight Compute artifacts for the custom
L20-only Q/K norm + Q/K RoPE + KV write kernel.

## Tooling Status

The remote L20 host did have Nsight Compute installed, but `ncu` was not in the
default shell `PATH`.

Detected binaries:

- `/usr/local/cuda-13.0/bin/ncu`
- `/opt/nvidia/nsight-compute/2025.3.1/ncu`
- `/opt/nvidia/nsight-compute/2025.3.1/target/linux-desktop-glibc_2_11_3-x64/ncu`

Detected packages:

- `cuda-nsight-compute-13-0 13.0.3-1`
- `nsight-compute-2025.3.1 2025.3.1.4-1`

Hardware:

- GPU: NVIDIA L20
- Driver: 580.159.04
- Visible memory: 46068 MiB

Normal-user counter collection failed with `ERR_NVGPUCTRPERM`, and
`/proc/driver/nvidia/params` had `RmProfilingAdminOnly: 1`. The checked-in
counter artifacts were collected through an elevated Nsight Compute invocation.
No local credential or sudo material is stored in this repo.

## Commands

The profiling wrapper now auto-discovers common Nsight Compute locations:

```bash
scripts/profile_kernel.sh \
  --output benchmarks/results/ncu/qk-norm-rope-kv/qk-norm-rope-kv-sudo-first \
  --kernel-name 'regex:_l20_qk_norm_rope_kv_kernel' \
  -- env PYTHONPATH=src python scripts/benchmark_qk_norm_rope_kv.py \
    --output benchmarks/results/ncu/qk-norm-rope-kv/qk-norm-rope-kv-sudo-bench.json
```

If the host blocks counters for normal users, run the same command through a
local root wrapper or sudo session. Do not add the wrapper or credentials to the
repository.

## Microbenchmark Timing

All reported token shapes passed correctness. The sudo timing run measured:

| Tokens | Baseline | Fused | Speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.008992 ms | 0.006287 ms | 1.430x |
| 8 | 0.009430 ms | 0.007163 ms | 1.316x |
| 16 | 0.009656 ms | 0.007467 ms | 1.293x |
| 32 | 0.010272 ms | 0.007899 ms | 1.300x |
| 64 | 0.011351 ms | 0.008251 ms | 1.376x |

## Nsight Counter Summary

The first captured `_l20_qk_norm_rope_kv_kernel` launch is a tiny launch:
`grid=(1,16,1)`, `block=(128,1,1)`, and `waves_per_sm=0.01`.

| Run | Duration | DRAM bytes | DRAM BW | DRAM peak | L2 hit | L1 hit | Active warps | Reg/thread | Tensor pipe | Long scoreboard |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qk-norm-rope-kv-sudo-first` | 4.352 us | 17,536 | 4.03 GB/s | 0.47% | 81.57% | 39.24% | 8.30% | 28 | 0.00% | 48.63% |
| `qk-norm-rope-kv-sudo-tokens64` | 4.384 us | 17,536 | 4.00 GB/s | 0.47% | 82.10% | 39.54% | 8.31% | 28 | 0.00% | 51.51% |

The `tokens64` file name reflects the intended launch-skip experiment, but the
Nsight launch metrics are still the same tiny `grid=(1,16,1)` launch with the
same DRAM byte count. Treat it as a repeat tiny-launch profile, not as 64-token
kernel evidence. A deterministic single-shape profiler is still required before
claiming 64-token or serving-shape counter behavior for this kernel.

## Interpretation

For the tiny launch, the kernel is not close to saturating L20 DRAM or SM
compute: DRAM utilization is below 1%, active warps are about 8%, and tensor
pipe utilization is zero. The useful conclusion is therefore narrow:

- this shape is launch/occupancy dominated, not bandwidth saturated;
- there is no register-pressure signal; the kernel uses 28 registers/thread;
- the next useful profiling step is a deterministic single-shape harness or a
  serving timeline with NVTX names and kernel counts.

Serving-level ITL claims must continue to use the checked-in vLLM benchmark
matrix, not these tiny-launch counters.
