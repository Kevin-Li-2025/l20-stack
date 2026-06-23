# Nsight Roofline Summary

| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | L1 hit % | SM % | Active warps % | Reg/thread | Long scoreboard % | Sector excess |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _gqa_decode_attention_partial_kernel | 2.37 | memory_bound | 330.85 | 38.37 | 12.81 | 0.34 | 11.62 | 16.66 | 64 | 48.61 | 1.00 |

Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred.
