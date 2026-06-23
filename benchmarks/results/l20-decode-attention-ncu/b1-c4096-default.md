# Nsight Roofline Summary

| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | L1 hit % | SM % | Active warps % | Reg/thread | Long scoreboard % | Sector excess |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _gqa_decode_attention_partial_kernel | 2.52 | memory_bound | 281.13 | 32.60 | 10.92 | 0.00 | 14.21 | 11.92 | 56 | 38.75 | 1.00 |

Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred.
