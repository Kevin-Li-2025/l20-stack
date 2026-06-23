# Nsight Roofline Summary

| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | L1 hit % | SM % | Active warps % | Reg/thread | Long scoreboard % | Sector excess |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _l20_neox_rope_kv_kernel | 2.07 | memory_bound | 509.62 | 59.13 | 33.65 | 25.21 | 20.83 | 30.17 | 32 | 77.73 | 1.03 |

Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred.
