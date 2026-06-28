# Nsight Roofline Summary

| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | L1 hit % | SM % | Tensor % | Active warps % | Reg/thread | Long scoreboard % | Sector excess |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _l20_qk_norm_rope_kv_kernel | 4.63 | memory_bound | 421.28 | 49.08 | 25.87 | 49.13 | 60.66 | 0.00 | 76.37 | 28 | 37.69 | 1.51 |

Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred.
