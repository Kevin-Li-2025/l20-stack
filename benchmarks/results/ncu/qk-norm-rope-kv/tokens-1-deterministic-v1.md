# Nsight Roofline Summary

| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | L1 hit % | SM % | Tensor % | Active warps % | Reg/thread | Long scoreboard % | Sector excess |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _l20_qk_norm_rope_kv_kernel | 3.68 | memory_bound | 4.00 | 0.47 | 0.54 | 39.36 | 0.72 | 0.00 | 8.30 | 28 | 43.58 | 1.26 |

Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred.
