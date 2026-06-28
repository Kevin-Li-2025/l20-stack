# Nsight Roofline Summary

| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | L1 hit % | SM % | Tensor % | Active warps % | Reg/thread | Long scoreboard % | Sector excess |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _l20_qk_norm_rope_kv_kernel | 4.78 | memory_bound | 125.21 | 14.67 | 8.49 | 47.30 | 18.81 | 0.00 | 32.42 | 28 | 43.68 | 1.47 |

Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred.
