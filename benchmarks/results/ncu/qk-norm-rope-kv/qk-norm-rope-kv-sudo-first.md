# Nsight Roofline Summary

| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | L1 hit % | SM % | Tensor % | Active warps % | Reg/thread | Long scoreboard % | Sector excess |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _l20_qk_norm_rope_kv_kernel | 3.68 | memory_bound | 4.03 | 0.47 | 0.55 | 39.24 | 0.72 | 0.00 | 8.30 | 28 | 48.63 | 1.25 |

Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred.
