# Nsight Roofline Summary

| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | L1 hit % | SM % | Tensor % | Active warps % | Reg/thread | Long scoreboard % | Sector excess |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _gqa_decode_attention_reduce_kernel | n/a | n/a | 23.54 | n/a | 0.95 | 11.84 | 1.26 | 0.00 | 8.24 | 26 | 2.72 | n/a |

Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred.
