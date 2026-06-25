# Nsight Roofline Summary

| Kernel | AI FLOP/B | Roofline | DRAM GB/s | DRAM % | L2 % | L1 hit % | SM % | Tensor % | Active warps % | Reg/thread | Long scoreboard % | Sector excess |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| _gqa_decode_attention_reduce_kernel | n/a | n/a | 11.60 | n/a | 0.61 | 16.67 | 1.60 | 0.00 | 8.32 | 31 | 2.05 | n/a |

Null values mean Nsight Compute did not emit that metric in the imported CSV; they are not inferred.
