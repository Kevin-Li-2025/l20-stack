"""L20-oriented groupwise INT4 decode GEMV."""

from __future__ import annotations

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


if triton is not None:  # pragma: no cover - requires CUDA

    @triton.jit
    def _int4_groupwise_gemv_kernel(
        x,
        packed_weight,
        scales,
        output,
        K: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_K)
        mask = offsets < K
        packed_offsets = offsets // 2
        packed = tl.load(
            packed_weight + row * (K // 2) + packed_offsets,
            mask=mask,
            other=0,
        ).to(tl.int32)
        shift = (offsets & 1) * 4
        quant = ((packed >> shift) & 0xF) - 8
        scale = tl.load(
            scales + row * (K // GROUP_SIZE) + offsets // GROUP_SIZE,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        values = quant.to(tl.float32) * scale
        activations = tl.load(x + offsets, mask=mask, other=0.0).to(tl.float32)
        result = tl.sum(values * activations, axis=0)
        tl.store(output + row, result)


def int4_groupwise_gemv(x, packed_weight, scales, group_size: int = 128):
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    if x.ndim != 1 or packed_weight.ndim != 2 or scales.ndim != 2:
        raise ValueError("expected x=[K], packed_weight=[N,K/2], scales=[N,K/group]")
    n, packed_k = packed_weight.shape
    k = x.numel()
    if packed_k * 2 != k or k % group_size:
        raise ValueError("incompatible INT4 dimensions")
    if scales.shape != (n, k // group_size):
        raise ValueError("scale shape does not match group size")
    block_k = triton.next_power_of_2(k)
    if block_k > 16384:
        raise ValueError("K above 16384 is not supported")
    output = torch.empty(n, device=x.device, dtype=x.dtype)
    _int4_groupwise_gemv_kernel[(n,)](
        x,
        packed_weight,
        scales,
        output,
        K=k,
        GROUP_SIZE=group_size,
        BLOCK_K=block_k,
        num_warps=8 if block_k >= 4096 else 4,
        num_stages=1,
    )
    return output


def dequantize_int4_reference(packed_weight, scales, group_size: int = 128):
    if torch is None:
        raise RuntimeError("requires PyTorch")
    low = (packed_weight & 0xF).to(torch.int8) - 8
    high = ((packed_weight >> 4) & 0xF).to(torch.int8) - 8
    quant = torch.stack((low, high), dim=-1).flatten(-2).float()
    expanded_scales = scales.repeat_interleave(group_size, dim=1).float()
    return quant * expanded_scales
