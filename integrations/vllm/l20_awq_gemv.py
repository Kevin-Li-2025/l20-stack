"""L20 AWQ W4A16 GEMV for small decode batches."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _l20_awq_gemv_kernel(
    x,
    qweight,
    scales,
    qzeros,
    output,
    x_stride_m,
    qw_stride_k,
    scales_stride_g,
    qzeros_stride_g,
    out_stride_m,
    K: tl.constexpr,
    N: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    output_index = tl.program_id(0)
    row = tl.program_id(1)
    column = output_index
    offsets = tl.arange(0, BLOCK_K)
    mask = offsets < K
    packed_column = column // 8
    nibble = column % 8
    # AWQ output packing order is [0, 4, 1, 5, 2, 6, 3, 7].
    shift = (nibble // 2 + (nibble % 2) * 4) * 4
    packed_weight = tl.load(
        qweight + offsets * qw_stride_k + packed_column,
        mask=mask,
        other=0,
    ).to(tl.int32)
    packed_zero = tl.load(
        qzeros
        + (offsets // GROUP_SIZE) * qzeros_stride_g
        + packed_column,
        mask=mask,
        other=0,
    ).to(tl.int32)
    quant = (packed_weight >> shift) & 0xF
    zero = (packed_zero >> shift) & 0xF
    scale = tl.load(
        scales
        + (offsets // GROUP_SIZE) * scales_stride_g
        + column,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    activation = tl.load(
        x + row * x_stride_m + offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    result = tl.sum((quant - zero).to(tl.float32) * scale * activation, axis=0)
    tl.store(output + row * out_stride_m + column, result)


def l20_awq_gemv(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if torch.cuda.get_device_capability(x.device) != (8, 9):
        raise RuntimeError("requires an SM89 GPU")
    if x.ndim != 2 or x.shape[0] > 8:
        raise RuntimeError("requires flattened decode input [tokens,K] with tokens <= 8")
    if group_size != 128 or x.shape[1] % group_size:
        raise RuntimeError("requires AWQ group size 128")
    k = x.shape[1]
    n = qweight.shape[1] * 8
    if qweight.shape[0] != k:
        raise RuntimeError("AWQ qweight shape does not match input")
    output = torch.empty((x.shape[0], n), device=x.device, dtype=x.dtype)
    block_k = triton.next_power_of_2(k)
    _l20_awq_gemv_kernel[(n, x.shape[0])](
        x,
        qweight,
        scales,
        qzeros,
        output,
        x.stride(0),
        qweight.stride(0),
        scales.stride(0),
        qzeros.stride(0),
        output.stride(0),
        K=k,
        N=n,
        GROUP_SIZE=group_size,
        BLOCK_K=block_k,
        num_warps=8 if block_k >= 4096 else 4,
        num_stages=1,
    )
    return output


def should_use_l20_awq_gemv(x: torch.Tensor, group_size: int) -> bool:
    # The scalar-output implementation is a correctness/reference path.
    # It does not beat vLLM awq_gemm broadly enough for production dispatch.
    del x, group_size
    return False
