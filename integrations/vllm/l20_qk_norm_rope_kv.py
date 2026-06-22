"""L20 fused Q/K RMSNorm, NeoX RoPE, and paged KV-cache update."""

from __future__ import annotations

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _l20_qk_norm_rope_kv_kernel(
    qkv,
    positions,
    cos_sin_cache,
    q_weight,
    k_weight,
    slot_mapping,
    key_cache,
    value_cache,
    qkv_stride_t,
    kc_stride_b,
    kc_stride_s,
    kc_stride_h,
    vc_stride_b,
    vc_stride_s,
    vc_stride_h,
    cos_stride_t,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_dim: tl.constexpr,
    cache_block_size: tl.constexpr,
    eps: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    token = tl.program_id(0)
    head = tl.program_id(1)
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < head_dim
    half = head_dim // 2
    pair_mask = offsets < half
    position = tl.load(positions + token)
    cos = tl.load(
        cos_sin_cache + position * cos_stride_t + offsets,
        mask=pair_mask,
        other=1.0,
    ).to(tl.float32)
    sin = tl.load(
        cos_sin_cache + position * cos_stride_t + half + offsets,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32)

    token_base = token * qkv_stride_t
    q_base = token_base + head * head_dim
    q = tl.load(qkv + q_base + offsets, mask=mask, other=0.0).to(tl.float32)
    q_inv_rms = tl.rsqrt(tl.sum(q * q, axis=0) / head_dim + eps)
    weight = tl.load(q_weight + offsets, mask=mask, other=0.0).to(tl.float32)
    q_norm = q * q_inv_rms * weight
    q_left = tl.where(pair_mask, q_norm, 0.0)
    q_right = tl.where(pair_mask, tl.load(
        qkv + q_base + half + offsets,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32) * q_inv_rms * tl.load(
        q_weight + half + offsets,
        mask=pair_mask,
        other=0.0,
    ).to(tl.float32), 0.0)
    tl.store(qkv + q_base + offsets, q_left * cos - q_right * sin, mask=pair_mask)
    tl.store(
        qkv + q_base + half + offsets,
        q_right * cos + q_left * sin,
        mask=pair_mask,
    )

    if head < num_kv_heads:
        k_offset = num_q_heads * head_dim
        v_offset = (num_q_heads + num_kv_heads) * head_dim
        k_base = token_base + k_offset + head * head_dim
        k = tl.load(qkv + k_base + offsets, mask=mask, other=0.0).to(tl.float32)
        k_inv_rms = tl.rsqrt(tl.sum(k * k, axis=0) / head_dim + eps)
        k_weight_values = tl.load(
            k_weight + offsets, mask=mask, other=0.0
        ).to(tl.float32)
        k_norm = k * k_inv_rms * k_weight_values
        k_left = tl.where(pair_mask, k_norm, 0.0)
        k_right = tl.where(pair_mask, tl.load(
            qkv + k_base + half + offsets,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32) * k_inv_rms * tl.load(
            k_weight + half + offsets,
            mask=pair_mask,
            other=0.0,
        ).to(tl.float32), 0.0)
        k_left_out = k_left * cos - k_right * sin
        k_right_out = k_right * cos + k_left * sin
        tl.store(qkv + k_base + offsets, k_left_out, mask=pair_mask)
        tl.store(qkv + k_base + half + offsets, k_right_out, mask=pair_mask)

        slot = tl.load(slot_mapping + token)
        valid_slot = slot >= 0
        safe_slot = tl.where(valid_slot, slot, 0)
        physical_block = safe_slot // cache_block_size
        block_offset = safe_slot % cache_block_size
        k_cache_base = (
            physical_block * kc_stride_b
            + block_offset * kc_stride_s
            + head * kc_stride_h
        )
        v_cache_base = (
            physical_block * vc_stride_b
            + block_offset * vc_stride_s
            + head * vc_stride_h
        )
        value = tl.load(
            qkv + token_base + v_offset + head * head_dim + offsets,
            mask=mask,
            other=0.0,
        )
        tl.store(
            key_cache + k_cache_base + offsets,
            k_left_out,
            mask=pair_mask & valid_slot,
        )
        tl.store(
            key_cache + k_cache_base + half + offsets,
            k_right_out,
            mask=pair_mask & valid_slot,
        )
        tl.store(
            value_cache + v_cache_base + offsets,
            value,
            mask=mask & valid_slot,
        )


def l20_qk_norm_rope_and_cache(
    qkv: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    *,
    num_q_heads: int,
    num_kv_heads: int,
    eps: float,
) -> None:
    if torch.cuda.get_device_capability(qkv.device) != (8, 9):
        raise RuntimeError("requires an SM89 GPU")
    if qkv.dtype not in (torch.float16, torch.bfloat16):
        raise RuntimeError("supports FP16 and BF16")
    if qkv.ndim != 2 or not qkv.is_contiguous():
        raise RuntimeError("qkv must be contiguous [tokens, packed_heads]")
    head_dim = q_weight.numel()
    if head_dim != 128 or k_weight.numel() != head_dim:
        raise RuntimeError("the L20 QK norm fusion currently requires head_dim=128")
    expected = (num_q_heads + 2 * num_kv_heads) * head_dim
    if qkv.shape[1] != expected:
        raise RuntimeError("packed QKV width does not match the head configuration")
    if cos_sin_cache.shape[1] != head_dim:
        raise RuntimeError("only full-dimension NeoX RoPE is supported")
    _l20_qk_norm_rope_kv_kernel[(qkv.shape[0], num_q_heads)](
        qkv,
        positions,
        cos_sin_cache,
        q_weight,
        k_weight,
        slot_mapping,
        key_cache,
        value_cache,
        qkv.stride(0),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        cos_sin_cache.stride(0),
        num_q_heads,
        num_kv_heads,
        head_dim,
        key_cache.shape[1],
        eps,
        BLOCK_SIZE=128,
        num_warps=4 if qkv.shape[0] < 32 else 2,
        num_stages=1,
    )
