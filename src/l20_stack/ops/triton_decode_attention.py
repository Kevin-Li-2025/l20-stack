"""L20-oriented contiguous GQA decode attention."""

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
    def _gqa_decode_attention_kernel(
        query,
        key,
        value,
        output,
        q_stride_b,
        q_stride_h,
        k_stride_b,
        k_stride_t,
        k_stride_h,
        v_stride_b,
        v_stride_t,
        v_stride_h,
        o_stride_b,
        o_stride_h,
        context_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        program = tl.program_id(0)
        batch = program // num_q_heads
        q_head = program % num_q_heads
        kv_head = q_head // (num_q_heads // num_kv_heads)
        dim = tl.arange(0, head_dim)
        query_values = tl.load(
            query + batch * q_stride_b + q_head * q_stride_h + dim
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = -float("inf")
        normalizer = 0.0
        accumulator = tl.zeros((head_dim,), tl.float32)

        for start in range(0, context_length, BLOCK_T):
            token = start + tl.arange(0, BLOCK_T)
            token_mask = token < context_length
            key_offsets = (
                batch * k_stride_b
                + token[:, None] * k_stride_t
                + kv_head * k_stride_h
                + dim[None, :]
            )
            keys = tl.load(
                key + key_offsets,
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query_values[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max)
            tile_sum = tl.sum(probabilities, axis=0)
            value_offsets = (
                batch * v_stride_b
                + token[:, None] * v_stride_t
                + kv_head * v_stride_h
                + dim[None, :]
            )
            values = tl.load(
                value + value_offsets,
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator = (
                accumulator * old_scale
                + tl.sum(probabilities[:, None] * values, axis=0)
            )
            normalizer = normalizer * old_scale + tile_sum
            max_score = next_max

        tl.store(
            output + batch * o_stride_b + q_head * o_stride_h + dim,
            accumulator / normalizer,
        )


def gqa_decode_attention(query, key, value):
    """Run single-token contiguous-cache GQA attention.

    Shapes are query=[B,Hq,D], key/value=[B,T,Hkv,D].
    """
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    if query.ndim != 3 or key.ndim != 4 or value.shape != key.shape:
        raise ValueError("expected query=[B,Hq,D], key/value=[B,T,Hkv,D]")
    batch, num_q_heads, head_dim = query.shape
    key_batch, context_length, num_kv_heads, key_dim = key.shape
    if key_batch != batch or key_dim != head_dim:
        raise ValueError("query and KV dimensions do not match")
    if head_dim != 128 or num_q_heads % num_kv_heads:
        raise ValueError("requires head_dim=128 and an integral GQA ratio")
    output = torch.empty_like(query)
    block_t = 32 if context_length <= 1024 else 16
    _gqa_decode_attention_kernel[(batch * num_q_heads,)](
        query,
        key,
        value,
        output,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        output.stride(0),
        output.stride(1),
        context_length=context_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_T=block_t,
        num_warps=4,
        num_stages=1,
    )
    return output


def should_use_l20_gqa_decode_attention(batch: int, context_length: int) -> bool:
    """Conservative gate derived from L20 BF16 head_dim=128 measurements."""
    return batch >= 2 or context_length <= 512
