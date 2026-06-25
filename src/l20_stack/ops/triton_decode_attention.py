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

    @triton.jit
    def _gqa_decode_attention_partial_kernel(
        query,
        key,
        value,
        partial_output,
        partial_max,
        partial_sum,
        q_stride_b,
        q_stride_h,
        k_stride_b,
        k_stride_t,
        k_stride_h,
        v_stride_b,
        v_stride_t,
        v_stride_h,
        context_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        split = tl.program_id(0)
        head_program = tl.program_id(1)
        batch = head_program // num_q_heads
        q_head = head_program % num_q_heads
        kv_head = q_head // (num_q_heads // num_kv_heads)
        dim = tl.arange(0, head_dim)
        query_values = tl.load(
            query + batch * q_stride_b + q_head * q_stride_h + dim
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = -float("inf")
        normalizer = 0.0
        accumulator = tl.zeros((head_dim,), tl.float32)
        split_start = split * SPLIT_SIZE

        for offset in range(0, SPLIT_SIZE, BLOCK_T):
            token = split_start + offset + tl.arange(0, BLOCK_T)
            token_mask = token < tl.minimum(split_start + SPLIT_SIZE, context_length)
            keys = tl.load(
                key
                + batch * k_stride_b
                + token[:, None] * k_stride_t
                + kv_head * k_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query_values[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max)
            values = tl.load(
                value
                + batch * v_stride_b
                + token[:, None] * v_stride_t
                + kv_head * v_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32)
            accumulator = (
                accumulator * old_scale
                + tl.sum(probabilities[:, None] * values, axis=0)
            )
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=0)
            max_score = next_max

        partial_index = head_program * tl.num_programs(0) + split
        tl.store(partial_output + partial_index * head_dim + dim, accumulator)
        tl.store(partial_max + partial_index, max_score)
        tl.store(partial_sum + partial_index, normalizer)

    @triton.jit
    def _gqa_decode_attention_fp8_partial_kernel(
        query,
        key,
        value,
        partial_output,
        partial_max,
        partial_sum,
        q_stride_b,
        q_stride_h,
        k_stride_b,
        k_stride_t,
        k_stride_h,
        v_stride_b,
        v_stride_t,
        v_stride_h,
        context_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        k_scale: tl.constexpr,
        v_scale: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
    ):
        split = tl.program_id(0)
        head_program = tl.program_id(1)
        batch = head_program // num_q_heads
        q_head = head_program % num_q_heads
        kv_head = q_head // (num_q_heads // num_kv_heads)
        dim = tl.arange(0, head_dim)
        query_values = tl.load(
            query + batch * q_stride_b + q_head * q_stride_h + dim
        ).to(tl.float32)
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = -float("inf")
        normalizer = 0.0
        accumulator = tl.zeros((head_dim,), tl.float32)
        split_start = split * SPLIT_SIZE

        for offset in range(0, SPLIT_SIZE, BLOCK_T):
            token = split_start + offset + tl.arange(0, BLOCK_T)
            token_mask = token < tl.minimum(split_start + SPLIT_SIZE, context_length)
            keys = tl.load(
                key
                + batch * k_stride_b
                + token[:, None] * k_stride_t
                + kv_head * k_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32) * k_scale
            scores = tl.sum(keys * query_values[None, :], axis=1) * scale
            scores = tl.where(token_mask, scores, -float("inf"))
            tile_max = tl.max(scores, axis=0)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max)
            values = tl.load(
                value
                + batch * v_stride_b
                + token[:, None] * v_stride_t
                + kv_head * v_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            ).to(tl.float32) * v_scale
            accumulator = (
                accumulator * old_scale
                + tl.sum(probabilities[:, None] * values, axis=0)
            )
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=0)
            max_score = next_max

        partial_index = head_program * tl.num_programs(0) + split
        tl.store(partial_output + partial_index * head_dim + dim, accumulator)
        tl.store(partial_max + partial_index, max_score)
        tl.store(partial_sum + partial_index, normalizer)

    @triton.jit
    def _gqa_decode_attention_reduce_kernel(
        partial_output,
        partial_max,
        partial_sum,
        output,
        out_stride_b,
        out_stride_h,
        num_q_heads: tl.constexpr,
        head_dim: tl.constexpr,
        NUM_SPLITS: tl.constexpr,
    ):
        head_program = tl.program_id(0)
        batch = head_program // num_q_heads
        q_head = head_program % num_q_heads
        dim = tl.arange(0, head_dim)
        splits = tl.arange(0, NUM_SPLITS)
        base = head_program * NUM_SPLITS
        maxima = tl.load(partial_max + base + splits)
        global_max = tl.max(maxima, axis=0)
        correction = tl.exp(maxima - global_max)
        sums = tl.load(partial_sum + base + splits)
        denominator = tl.sum(sums * correction, axis=0)
        partials = tl.load(
            partial_output
            + (base + splits[:, None]) * head_dim
            + dim[None, :]
        )
        numerator = tl.sum(partials * correction[:, None], axis=0)
        tl.store(
            output + batch * out_stride_b + q_head * out_stride_h + dim,
            numerator / denominator,
        )

    @triton.jit
    def _gqa_decode_attention_tc_partial_kernel(
        query,
        key,
        value,
        partial_output,
        partial_max,
        partial_sum,
        q_stride_b,
        q_stride_h,
        k_stride_b,
        k_stride_t,
        k_stride_h,
        v_stride_b,
        v_stride_t,
        v_stride_h,
        context_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        SPLIT_SIZE: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_Q: tl.constexpr,
    ):
        split = tl.program_id(0)
        kv_program = tl.program_id(1)
        q_group = tl.program_id(2)
        batch = kv_program // num_kv_heads
        kv_head = kv_program % num_kv_heads
        gqa_ratio: tl.constexpr = num_q_heads // num_kv_heads
        q_offsets = q_group * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < gqa_ratio
        q_heads = kv_head * gqa_ratio + q_offsets
        dim = tl.arange(0, head_dim)
        query_values = tl.load(
            query
            + batch * q_stride_b
            + q_heads[:, None] * q_stride_h
            + dim[None, :],
            mask=q_mask[:, None],
            other=0.0,
        )
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_Q,), tl.float32)
        accumulator = tl.zeros((BLOCK_Q, head_dim), tl.float32)
        split_start = split * SPLIT_SIZE

        for offset in range(0, SPLIT_SIZE, BLOCK_T):
            token = split_start + offset + tl.arange(0, BLOCK_T)
            token_mask = token < tl.minimum(split_start + SPLIT_SIZE, context_length)
            keys = tl.load(
                key
                + batch * k_stride_b
                + token[:, None] * k_stride_t
                + kv_head * k_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(query_values, tl.trans(keys)) * scale
            scores = tl.where(q_mask[:, None] & token_mask[None, :], scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max[:, None])
            values = tl.load(
                value
                + batch * v_stride_b
                + token[:, None] * v_stride_t
                + kv_head * v_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(values.dtype), values
            )
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=1)
            max_score = next_max

        partial_base = (batch * num_q_heads + q_heads) * tl.num_programs(0) + split
        tl.store(
            partial_output + partial_base[:, None] * head_dim + dim[None, :],
            accumulator,
            mask=q_mask[:, None],
        )
        tl.store(partial_max + partial_base, max_score, mask=q_mask)
        tl.store(partial_sum + partial_base, normalizer, mask=q_mask)


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


def gqa_decode_attention_split_kv(
    query,
    key,
    value,
    split_size: int = 512,
    block_t: int = 32,
    num_warps: int = 4,
):
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    batch, num_q_heads, head_dim = query.shape
    key_batch, context_length, num_kv_heads, key_dim = key.shape
    if (
        query.ndim != 3
        or key.ndim != 4
        or value.shape != key.shape
        or key_batch != batch
        or key_dim != head_dim
        or head_dim != 128
        or num_q_heads % num_kv_heads
    ):
        raise ValueError("requires compatible GQA tensors with head_dim=128")
    num_splits = triton.cdiv(context_length, split_size)
    if num_splits > 16:
        raise ValueError("split-KV path supports at most 16 splits")
    if block_t not in {16, 32, 64, 128}:
        raise ValueError("block_t must be one of 16, 32, 64, 128")
    if split_size % block_t:
        raise ValueError("split_size must be divisible by block_t")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, 8")
    partial_shape = (batch, num_q_heads, num_splits)
    partial_output = torch.empty(
        (*partial_shape, head_dim), device=query.device, dtype=torch.float32
    )
    partial_max = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    output = torch.empty_like(query)
    _gqa_decode_attention_partial_kernel[(num_splits, batch * num_q_heads)](
        query,
        key,
        value,
        partial_output,
        partial_max,
        partial_sum,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        context_length=context_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        SPLIT_SIZE=split_size,
        BLOCK_T=block_t,
        num_warps=num_warps,
        num_stages=1,
    )
    _gqa_decode_attention_reduce_kernel[(batch * num_q_heads,)](
        partial_output,
        partial_max,
        partial_sum,
        output,
        output.stride(0),
        output.stride(1),
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        NUM_SPLITS=num_splits,
        num_warps=4,
        num_stages=1,
    )
    return output


def gqa_decode_attention_split_kv_tensor_core_candidate(
    query,
    key,
    value,
    split_size: int = 512,
    block_t: int = 64,
    block_q: int = 2,
    num_warps: int = 4,
):
    """Experimental split-KV decode attention that groups Q heads for tl.dot.

    This path is intentionally not used by dispatch. It exists to measure
    whether grouped-Q Tensor-Core tiling is viable on L20 before replacing the
    scalar split-KV path.
    """
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    batch, context_length, num_q_heads, num_kv_heads, head_dim, num_splits = (
        _validate_split_kv_args(query, key, value, split_size, block_t, num_warps)
    )
    gqa_ratio = num_q_heads // num_kv_heads
    if block_q not in {1, 2, 4, 8, 16}:
        raise ValueError("block_q must be one of 1, 2, 4, 8, 16")
    if block_q > gqa_ratio:
        raise ValueError("block_q must not exceed the GQA ratio")
    q_groups = triton.cdiv(gqa_ratio, block_q)
    partial_shape = (batch, num_q_heads, num_splits)
    partial_output = torch.empty(
        (*partial_shape, head_dim), device=query.device, dtype=torch.float32
    )
    partial_max = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    output = torch.empty_like(query)
    _gqa_decode_attention_tc_partial_kernel[
        (num_splits, batch * num_kv_heads, q_groups)
    ](
        query,
        key,
        value,
        partial_output,
        partial_max,
        partial_sum,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        context_length=context_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        SPLIT_SIZE=split_size,
        BLOCK_T=block_t,
        BLOCK_Q=block_q,
        num_warps=num_warps,
        num_stages=1,
    )
    _gqa_decode_attention_reduce_kernel[(batch * num_q_heads,)](
        partial_output,
        partial_max,
        partial_sum,
        output,
        output.stride(0),
        output.stride(1),
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        NUM_SPLITS=num_splits,
        num_warps=4,
        num_stages=1,
    )
    return output


def _validate_split_kv_args(query, key, value, split_size, block_t, num_warps):
    batch, num_q_heads, head_dim = query.shape
    key_batch, context_length, num_kv_heads, key_dim = key.shape
    if (
        query.ndim != 3
        or key.ndim != 4
        or value.shape != key.shape
        or key_batch != batch
        or key_dim != head_dim
        or head_dim != 128
        or num_q_heads % num_kv_heads
    ):
        raise ValueError("requires compatible GQA tensors with head_dim=128")
    num_splits = triton.cdiv(context_length, split_size)
    if num_splits > 16:
        raise ValueError("split-KV path supports at most 16 splits")
    if block_t not in {16, 32, 64, 128}:
        raise ValueError("block_t must be one of 16, 32, 64, 128")
    if split_size % block_t:
        raise ValueError("split_size must be divisible by block_t")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, 8")
    return batch, context_length, num_q_heads, num_kv_heads, head_dim, num_splits


def gqa_decode_attention_fp8_split_kv(
    query,
    key,
    value,
    k_scale: float,
    v_scale: float,
    split_size: int = 512,
    block_t: int = 128,
    num_warps: int = 8,
):
    """Run split-KV decode attention with fused FP8 E4M3 KV dequantization.

    Shapes are query=[B,Hq,D], key/value=[B,T,Hkv,D]. The key/value tensors are
    expected to use a torch FP8 dtype; k_scale and v_scale are scalar dequant
    scales applied inside the attention kernel.
    """
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    batch, context_length, num_q_heads, num_kv_heads, head_dim, num_splits = (
        _validate_split_kv_args(query, key, value, split_size, block_t, num_warps)
    )
    if (
        key.dtype not in {torch.float8_e4m3fn, torch.float8_e5m2}
        or value.dtype != key.dtype
    ):
        raise ValueError("key/value must use a torch FP8 dtype")
    partial_shape = (batch, num_q_heads, num_splits)
    partial_output = torch.empty(
        (*partial_shape, head_dim), device=query.device, dtype=torch.float32
    )
    partial_max = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    output = torch.empty_like(query)
    _gqa_decode_attention_fp8_partial_kernel[(num_splits, batch * num_q_heads)](
        query,
        key,
        value,
        partial_output,
        partial_max,
        partial_sum,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        key.stride(2),
        value.stride(0),
        value.stride(1),
        value.stride(2),
        context_length=context_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        k_scale=float(k_scale),
        v_scale=float(v_scale),
        SPLIT_SIZE=split_size,
        BLOCK_T=block_t,
        num_warps=num_warps,
        num_stages=1,
    )
    _gqa_decode_attention_reduce_kernel[(batch * num_q_heads,)](
        partial_output,
        partial_max,
        partial_sum,
        output,
        output.stride(0),
        output.stride(1),
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        NUM_SPLITS=num_splits,
        num_warps=4,
        num_stages=1,
    )
    return output


def should_use_l20_gqa_decode_attention(batch: int, context_length: int) -> bool:
    """Conservative gate derived from L20 BF16 head_dim=128 measurements."""
    return batch >= 4 or context_length <= 1024


def should_use_l20_split_kv_attention(context_length: int) -> bool:
    return context_length >= 2048
