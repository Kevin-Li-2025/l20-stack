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


def _next_power_of_2(value: int) -> int:
    return 1 << (value - 1).bit_length()


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
        BLOCK_SPLITS: tl.constexpr,
    ):
        head_program = tl.program_id(0)
        batch = head_program // num_q_heads
        q_head = head_program % num_q_heads
        dim = tl.arange(0, head_dim)
        splits = tl.arange(0, BLOCK_SPLITS)
        split_mask = splits < NUM_SPLITS
        base = head_program * NUM_SPLITS
        maxima = tl.load(partial_max + base + splits, mask=split_mask, other=-float("inf"))
        global_max = tl.max(maxima, axis=0)
        correction = tl.exp(maxima - global_max)
        sums = tl.load(partial_sum + base + splits, mask=split_mask, other=0.0)
        denominator = tl.sum(sums * correction, axis=0)
        partials = tl.load(
            partial_output
            + (base + splits[:, None]) * head_dim
            + dim[None, :],
            mask=split_mask[:, None],
            other=0.0,
        ).to(tl.float32)
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

    @triton.jit
    def _gqa_decode_attention_tc_dsplit_partial_kernel(
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
        BLOCK_D: tl.constexpr,
    ):
        split = tl.program_id(0)
        kv_program = tl.program_id(1)
        qd_program = tl.program_id(2)
        batch = kv_program // num_kv_heads
        kv_head = kv_program % num_kv_heads
        gqa_ratio: tl.constexpr = num_q_heads // num_kv_heads
        d_blocks: tl.constexpr = tl.cdiv(head_dim, BLOCK_D)
        q_group = qd_program // d_blocks
        d_block = qd_program % d_blocks
        q_offsets = q_group * BLOCK_Q + tl.arange(0, BLOCK_Q)
        q_mask = q_offsets < gqa_ratio
        q_heads = kv_head * gqa_ratio + q_offsets
        score_dim = tl.arange(0, head_dim)
        out_dim = d_block * BLOCK_D + tl.arange(0, BLOCK_D)
        out_mask = out_dim < head_dim
        query_values = tl.load(
            query
            + batch * q_stride_b
            + q_heads[:, None] * q_stride_h
            + score_dim[None, :],
            mask=q_mask[:, None],
            other=0.0,
        )
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = tl.full((BLOCK_Q,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_Q,), tl.float32)
        accumulator = tl.zeros((BLOCK_Q, BLOCK_D), tl.float32)
        split_start = split * SPLIT_SIZE

        for offset in range(0, SPLIT_SIZE, BLOCK_T):
            token = split_start + offset + tl.arange(0, BLOCK_T)
            token_mask = token < tl.minimum(split_start + SPLIT_SIZE, context_length)
            keys = tl.load(
                key
                + batch * k_stride_b
                + token[:, None] * k_stride_t
                + kv_head * k_stride_h
                + score_dim[None, :],
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
                + out_dim[None, :],
                mask=token_mask[:, None] & out_mask[None, :],
                other=0.0,
            )
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(values.dtype), values
            )
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=1)
            max_score = next_max

        partial_base = (batch * num_q_heads + q_heads) * tl.num_programs(0) + split
        tl.store(
            partial_output + partial_base[:, None] * head_dim + out_dim[None, :],
            accumulator,
            mask=q_mask[:, None] & out_mask[None, :],
        )
        if d_block == 0:
            tl.store(partial_max + partial_base, max_score, mask=q_mask)
            tl.store(partial_sum + partial_base, normalizer, mask=q_mask)

    @triton.jit
    def _shared_prefix_gqa_attention_kernel(
        query,
        key,
        value,
        output,
        q_stride_b,
        q_stride_h,
        k_stride_t,
        k_stride_h,
        v_stride_t,
        v_stride_h,
        o_stride_b,
        o_stride_h,
        batch_size: tl.constexpr,
        prefix_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        kv_head = tl.program_id(0)
        q_block = tl.program_id(1)
        gqa_ratio: tl.constexpr = num_q_heads // num_kv_heads
        row = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row < (batch_size * gqa_ratio)
        batch = row // gqa_ratio
        q_offset = row - batch * gqa_ratio
        q_head = kv_head * gqa_ratio + q_offset
        dim = tl.arange(0, head_dim)
        query_values = tl.load(
            query
            + batch[:, None] * q_stride_b
            + q_head[:, None] * q_stride_h
            + dim[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, head_dim), tl.float32)

        for start in range(0, prefix_length, BLOCK_T):
            token = start + tl.arange(0, BLOCK_T)
            token_mask = token < prefix_length
            keys = tl.load(
                key
                + token[:, None] * k_stride_t
                + kv_head * k_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(query_values, tl.trans(keys)) * scale
            scores = tl.where(row_mask[:, None] & token_mask[None, :], scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max[:, None])
            values = tl.load(
                value
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

        tl.store(
            output
            + batch[:, None] * o_stride_b
            + q_head[:, None] * o_stride_h
            + dim[None, :],
            accumulator / normalizer[:, None],
            mask=row_mask[:, None],
        )

    @triton.jit
    def _shared_prefix_gqa_partial_kernel(
        query,
        key,
        value,
        prefix_output,
        prefix_max,
        prefix_sum,
        q_stride_b,
        q_stride_h,
        k_stride_t,
        k_stride_h,
        v_stride_t,
        v_stride_h,
        batch_size: tl.constexpr,
        prefix_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        kv_head = tl.program_id(0)
        q_block = tl.program_id(1)
        gqa_ratio: tl.constexpr = num_q_heads // num_kv_heads
        row = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row < (batch_size * gqa_ratio)
        batch = row // gqa_ratio
        q_offset = row - batch * gqa_ratio
        q_head = kv_head * gqa_ratio + q_offset
        dim = tl.arange(0, head_dim)
        query_values = tl.load(
            query
            + batch[:, None] * q_stride_b
            + q_head[:, None] * q_stride_h
            + dim[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, head_dim), tl.float32)

        for start in range(0, prefix_length, BLOCK_T):
            token = start + tl.arange(0, BLOCK_T)
            token_mask = token < prefix_length
            keys = tl.load(
                key
                + token[:, None] * k_stride_t
                + kv_head * k_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(query_values, tl.trans(keys)) * scale
            scores = tl.where(row_mask[:, None] & token_mask[None, :], scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max[:, None])
            values = tl.load(
                value
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

        partial_index = batch * num_q_heads + q_head
        tl.store(
            prefix_output + partial_index[:, None] * head_dim + dim[None, :],
            accumulator,
            mask=row_mask[:, None],
        )
        tl.store(prefix_max + partial_index, max_score, mask=row_mask)
        tl.store(prefix_sum + partial_index, normalizer, mask=row_mask)

    @triton.jit
    def _prefix_suffix_gqa_reduce_kernel(
        prefix_output,
        prefix_max,
        prefix_sum,
        suffix_output,
        suffix_max,
        suffix_sum,
        output,
        out_stride_b,
        out_stride_h,
        num_q_heads: tl.constexpr,
        head_dim: tl.constexpr,
        NUM_SUFFIX_SPLITS: tl.constexpr,
        BLOCK_SUFFIX_SPLITS: tl.constexpr,
    ):
        head_program = tl.program_id(0)
        batch = head_program // num_q_heads
        q_head = head_program % num_q_heads
        dim = tl.arange(0, head_dim)
        splits = tl.arange(0, BLOCK_SUFFIX_SPLITS)
        split_mask = splits < NUM_SUFFIX_SPLITS
        prefix_m = tl.load(prefix_max + head_program)
        suffix_base = head_program * NUM_SUFFIX_SPLITS
        suffix_m = tl.load(
            suffix_max + suffix_base + splits,
            mask=split_mask,
            other=-float("inf"),
        )
        suffix_global_m = tl.max(suffix_m, axis=0)
        global_m = tl.maximum(prefix_m, suffix_global_m)
        prefix_l = tl.load(prefix_sum + head_program)
        suffix_l = tl.load(suffix_sum + suffix_base + splits, mask=split_mask, other=0.0)
        prefix_scale = tl.exp(prefix_m - global_m)
        suffix_scale = tl.exp(suffix_m - global_m)
        denominator = prefix_l * prefix_scale + tl.sum(suffix_l * suffix_scale, axis=0)
        prefix_acc = tl.load(prefix_output + head_program * head_dim + dim)
        suffix_acc = tl.load(
            suffix_output
            + (suffix_base + splits[:, None]) * head_dim
            + dim[None, :],
            mask=split_mask[:, None],
            other=0.0,
        )
        numerator = prefix_acc * prefix_scale + tl.sum(
            suffix_acc * suffix_scale[:, None], axis=0
        )
        tl.store(
            output + batch * out_stride_b + q_head * out_stride_h + dim,
            numerator / denominator,
        )

    @triton.jit
    def _shared_paged_prefix_gqa_partial_kernel(
        query,
        key_cache,
        value_cache,
        block_table,
        prefix_output,
        prefix_max,
        prefix_sum,
        q_stride_b,
        q_stride_h,
        kc_stride_p,
        kc_stride_t,
        kc_stride_h,
        vc_stride_p,
        vc_stride_t,
        vc_stride_h,
        batch_size: tl.constexpr,
        prefix_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        BLOCK_T: tl.constexpr,
        BLOCK_M: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
    ):
        kv_head = tl.program_id(0)
        q_block = tl.program_id(1)
        gqa_ratio: tl.constexpr = num_q_heads // num_kv_heads
        row = q_block * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = row < (batch_size * gqa_ratio)
        batch = row // gqa_ratio
        q_offset = row - batch * gqa_ratio
        q_head = kv_head * gqa_ratio + q_offset
        dim = tl.arange(0, head_dim)
        query_values = tl.load(
            query
            + batch[:, None] * q_stride_b
            + q_head[:, None] * q_stride_h
            + dim[None, :],
            mask=row_mask[:, None],
            other=0.0,
        )
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        normalizer = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, head_dim), tl.float32)

        for start in range(0, prefix_length, BLOCK_T):
            tile_offset = tl.arange(0, BLOCK_T)
            token = start + tile_offset
            token_mask = token < prefix_length
            page_index = tl.arange(0, BLOCK_T // PAGE_SIZE)
            logical_page_base = start // PAGE_SIZE
            page_mask = logical_page_base + page_index < (
                prefix_length + PAGE_SIZE - 1
            ) // PAGE_SIZE
            tile_pages = tl.load(
                block_table + logical_page_base + page_index,
                mask=page_mask,
                other=0,
            )
            page_slot = tile_offset // PAGE_SIZE
            physical_page = tl.sum(
                tl.where(
                    page_slot[:, None] == page_index[None, :],
                    tile_pages[None, :],
                    0,
                ),
                axis=1,
            )
            page_offset = token % PAGE_SIZE
            keys = tl.load(
                key_cache
                + physical_page[:, None] * kc_stride_p
                + page_offset[:, None] * kc_stride_t
                + kv_head * kc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            scores = tl.dot(query_values, tl.trans(keys)) * scale
            scores = tl.where(row_mask[:, None] & token_mask[None, :], scores, -float("inf"))
            tile_max = tl.max(scores, axis=1)
            next_max = tl.maximum(max_score, tile_max)
            old_scale = tl.exp(max_score - next_max)
            probabilities = tl.exp(scores - next_max[:, None])
            values = tl.load(
                value_cache
                + physical_page[:, None] * vc_stride_p
                + page_offset[:, None] * vc_stride_t
                + kv_head * vc_stride_h
                + dim[None, :],
                mask=token_mask[:, None],
                other=0.0,
            )
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities.to(values.dtype), values
            )
            normalizer = normalizer * old_scale + tl.sum(probabilities, axis=1)
            max_score = next_max

        partial_index = batch * num_q_heads + q_head
        tl.store(
            prefix_output + partial_index[:, None] * head_dim + dim[None, :],
            accumulator,
            mask=row_mask[:, None],
        )
        tl.store(prefix_max + partial_index, max_score, mask=row_mask)
        tl.store(prefix_sum + partial_index, normalizer, mask=row_mask)

    @triton.jit
    def _paged_suffix_gqa_partial_kernel(
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        suffix_output,
        suffix_max,
        suffix_sum,
        q_stride_b,
        q_stride_h,
        kc_stride_p,
        kc_stride_t,
        kc_stride_h,
        vc_stride_p,
        vc_stride_t,
        vc_stride_h,
        bt_stride_b,
        prefix_length: tl.constexpr,
        num_q_heads: tl.constexpr,
        num_kv_heads: tl.constexpr,
        head_dim: tl.constexpr,
        PAGE_SIZE: tl.constexpr,
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
        seq_len = tl.load(seq_lens + batch)
        scale = 1.0 / tl.sqrt(float(head_dim))
        max_score = -float("inf")
        normalizer = 0.0
        accumulator = tl.zeros((head_dim,), tl.float32)
        split_start = prefix_length + split * SPLIT_SIZE

        for offset in range(0, SPLIT_SIZE, BLOCK_T):
            token = split_start + offset + tl.arange(0, BLOCK_T)
            token_mask = token < tl.minimum(split_start + SPLIT_SIZE, seq_len)
            logical_page = token // PAGE_SIZE
            page_offset = token % PAGE_SIZE
            physical_page = tl.load(
                block_tables + batch * bt_stride_b + tl.where(token_mask, logical_page, 0),
                mask=token_mask,
                other=0,
            )
            keys = tl.load(
                key_cache
                + physical_page[:, None] * kc_stride_p
                + page_offset[:, None] * kc_stride_t
                + kv_head * kc_stride_h
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
                value_cache
                + physical_page[:, None] * vc_stride_p
                + page_offset[:, None] * vc_stride_t
                + kv_head * vc_stride_h
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
        valid_split = split_start < seq_len
        tl.store(
            suffix_output + partial_index * head_dim + dim,
            tl.where(valid_split, accumulator, 0.0),
        )
        tl.store(
            suffix_max + partial_index,
            tl.where(valid_split, max_score, -float("inf")),
        )
        tl.store(
            suffix_sum + partial_index,
            tl.where(valid_split, normalizer, 0.0),
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


def shared_prefix_gqa_decode_attention(
    query,
    key,
    value,
    block_t: int = 128,
    block_m: int = 4,
    num_warps: int = 4,
):
    """Run decode attention for many queries sharing one contiguous KV prefix.

    Shapes are query=[B,Hq,D], key/value=[T,Hkv,D]. This is an experimental
    prefix-aware candidate; it only computes the shared-prefix attention output
    and does not include per-request suffix merging.
    """
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    if query.ndim != 3 or key.ndim != 3 or value.shape != key.shape:
        raise ValueError("expected query=[B,Hq,D], key/value=[T,Hkv,D]")
    batch, num_q_heads, head_dim = query.shape
    prefix_length, num_kv_heads, key_dim = key.shape
    if key_dim != head_dim or head_dim != 128 or num_q_heads % num_kv_heads:
        raise ValueError("requires compatible GQA tensors with head_dim=128")
    if block_t not in {32, 64, 128}:
        raise ValueError("block_t must be one of 32, 64, 128")
    if block_m not in {1, 2, 4, 8}:
        raise ValueError("block_m must be one of 1, 2, 4, 8")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, 8")
    output = torch.empty_like(query)
    rows_per_kv = batch * (num_q_heads // num_kv_heads)
    _shared_prefix_gqa_attention_kernel[
        (num_kv_heads, triton.cdiv(rows_per_kv, block_m))
    ](
        query,
        key,
        value,
        output,
        query.stride(0),
        query.stride(1),
        key.stride(0),
        key.stride(1),
        value.stride(0),
        value.stride(1),
        output.stride(0),
        output.stride(1),
        batch_size=batch,
        prefix_length=prefix_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_T=block_t,
        BLOCK_M=block_m,
        num_warps=num_warps,
        num_stages=1,
    )
    return output


def shared_prefix_suffix_gqa_decode_attention(
    query,
    prefix_key,
    prefix_value,
    suffix_key,
    suffix_value,
    prefix_block_t: int = 128,
    prefix_block_m: int = 4,
    suffix_split_size: int = 512,
    suffix_block_t: int = 128,
    num_warps: int = 4,
):
    """Run decode attention over a shared prefix plus per-request suffix.

    Shapes are query=[B,Hq,D], prefix_key/value=[P,Hkv,D], and
    suffix_key/value=[B,S,Hkv,D]. The implementation packs the shared-prefix
    computation across requests, computes suffix partials per request, then
    merges both regions with online-softmax statistics.
    """
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    if (
        query.ndim != 3
        or prefix_key.ndim != 3
        or suffix_key.ndim != 4
        or prefix_value.shape != prefix_key.shape
        or suffix_value.shape != suffix_key.shape
    ):
        raise ValueError(
            "expected query=[B,Hq,D], prefix=[P,Hkv,D], suffix=[B,S,Hkv,D]"
        )
    batch, num_q_heads, head_dim = query.shape
    prefix_length, num_kv_heads, prefix_dim = prefix_key.shape
    suffix_batch, suffix_length, suffix_heads, suffix_dim = suffix_key.shape
    if (
        suffix_batch != batch
        or suffix_heads != num_kv_heads
        or prefix_dim != head_dim
        or suffix_dim != head_dim
        or head_dim != 128
        or num_q_heads % num_kv_heads
    ):
        raise ValueError("requires compatible GQA tensors with head_dim=128")
    if prefix_block_t not in {32, 64, 128}:
        raise ValueError("prefix_block_t must be one of 32, 64, 128")
    if prefix_block_m not in {1, 2, 4, 8}:
        raise ValueError("prefix_block_m must be one of 1, 2, 4, 8")
    if suffix_block_t not in {16, 32, 64, 128}:
        raise ValueError("suffix_block_t must be one of 16, 32, 64, 128")
    if suffix_split_size % suffix_block_t:
        raise ValueError("suffix_split_size must be divisible by suffix_block_t")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, 8")
    num_suffix_splits = triton.cdiv(suffix_length, suffix_split_size)
    if num_suffix_splits < 1 or num_suffix_splits > 16:
        raise ValueError("suffix path supports 1 to 16 splits")

    prefix_shape = (batch, num_q_heads)
    prefix_output = torch.empty(
        (*prefix_shape, head_dim), device=query.device, dtype=torch.float32
    )
    prefix_max = torch.empty(prefix_shape, device=query.device, dtype=torch.float32)
    prefix_sum = torch.empty_like(prefix_max)
    suffix_shape = (batch, num_q_heads, num_suffix_splits)
    suffix_output = torch.empty(
        (*suffix_shape, head_dim), device=query.device, dtype=torch.float32
    )
    suffix_max = torch.empty(suffix_shape, device=query.device, dtype=torch.float32)
    suffix_sum = torch.empty_like(suffix_max)
    output = torch.empty_like(query)

    rows_per_kv = batch * (num_q_heads // num_kv_heads)
    _shared_prefix_gqa_partial_kernel[
        (num_kv_heads, triton.cdiv(rows_per_kv, prefix_block_m))
    ](
        query,
        prefix_key,
        prefix_value,
        prefix_output,
        prefix_max,
        prefix_sum,
        query.stride(0),
        query.stride(1),
        prefix_key.stride(0),
        prefix_key.stride(1),
        prefix_value.stride(0),
        prefix_value.stride(1),
        batch_size=batch,
        prefix_length=prefix_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_T=prefix_block_t,
        BLOCK_M=prefix_block_m,
        num_warps=num_warps,
        num_stages=1,
    )
    _gqa_decode_attention_partial_kernel[(num_suffix_splits, batch * num_q_heads)](
        query,
        suffix_key,
        suffix_value,
        suffix_output,
        suffix_max,
        suffix_sum,
        query.stride(0),
        query.stride(1),
        suffix_key.stride(0),
        suffix_key.stride(1),
        suffix_key.stride(2),
        suffix_value.stride(0),
        suffix_value.stride(1),
        suffix_value.stride(2),
        context_length=suffix_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        SPLIT_SIZE=suffix_split_size,
        BLOCK_T=suffix_block_t,
        num_warps=num_warps,
        num_stages=1,
    )
    _prefix_suffix_gqa_reduce_kernel[(batch * num_q_heads,)](
        prefix_output,
        prefix_max,
        prefix_sum,
        suffix_output,
        suffix_max,
        suffix_sum,
        output,
        output.stride(0),
        output.stride(1),
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        NUM_SUFFIX_SPLITS=num_suffix_splits,
        BLOCK_SUFFIX_SPLITS=_next_power_of_2(num_suffix_splits),
        num_warps=4,
        num_stages=1,
    )
    return output


def shared_paged_prefix_suffix_gqa_decode_attention(
    query,
    key_cache,
    value_cache,
    block_table,
    suffix_key,
    suffix_value,
    prefix_length: int,
    page_size: int = 16,
    prefix_block_t: int = 128,
    prefix_block_m: int = 4,
    suffix_split_size: int = 512,
    suffix_block_t: int = 128,
    num_warps: int = 4,
):
    """Run shared-prefix decode attention from a paged NHD KV cache.

    Shapes are query=[B,Hq,D], key/value_cache=[pages,page_size,Hkv,D],
    block_table=[prefix_pages], and suffix_key/value=[B,S,Hkv,D]. All requests
    are assumed to share the same prefix block chain.
    """
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    if (
        query.ndim != 3
        or key_cache.ndim != 4
        or value_cache.shape != key_cache.shape
        or block_table.ndim != 1
        or suffix_key.ndim != 4
        or suffix_value.shape != suffix_key.shape
    ):
        raise ValueError(
            "expected query=[B,Hq,D], cache=[pages,page,Hkv,D], "
            "block_table=[prefix_pages], suffix=[B,S,Hkv,D]"
        )
    batch, num_q_heads, head_dim = query.shape
    _, cache_page_size, num_kv_heads, cache_dim = key_cache.shape
    suffix_batch, suffix_length, suffix_heads, suffix_dim = suffix_key.shape
    if (
        suffix_batch != batch
        or suffix_heads != num_kv_heads
        or cache_dim != head_dim
        or suffix_dim != head_dim
        or head_dim != 128
        or num_q_heads % num_kv_heads
        or cache_page_size != page_size
        or page_size != 16
    ):
        raise ValueError("requires compatible page-16 GQA tensors with head_dim=128")
    if block_table.shape[0] * page_size < prefix_length:
        raise ValueError("block_table does not cover prefix_length")
    if prefix_block_t not in {32, 64, 128} or prefix_block_t % page_size:
        raise ValueError("prefix_block_t must be 32, 64, or 128 and divisible by page_size")
    if prefix_block_m not in {1, 2, 4, 8}:
        raise ValueError("prefix_block_m must be one of 1, 2, 4, 8")
    if suffix_block_t not in {16, 32, 64, 128}:
        raise ValueError("suffix_block_t must be one of 16, 32, 64, 128")
    if suffix_split_size % suffix_block_t:
        raise ValueError("suffix_split_size must be divisible by suffix_block_t")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, 8")
    num_suffix_splits = triton.cdiv(suffix_length, suffix_split_size)
    if num_suffix_splits < 1 or num_suffix_splits > 16:
        raise ValueError("suffix path supports 1 to 16 splits")

    prefix_shape = (batch, num_q_heads)
    prefix_output = torch.empty(
        (*prefix_shape, head_dim), device=query.device, dtype=torch.float32
    )
    prefix_max = torch.empty(prefix_shape, device=query.device, dtype=torch.float32)
    prefix_sum = torch.empty_like(prefix_max)
    suffix_shape = (batch, num_q_heads, num_suffix_splits)
    suffix_output = torch.empty(
        (*suffix_shape, head_dim), device=query.device, dtype=torch.float32
    )
    suffix_max = torch.empty(suffix_shape, device=query.device, dtype=torch.float32)
    suffix_sum = torch.empty_like(suffix_max)
    output = torch.empty_like(query)

    rows_per_kv = batch * (num_q_heads // num_kv_heads)
    _shared_paged_prefix_gqa_partial_kernel[
        (num_kv_heads, triton.cdiv(rows_per_kv, prefix_block_m))
    ](
        query,
        key_cache,
        value_cache,
        block_table,
        prefix_output,
        prefix_max,
        prefix_sum,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        batch_size=batch,
        prefix_length=prefix_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_T=prefix_block_t,
        BLOCK_M=prefix_block_m,
        PAGE_SIZE=page_size,
        num_warps=num_warps,
        num_stages=1,
    )
    _gqa_decode_attention_partial_kernel[(num_suffix_splits, batch * num_q_heads)](
        query,
        suffix_key,
        suffix_value,
        suffix_output,
        suffix_max,
        suffix_sum,
        query.stride(0),
        query.stride(1),
        suffix_key.stride(0),
        suffix_key.stride(1),
        suffix_key.stride(2),
        suffix_value.stride(0),
        suffix_value.stride(1),
        suffix_value.stride(2),
        context_length=suffix_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        SPLIT_SIZE=suffix_split_size,
        BLOCK_T=suffix_block_t,
        num_warps=num_warps,
        num_stages=1,
    )
    _prefix_suffix_gqa_reduce_kernel[(batch * num_q_heads,)](
        prefix_output,
        prefix_max,
        prefix_sum,
        suffix_output,
        suffix_max,
        suffix_sum,
        output,
        output.stride(0),
        output.stride(1),
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        NUM_SUFFIX_SPLITS=num_suffix_splits,
        BLOCK_SUFFIX_SPLITS=_next_power_of_2(num_suffix_splits),
        num_warps=4,
        num_stages=1,
    )
    return output


def shared_paged_prefix_paged_suffix_gqa_decode_attention(
    query,
    key_cache,
    value_cache,
    block_tables,
    seq_lens,
    prefix_length: int,
    page_size: int = 16,
    prefix_block_t: int = 128,
    prefix_block_m: int = 4,
    suffix_split_size: int = 512,
    suffix_block_t: int = 128,
    num_warps: int = 4,
):
    """Run decode attention over shared paged prefix and per-request paged suffix."""
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    if (
        query.ndim != 3
        or key_cache.ndim != 4
        or value_cache.shape != key_cache.shape
        or block_tables.ndim != 2
        or seq_lens.ndim != 1
    ):
        raise ValueError(
            "expected query=[B,Hq,D], cache=[pages,page,Hkv,D], "
            "block_tables=[B,pages], seq_lens=[B]"
        )
    batch, num_q_heads, head_dim = query.shape
    _, cache_page_size, num_kv_heads, cache_dim = key_cache.shape
    if (
        block_tables.shape[0] != batch
        or seq_lens.shape[0] != batch
        or cache_dim != head_dim
        or head_dim != 128
        or num_q_heads % num_kv_heads
        or cache_page_size != page_size
        or page_size != 16
    ):
        raise ValueError("requires compatible page-16 GQA tensors with head_dim=128")
    if block_tables.shape[1] * page_size < int(seq_lens.max().item()):
        raise ValueError("block_tables do not cover seq_lens")
    if prefix_length < 1 or prefix_length > int(seq_lens.min().item()):
        raise ValueError("prefix_length must be positive and no larger than min seq_len")
    if prefix_block_t not in {32, 64, 128} or prefix_block_t % page_size:
        raise ValueError("prefix_block_t must be 32, 64, or 128 and divisible by page_size")
    if prefix_block_m not in {1, 2, 4, 8}:
        raise ValueError("prefix_block_m must be one of 1, 2, 4, 8")
    if suffix_block_t not in {16, 32, 64, 128}:
        raise ValueError("suffix_block_t must be one of 16, 32, 64, 128")
    if suffix_split_size % suffix_block_t:
        raise ValueError("suffix_split_size must be divisible by suffix_block_t")
    if num_warps not in {1, 2, 4, 8}:
        raise ValueError("num_warps must be one of 1, 2, 4, 8")
    max_suffix_length = int(seq_lens.max().item()) - prefix_length
    num_suffix_splits = triton.cdiv(max_suffix_length, suffix_split_size)
    if num_suffix_splits < 1 or num_suffix_splits > 16:
        raise ValueError("suffix path supports 1 to 16 splits")

    prefix_shape = (batch, num_q_heads)
    prefix_output = torch.empty(
        (*prefix_shape, head_dim), device=query.device, dtype=torch.float32
    )
    prefix_max = torch.empty(prefix_shape, device=query.device, dtype=torch.float32)
    prefix_sum = torch.empty_like(prefix_max)
    suffix_shape = (batch, num_q_heads, num_suffix_splits)
    suffix_output = torch.empty(
        (*suffix_shape, head_dim), device=query.device, dtype=torch.float32
    )
    suffix_max = torch.empty(suffix_shape, device=query.device, dtype=torch.float32)
    suffix_sum = torch.empty_like(suffix_max)
    output = torch.empty_like(query)

    rows_per_kv = batch * (num_q_heads // num_kv_heads)
    _shared_paged_prefix_gqa_partial_kernel[
        (num_kv_heads, triton.cdiv(rows_per_kv, prefix_block_m))
    ](
        query,
        key_cache,
        value_cache,
        block_tables[0],
        prefix_output,
        prefix_max,
        prefix_sum,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        batch_size=batch,
        prefix_length=prefix_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        BLOCK_T=prefix_block_t,
        BLOCK_M=prefix_block_m,
        PAGE_SIZE=page_size,
        num_warps=num_warps,
        num_stages=1,
    )
    _paged_suffix_gqa_partial_kernel[(num_suffix_splits, batch * num_q_heads)](
        query,
        key_cache,
        value_cache,
        block_tables,
        seq_lens,
        suffix_output,
        suffix_max,
        suffix_sum,
        query.stride(0),
        query.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        block_tables.stride(0),
        prefix_length=prefix_length,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        PAGE_SIZE=page_size,
        SPLIT_SIZE=suffix_split_size,
        BLOCK_T=suffix_block_t,
        num_warps=num_warps,
        num_stages=1,
    )
    _prefix_suffix_gqa_reduce_kernel[(batch * num_q_heads,)](
        prefix_output,
        prefix_max,
        prefix_sum,
        suffix_output,
        suffix_max,
        suffix_sum,
        output,
        output.stride(0),
        output.stride(1),
        num_q_heads=num_q_heads,
        head_dim=head_dim,
        NUM_SUFFIX_SPLITS=num_suffix_splits,
        BLOCK_SUFFIX_SPLITS=_next_power_of_2(num_suffix_splits),
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
        BLOCK_SPLITS=_next_power_of_2(num_splits),
        num_warps=4,
        num_stages=1,
    )
    return output


def gqa_decode_attention_split_kv_bf16_partials_candidate(
    query,
    key,
    value,
    split_size: int = 512,
    block_t: int = 32,
    num_warps: int = 4,
):
    """Experimental split-KV path that stores partial vectors in BF16.

    This keeps the compute boundary identical to the scalar split-KV path while
    halving partial-output traffic into the reduce kernel. It is not wired into
    dispatch until correctness and latency are proven on L20.
    """
    if torch is None or triton is None:
        raise RuntimeError("requires PyTorch and Triton")
    batch, context_length, num_q_heads, num_kv_heads, head_dim, num_splits = (
        _validate_split_kv_args(query, key, value, split_size, block_t, num_warps)
    )
    partial_shape = (batch, num_q_heads, num_splits)
    partial_output = torch.empty(
        (*partial_shape, head_dim), device=query.device, dtype=query.dtype
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
        BLOCK_SPLITS=_next_power_of_2(num_splits),
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
        BLOCK_SPLITS=_next_power_of_2(num_splits),
        num_warps=4,
        num_stages=1,
    )
    return output


def gqa_decode_attention_split_kv_tensor_core_dsplit_candidate(
    query,
    key,
    value,
    split_size: int = 512,
    block_t: int = 64,
    block_q: int = 2,
    block_d: int = 64,
    num_warps: int = 4,
):
    """Experimental grouped-Q Tensor-Core path with head-dim split output tiles."""
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
    if block_d not in {32, 64, 128}:
        raise ValueError("block_d must be one of 32, 64, 128")
    q_groups = triton.cdiv(gqa_ratio, block_q)
    d_blocks = triton.cdiv(head_dim, block_d)
    partial_shape = (batch, num_q_heads, num_splits)
    partial_output = torch.empty(
        (*partial_shape, head_dim), device=query.device, dtype=torch.float32
    )
    partial_max = torch.empty(partial_shape, device=query.device, dtype=torch.float32)
    partial_sum = torch.empty_like(partial_max)
    output = torch.empty_like(query)
    _gqa_decode_attention_tc_dsplit_partial_kernel[
        (num_splits, batch * num_kv_heads, q_groups * d_blocks)
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
        BLOCK_D=block_d,
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
        BLOCK_SPLITS=_next_power_of_2(num_splits),
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
        BLOCK_SPLITS=_next_power_of_2(num_splits),
        num_warps=4,
        num_stages=1,
    )
    return output


def should_use_l20_gqa_decode_attention(batch: int, context_length: int) -> bool:
    """Conservative gate derived from L20 BF16 head_dim=128 measurements."""
    return batch >= 4 or context_length <= 1024


def should_use_l20_split_kv_attention(context_length: int) -> bool:
    return context_length >= 2048
