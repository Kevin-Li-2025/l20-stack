"""Opt-in dispatch helper for experimental L20 tree attention."""

from __future__ import annotations

import os
from typing import Optional

import torch
from vllm.v1.attention.ops.l20_tree_attention import (
    allocate_tree_attention_workspace,
    causal_verifier_attention_paged,
    hybrid_tree_attention_paged_prefix,
)


def l20_tree_attention_enabled() -> bool:
    return os.getenv("VLLM_ENABLE_L20_TREE_ATTENTION", "0") == "1"


def should_dispatch_l20_tree_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    suffix_key: torch.Tensor,
    block_table: torch.Tensor,
    cached_length: int,
    *,
    min_cached_length: int = 4096,
) -> bool:
    if not l20_tree_attention_enabled():
        return False
    if torch.cuda.get_device_capability() != (8, 9):
        return False
    if torch.cuda.is_current_stream_capturing():
        return False
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False
    if key_cache.dtype != query.dtype:
        return False
    if query.ndim != 4 or key_cache.ndim != 4 or suffix_key.ndim != 4:
        return False
    if query.shape[-1] != 128 or key_cache.shape[-1] != 128:
        return False
    if key_cache.shape[1] != 16:
        return False
    if query.shape[0] != block_table.shape[0]:
        return False
    if query.shape[1] != suffix_key.shape[1]:
        return False
    return cached_length >= min_cached_length


def should_dispatch_l20_causal_verifier_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    suffix_key: torch.Tensor,
    block_table: torch.Tensor,
    cached_length: int,
) -> bool:
    if query.ndim != 4:
        return False
    draft_length = query.shape[1]
    if draft_length < 2 or draft_length > 64:
        return False
    return should_dispatch_l20_tree_attention(
        query,
        key_cache,
        suffix_key,
        block_table,
        cached_length,
        min_cached_length=1024,
    )


def maybe_l20_tree_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    block_table: torch.Tensor,
    ancestor_mask: torch.Tensor,
    cached_length: int,
    *,
    workspace=None,
) -> Optional[torch.Tensor]:
    if not should_dispatch_l20_tree_attention(
        query,
        key_cache,
        suffix_key,
        block_table,
        cached_length,
    ):
        return None
    if workspace is None:
        workspace = allocate_tree_attention_workspace(query)
    return hybrid_tree_attention_paged_prefix(
        query,
        key_cache,
        value_cache,
        suffix_key,
        suffix_value,
        block_table,
        ancestor_mask,
        cached_length,
        workspace=workspace,
    )


def maybe_l20_causal_verifier_attention(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    suffix_key: torch.Tensor,
    suffix_value: torch.Tensor,
    block_table: torch.Tensor,
    cached_length: int,
    *,
    workspace=None,
) -> Optional[torch.Tensor]:
    if not should_dispatch_l20_causal_verifier_attention(
        query,
        key_cache,
        suffix_key,
        block_table,
        cached_length,
    ):
        return None
    return causal_verifier_attention_paged(
        query,
        key_cache,
        value_cache,
        suffix_key,
        suffix_value,
        block_table,
        cached_length,
    )
