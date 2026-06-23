#!/usr/bin/env python3
"""Smoke test the L20 tree-attention op through the vLLM namespace."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch


def latency_ms(function, warmup=20, iterations=100):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        function()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iterations


def make_chain_tree_mask(draft_length: int, *, device):
    positions = torch.arange(draft_length, device=device)
    return positions[None, :] <= positions[:, None]


def make_paged_prefix(prefix_key, prefix_value, page_size: int = 16):
    batch, cached_length, num_kv_heads, head_dim = prefix_key.shape
    pages_per_batch = cached_length // page_size
    num_pages = batch * pages_per_batch
    block_table = torch.randperm(num_pages, device=prefix_key.device, dtype=torch.int32).reshape(
        batch, pages_per_batch
    )
    key_cache = torch.empty(
        num_pages,
        page_size,
        num_kv_heads,
        head_dim,
        device=prefix_key.device,
        dtype=prefix_key.dtype,
    )
    value_cache = torch.empty_like(key_cache)
    key_pages = prefix_key.reshape(batch, pages_per_batch, page_size, num_kv_heads, head_dim)
    value_pages = prefix_value.reshape_as(key_pages)
    for batch_index in range(batch):
        key_cache[block_table[batch_index].long()] = key_pages[batch_index]
        value_cache[block_table[batch_index].long()] = value_pages[batch_index]
    return key_cache, value_cache, block_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--cached", type=int, default=4096)
    parser.add_argument("--draft", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_L20_TREE_ATTENTION", "1")
    from vllm.v1.attention.ops.l20_tree_attention import (
        allocate_tree_attention_workspace,
        torch_tree_attention_reference,
    )
    from vllm.v1.attention.ops.l20_tree_attention_dispatch import (
        maybe_l20_tree_attention,
        should_dispatch_l20_tree_attention,
    )

    torch.manual_seed(53)
    num_q_heads = 16
    num_kv_heads = 8
    head_dim = 128
    query = torch.randn(
        args.batch,
        args.draft,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    prefix_key = torch.randn(
        args.batch,
        args.cached,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    prefix_value = torch.randn_like(prefix_key)
    suffix_key = torch.randn(
        args.batch,
        args.draft,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    suffix_value = torch.randn_like(suffix_key)
    key = torch.cat([prefix_key, suffix_key], dim=1)
    value = torch.cat([prefix_value, suffix_value], dim=1)
    mask = make_chain_tree_mask(args.draft, device="cuda")
    key_cache, value_cache, block_table = make_paged_prefix(prefix_key, prefix_value)
    workspace = allocate_tree_attention_workspace(query)
    expected = torch_tree_attention_reference(query, key, value, mask, args.cached)
    should_dispatch = should_dispatch_l20_tree_attention(
        query,
        key_cache,
        suffix_key,
        block_table,
        args.cached,
    )
    actual = maybe_l20_tree_attention(
        query,
        key_cache,
        value_cache,
        suffix_key,
        suffix_value,
        block_table,
        mask,
        args.cached,
        workspace=workspace,
    )
    if actual is None:
        raise RuntimeError("L20 tree attention dispatch did not enable")
    runtime_ms = latency_ms(
        lambda: maybe_l20_tree_attention(
            query,
            key_cache,
            value_cache,
            suffix_key,
            suffix_value,
            block_table,
            mask,
            args.cached,
            workspace=workspace,
        ),
        iterations=args.iterations,
    )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "import_path": "vllm.v1.attention.ops.l20_tree_attention",
        "dispatch_path": "vllm.v1.attention.ops.l20_tree_attention_dispatch",
        "batch": args.batch,
        "cached_length": args.cached,
        "draft_length": args.draft,
        "should_dispatch": bool(should_dispatch),
        "correct": bool(torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)),
        "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        "paged_prefix_ms": runtime_ms,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
