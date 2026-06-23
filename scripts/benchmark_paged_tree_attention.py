#!/usr/bin/env python3
"""Benchmark L20 paged-prefix hybrid tree attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from l20_stack.ops.triton_tree_attention import (
    allocate_tree_attention_workspace,
    hybrid_tree_attention_paged_prefix,
    hybrid_tree_attention_split,
    make_chain_tree_mask,
    torch_tree_attention_reference,
)


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


def make_fork_tree_mask(draft_length: int, *, branch: int, device):
    mask = torch.eye(draft_length, device=device, dtype=torch.bool)
    for token in range(draft_length):
        child = token
        parent = (token - 1) // branch
        while child > 0:
            mask[token, parent] = True
            if parent == 0:
                break
            child = parent
            parent = (parent - 1) // branch
    return mask


def make_paged_prefix(prefix_key, prefix_value, page_size: int = 16):
    batch, cached_length, num_kv_heads, head_dim = prefix_key.shape
    if cached_length % page_size:
        raise ValueError("cached_length must be divisible by page_size")
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
    prefix_key_pages = prefix_key.reshape(batch, pages_per_batch, page_size, num_kv_heads, head_dim)
    prefix_value_pages = prefix_value.reshape_as(prefix_key_pages)
    for batch_index in range(batch):
        key_cache[block_table[batch_index].long()] = prefix_key_pages[batch_index]
        value_cache[block_table[batch_index].long()] = prefix_value_pages[batch_index]
    return key_cache, value_cache, block_table


def run_case(batch: int, cached: int, draft: int, tree: str, iterations: int):
    torch.manual_seed(41 + batch + cached + draft)
    num_q_heads = 16
    num_kv_heads = 8
    head_dim = 128
    query = torch.randn(batch, draft, num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    prefix_key = torch.randn(
        batch, cached, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
    )
    prefix_value = torch.randn_like(prefix_key)
    suffix_key = torch.randn(
        batch, draft, num_kv_heads, head_dim, device="cuda", dtype=torch.float16
    )
    suffix_value = torch.randn_like(suffix_key)
    key = torch.cat([prefix_key, suffix_key], dim=1)
    value = torch.cat([prefix_value, suffix_value], dim=1)
    if tree == "chain":
        mask = make_chain_tree_mask(draft, device="cuda")
    elif tree == "fork2":
        mask = make_fork_tree_mask(draft, branch=2, device="cuda")
    else:
        raise ValueError(f"unknown tree shape: {tree}")
    key_cache, value_cache, block_table = make_paged_prefix(prefix_key, prefix_value)
    expected = torch_tree_attention_reference(query, key, value, mask, cached)
    workspace = allocate_tree_attention_workspace(query)
    paged = hybrid_tree_attention_paged_prefix(
        query,
        key_cache,
        value_cache,
        suffix_key,
        suffix_value,
        block_table,
        mask,
        cached,
        workspace=workspace,
    )
    split = hybrid_tree_attention_split(
        query,
        key,
        value,
        mask,
        cached,
        workspace=workspace,
    )
    paged_ms = latency_ms(
        lambda: hybrid_tree_attention_paged_prefix(
            query,
            key_cache,
            value_cache,
            suffix_key,
            suffix_value,
            block_table,
            mask,
            cached,
            workspace=workspace,
        ),
        iterations=iterations,
    )
    contiguous_ms = latency_ms(
        lambda: hybrid_tree_attention_split(query, key, value, mask, cached, workspace=workspace),
        iterations=iterations,
    )
    return {
        "batch": batch,
        "cached_length": cached,
        "draft_length": draft,
        "tree": tree,
        "paged_correct": bool(torch.allclose(paged, expected, rtol=2e-2, atol=2e-2)),
        "split_correct": bool(torch.allclose(split, expected, rtol=2e-2, atol=2e-2)),
        "paged_max_abs_error": float((paged.float() - expected.float()).abs().max()),
        "split_max_abs_error": float((split.float() - expected.float()).abs().max()),
        "paged_prefix_ms": paged_ms,
        "contiguous_split_ms": contiguous_ms,
        "paged_vs_contiguous_split": contiguous_ms / paged_ms,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--cached", type=int, nargs="+", default=[2048, 4096])
    parser.add_argument("--draft", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--trees", nargs="+", default=["chain", "fork2"])
    args = parser.parse_args()

    reports = []
    for batch in args.batches:
        for cached in args.cached:
            for draft in args.draft:
                for tree in args.trees:
                    reports.append(run_case(batch, cached, draft, tree, args.iterations))
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "operator": "hybrid_tree_attention_paged_prefix",
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
