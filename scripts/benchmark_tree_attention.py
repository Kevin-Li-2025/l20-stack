#!/usr/bin/env python3
"""Benchmark L20 hybrid tree attention for speculative verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import torch

from l20_stack.ops.triton_decode_attention import gqa_decode_attention
from l20_stack.ops.triton_tree_attention import (
    allocate_tree_attention_workspace,
    hybrid_tree_attention,
    hybrid_tree_attention_split,
    l20_tree_attention_block_t,
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


def repeated_decode_chain(query, key, value, cached: int):
    outputs = []
    for draft_index in range(query.shape[1]):
        outputs.append(
            gqa_decode_attention(
                query[:, draft_index],
                key[:, : cached + draft_index + 1],
                value[:, : cached + draft_index + 1],
            )
        )
    return torch.stack(outputs, dim=1)


def run_case(
    batch: int,
    cached: int,
    draft: int,
    tree: str,
    iterations: int,
    block_t: Optional[int],
):
    torch.manual_seed(23 + batch + cached + draft)
    num_q_heads = 16
    num_kv_heads = 8
    head_dim = 128
    query = torch.randn(batch, draft, num_q_heads, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn(
        batch,
        cached + draft,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
    )
    value = torch.randn_like(key)
    if tree == "chain":
        mask = make_chain_tree_mask(draft, device="cuda")
    elif tree == "fork2":
        mask = make_fork_tree_mask(draft, branch=2, device="cuda")
    else:
        raise ValueError(f"unknown tree shape: {tree}")

    expected = torch_tree_attention_reference(query, key, value, mask, cached)
    actual = hybrid_tree_attention(query, key, value, mask, cached, block_t=block_t)
    workspace = allocate_tree_attention_workspace(query)
    split_actual = hybrid_tree_attention_split(
        query, key, value, mask, cached, workspace=workspace, block_t=block_t
    )
    correct = torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)
    split_correct = torch.allclose(split_actual, expected, rtol=2e-2, atol=2e-2)
    baseline_ms = latency_ms(
        lambda: torch_tree_attention_reference(query, key, value, mask, cached),
        iterations=max(10, iterations // 4),
    )
    l20_ms = latency_ms(
        lambda: hybrid_tree_attention(query, key, value, mask, cached, block_t=block_t),
        iterations=iterations,
    )
    split_ms = latency_ms(
        lambda: hybrid_tree_attention_split(
            query, key, value, mask, cached, workspace=workspace, block_t=block_t
        ),
        iterations=iterations,
    )
    report = {
        "batch": batch,
        "block_t": block_t or l20_tree_attention_block_t(cached),
        "cached_length": cached,
        "draft_length": draft,
        "tree": tree,
        "correct": bool(correct),
        "split_correct": bool(split_correct),
        "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        "split_max_abs_error": float((split_actual.float() - expected.float()).abs().max()),
        "torch_dense_ms": baseline_ms,
        "l20_tree_ms": l20_ms,
        "l20_split_tree_ms": split_ms,
        "speedup_vs_torch_dense": baseline_ms / l20_ms,
        "split_speedup_vs_torch_dense": baseline_ms / split_ms,
        "split_vs_monolithic": l20_ms / split_ms,
    }
    if tree == "chain":
        decode_expected = repeated_decode_chain(query, key, value, cached)
        decode_ms = latency_ms(
            lambda: repeated_decode_chain(query, key, value, cached),
            iterations=max(10, iterations // 2),
        )
        report.update(
            {
                "matches_repeated_decode": bool(
                    torch.allclose(actual, decode_expected, rtol=2e-2, atol=2e-2)
                ),
                "repeated_decode_ms": decode_ms,
                "speedup_vs_repeated_decode": decode_ms / l20_ms,
            }
        )
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--cached", type=int, nargs="+", default=[512, 2048])
    parser.add_argument("--draft", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--trees", nargs="+", default=["chain", "fork2"])
    parser.add_argument("--block-t", type=int, choices=[32, 64, 128])
    args = parser.parse_args()

    reports = []
    for batch in args.batches:
        for cached in args.cached:
            for draft in args.draft:
                for tree in args.trees:
                    reports.append(
                        run_case(
                            batch,
                            cached,
                            draft,
                            tree,
                            args.iterations,
                            args.block_t,
                        )
                    )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "operator": "hybrid_tree_attention",
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
