#!/usr/bin/env python3
"""Benchmark L20 paged shared-prefix plus suffix decode attention."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from l20_stack.ops.triton_decode_attention import (
    gqa_decode_attention_split_kv,
    shared_paged_prefix_suffix_gqa_decode_attention,
    shared_prefix_suffix_gqa_decode_attention,
)


def parse_ints(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def latency_ms(function, warmup: int, iterations: int) -> list[float]:
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def summarize(samples: list[float]) -> dict:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[round(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[round(0.90 * (len(ordered) - 1))],
        "samples_ms": samples,
    }


def make_paged_prefix(prefix_key, prefix_value, page_size: int, random_pages: bool):
    prefix_length, num_kv_heads, head_dim = prefix_key.shape
    if prefix_length % page_size:
        raise ValueError("prefix_length must be divisible by page_size")
    pages = prefix_length // page_size
    if random_pages:
        block_table = torch.randperm(pages, device=prefix_key.device, dtype=torch.int32)
    else:
        block_table = torch.arange(pages, device=prefix_key.device, dtype=torch.int32)
    key_cache = torch.empty(
        pages,
        page_size,
        num_kv_heads,
        head_dim,
        device=prefix_key.device,
        dtype=prefix_key.dtype,
    )
    value_cache = torch.empty_like(key_cache)
    key_pages = prefix_key.reshape(pages, page_size, num_kv_heads, head_dim)
    value_pages = prefix_value.reshape_as(key_pages)
    key_cache[block_table.long()] = key_pages
    value_cache[block_table.long()] = value_pages
    return key_cache, value_cache, block_table


def reference(query, prefix_key, prefix_value, suffix_key, suffix_value):
    batch = query.shape[0]
    prefix_key_batched = prefix_key.unsqueeze(0).expand(batch, -1, -1, -1)
    prefix_value_batched = prefix_value.unsqueeze(0).expand(batch, -1, -1, -1)
    key = torch.cat([prefix_key_batched, suffix_key], dim=1)
    value = torch.cat([prefix_value_batched, suffix_value], dim=1)
    ratio = query.shape[1] // key.shape[2]
    expanded_key = key.repeat_interleave(ratio, dim=2).transpose(1, 2)
    expanded_value = value.repeat_interleave(ratio, dim=2).transpose(1, 2)
    return torch.nn.functional.scaled_dot_product_attention(
        query.unsqueeze(2),
        expanded_key,
        expanded_value,
    ).squeeze(2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="8,16")
    parser.add_argument("--prefix-lengths", default="4096,8192")
    parser.add_argument("--suffix-lengths", default="64,256")
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--random-pages", action="store_true")
    parser.add_argument("--baseline-split-size", type=int, default=1024)
    parser.add_argument("--baseline-block-t", type=int, default=128)
    parser.add_argument("--prefix-block-t", type=int, default=128)
    parser.add_argument("--prefix-block-m", type=int, default=8)
    parser.add_argument("--suffix-split-size", type=int, default=512)
    parser.add_argument("--suffix-block-t", type=int, default=128)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(20260626)
    reports = []
    comparisons = []
    for batch in parse_ints(args.batches):
        for prefix_length in parse_ints(args.prefix_lengths):
            for suffix_length in parse_ints(args.suffix_lengths):
                query = torch.randn(
                    batch,
                    args.q_heads,
                    args.head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                prefix_key = torch.randn(
                    prefix_length,
                    args.kv_heads,
                    args.head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                prefix_value = torch.randn_like(prefix_key)
                suffix_key = torch.randn(
                    batch,
                    suffix_length,
                    args.kv_heads,
                    args.head_dim,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                suffix_value = torch.randn_like(suffix_key)
                key_cache, value_cache, block_table = make_paged_prefix(
                    prefix_key,
                    prefix_value,
                    args.page_size,
                    args.random_pages,
                )
                expected = reference(
                    query, prefix_key, prefix_value, suffix_key, suffix_value
                )
                baseline_key = torch.cat(
                    [
                        prefix_key.unsqueeze(0).expand(batch, -1, -1, -1),
                        suffix_key,
                    ],
                    dim=1,
                )
                baseline_value = torch.cat(
                    [
                        prefix_value.unsqueeze(0).expand(batch, -1, -1, -1),
                        suffix_value,
                    ],
                    dim=1,
                )
                candidates = {
                    "per_request_full_split_kv": lambda: gqa_decode_attention_split_kv(
                        query,
                        baseline_key,
                        baseline_value,
                        split_size=args.baseline_split_size,
                        block_t=args.baseline_block_t,
                        num_warps=args.num_warps,
                    ),
                    "shared_prefix_suffix_contiguous": lambda: shared_prefix_suffix_gqa_decode_attention(
                        query,
                        prefix_key,
                        prefix_value,
                        suffix_key,
                        suffix_value,
                        prefix_block_t=args.prefix_block_t,
                        prefix_block_m=args.prefix_block_m,
                        suffix_split_size=args.suffix_split_size,
                        suffix_block_t=args.suffix_block_t,
                        num_warps=args.num_warps,
                    ),
                    "shared_prefix_suffix_paged": lambda: shared_paged_prefix_suffix_gqa_decode_attention(
                        query,
                        key_cache,
                        value_cache,
                        block_table,
                        suffix_key,
                        suffix_value,
                        prefix_length,
                        page_size=args.page_size,
                        prefix_block_t=args.prefix_block_t,
                        prefix_block_m=args.prefix_block_m,
                        suffix_split_size=args.suffix_split_size,
                        suffix_block_t=args.suffix_block_t,
                        num_warps=args.num_warps,
                    ),
                }
                shape_reports = []
                for name, function in candidates.items():
                    actual = function()
                    correct = torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)
                    samples = latency_ms(function, args.warmup, args.iterations)
                    report = {
                        "path": name,
                        "batch": batch,
                        "prefix_length": prefix_length,
                        "suffix_length": suffix_length,
                        "correct": bool(correct),
                        "max_abs_error": float(
                            (actual.float() - expected.float()).abs().max()
                        ),
                        **summarize(samples),
                    }
                    reports.append(report)
                    shape_reports.append(report)
                baseline = next(
                    item
                    for item in shape_reports
                    if item["path"] == "per_request_full_split_kv"
                )
                contiguous = next(
                    item
                    for item in shape_reports
                    if item["path"] == "shared_prefix_suffix_contiguous"
                )
                paged = next(
                    item
                    for item in shape_reports
                    if item["path"] == "shared_prefix_suffix_paged"
                )
                comparisons.append(
                    {
                        "batch": batch,
                        "prefix_length": prefix_length,
                        "suffix_length": suffix_length,
                        "baseline_median_ms": baseline["median_ms"],
                        "contiguous_median_ms": contiguous["median_ms"],
                        "paged_median_ms": paged["median_ms"],
                        "contiguous_speedup_vs_baseline": (
                            baseline["median_ms"] / contiguous["median_ms"]
                        ),
                        "paged_speedup_vs_baseline": (
                            baseline["median_ms"] / paged["median_ms"]
                        ),
                        "paged_over_contiguous": (
                            contiguous["median_ms"] / paged["median_ms"]
                        ),
                    }
                )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "shape": {
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "dtype": "bfloat16",
            "page_size": args.page_size,
            "random_pages": args.random_pages,
        },
        "reports": reports,
        "comparisons": comparisons,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
