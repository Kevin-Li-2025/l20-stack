#!/usr/bin/env python3
"""Benchmark L20 shared-prefix packed decode attention."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from l20_stack.ops.triton_decode_attention import (
    gqa_decode_attention_split_kv,
    shared_prefix_gqa_decode_attention,
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


def reference(query, key, value):
    ratio = query.shape[1] // key.shape[1]
    expanded_key = key.repeat_interleave(ratio, dim=1).transpose(0, 1)
    expanded_value = value.repeat_interleave(ratio, dim=1).transpose(0, 1)
    return torch.nn.functional.scaled_dot_product_attention(
        query.unsqueeze(2),
        expanded_key.unsqueeze(0).expand(query.shape[0], -1, -1, -1),
        expanded_value.unsqueeze(0).expand(query.shape[0], -1, -1, -1),
    ).squeeze(2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batches", default="1,4,8,16")
    parser.add_argument("--contexts", default="1024,4096,8192")
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--split-size", type=int, default=512)
    parser.add_argument("--baseline-block-t", type=int, default=128)
    parser.add_argument("--baseline-num-warps", type=int, default=4)
    parser.add_argument("--shared-block-ts", default="64,128")
    parser.add_argument("--shared-block-ms", default="2,4,8")
    parser.add_argument("--shared-num-warps", default="4")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(931)
    reports = []
    for batch in parse_ints(args.batches):
        for context in parse_ints(args.contexts):
            query = torch.randn(
                batch,
                args.q_heads,
                args.head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            key = torch.randn(
                context,
                args.kv_heads,
                args.head_dim,
                device="cuda",
                dtype=torch.bfloat16,
            )
            value = torch.randn_like(key)
            expected = reference(query, key, value)
            baseline_key = key.unsqueeze(0).expand(batch, -1, -1, -1)
            baseline_value = value.unsqueeze(0).expand(batch, -1, -1, -1)
            baseline = gqa_decode_attention_split_kv(
                query,
                baseline_key,
                baseline_value,
                split_size=args.split_size,
                block_t=args.baseline_block_t,
                num_warps=args.baseline_num_warps,
            )
            baseline_correct = torch.allclose(
                baseline, expected, rtol=2e-2, atol=2e-2
            )
            baseline_samples = latency_ms(
                lambda: gqa_decode_attention_split_kv(
                    query,
                    baseline_key,
                    baseline_value,
                    split_size=args.split_size,
                    block_t=args.baseline_block_t,
                    num_warps=args.baseline_num_warps,
                ),
                args.warmup,
                args.iterations,
            )
            reports.append(
                {
                    "path": "per_request_split_kv",
                    "batch": batch,
                    "context": context,
                    "block_t": args.baseline_block_t,
                    "block_m": 1,
                    "num_warps": args.baseline_num_warps,
                    "correct": bool(baseline_correct),
                    "max_abs_error": float(
                        (baseline.float() - expected.float()).abs().max()
                    ),
                    **summarize(baseline_samples),
                }
            )
            for block_t in parse_ints(args.shared_block_ts):
                for block_m in parse_ints(args.shared_block_ms):
                    for num_warps in parse_ints(args.shared_num_warps):
                        actual = shared_prefix_gqa_decode_attention(
                            query,
                            key,
                            value,
                            block_t=block_t,
                            block_m=block_m,
                            num_warps=num_warps,
                        )
                        correct = torch.allclose(
                            actual, expected, rtol=2e-2, atol=2e-2
                        )
                        samples = latency_ms(
                            lambda block_t=block_t, block_m=block_m, num_warps=num_warps: (
                                shared_prefix_gqa_decode_attention(
                                    query,
                                    key,
                                    value,
                                    block_t=block_t,
                                    block_m=block_m,
                                    num_warps=num_warps,
                                )
                            ),
                            args.warmup,
                            args.iterations,
                        )
                        reports.append(
                            {
                                "path": "shared_prefix_packed",
                                "batch": batch,
                                "context": context,
                                "block_t": block_t,
                                "block_m": block_m,
                                "num_warps": num_warps,
                                "correct": bool(correct),
                                "max_abs_error": float(
                                    (actual.float() - expected.float()).abs().max()
                                ),
                                **summarize(samples),
                            }
                        )
    comparisons = []
    for batch in parse_ints(args.batches):
        for context in parse_ints(args.contexts):
            shape_reports = [
                item
                for item in reports
                if item["batch"] == batch and item["context"] == context and item["correct"]
            ]
            baseline = next(
                item for item in shape_reports if item["path"] == "per_request_split_kv"
            )
            candidates = [
                item for item in shape_reports if item["path"] == "shared_prefix_packed"
            ]
            best = min(candidates, key=lambda item: item["median_ms"])
            comparisons.append(
                {
                    "batch": batch,
                    "context": context,
                    "baseline_median_ms": baseline["median_ms"],
                    "best_shared_median_ms": best["median_ms"],
                    "best_shared_block_t": best["block_t"],
                    "best_shared_block_m": best["block_m"],
                    "best_shared_num_warps": best["num_warps"],
                    "speedup": baseline["median_ms"] / best["median_ms"],
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
            "shared_prefix_only": True,
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
