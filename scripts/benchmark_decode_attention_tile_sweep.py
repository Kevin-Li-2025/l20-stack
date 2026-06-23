#!/usr/bin/env python3
"""Sweep L20 split-KV decode-attention tile shapes."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from l20_stack.ops.triton_decode_attention import gqa_decode_attention_split_kv


def reference(query, key, value):
    ratio = query.shape[1] // key.shape[2]
    expanded_key = key.repeat_interleave(ratio, dim=2).transpose(1, 2)
    expanded_value = value.repeat_interleave(ratio, dim=2).transpose(1, 2)
    return torch.nn.functional.scaled_dot_product_attention(
        query.unsqueeze(2),
        expanded_key,
        expanded_value,
    ).squeeze(2)


def latency_ms(function, warmup=25, iterations=100):
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


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[round(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[round(0.90 * (len(ordered) - 1))],
        "samples_ms": samples,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--split-sizes", default="256,512,1024")
    parser.add_argument("--block-ts", default="16,32,64,128")
    parser.add_argument("--num-warps", default="2,4,8")
    parser.add_argument("--iterations", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def parse_ints(text):
    return [int(item) for item in text.split(",") if item]


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(31)
    query = torch.randn(
        args.batch,
        args.q_heads,
        args.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        args.batch,
        args.context,
        args.kv_heads,
        args.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    expected = reference(query, key, value)
    reports = []
    for split_size in parse_ints(args.split_sizes):
        for block_t in parse_ints(args.block_ts):
            if split_size % block_t:
                continue
            for num_warps in parse_ints(args.num_warps):
                actual = gqa_decode_attention_split_kv(
                    query,
                    key,
                    value,
                    split_size=split_size,
                    block_t=block_t,
                    num_warps=num_warps,
                )
                correct = torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)
                samples = latency_ms(
                    lambda split_size=split_size, block_t=block_t, num_warps=num_warps: (
                        gqa_decode_attention_split_kv(
                            query,
                            key,
                            value,
                            split_size=split_size,
                            block_t=block_t,
                            num_warps=num_warps,
                        )
                    ),
                    warmup=args.warmup,
                    iterations=args.iterations,
                )
                reports.append(
                    {
                        "split_size": split_size,
                        "block_t": block_t,
                        "num_warps": num_warps,
                        "correct": bool(correct),
                        "max_abs_error": float((actual.float() - expected.float()).abs().max()),
                        **summarize(samples),
                    }
                )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "context": args.context,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "dtype": "bfloat16",
        },
        "reports": reports,
    }
    correct_reports = [report for report in reports if report["correct"]]
    if correct_reports:
        result["best_correct"] = min(correct_reports, key=lambda item: item["median_ms"])
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
