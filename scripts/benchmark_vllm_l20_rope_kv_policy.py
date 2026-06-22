#!/usr/bin/env python3
"""Sweep L20 RoPE/KV launch policies against the current production choice."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 64, 96, 128, 256, 512],
    )
    parser.add_argument("--q-heads", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    return parser.parse_args()


def benchmark_case(
    torch,
    triton,
    kernel,
    tokens,
    num_warps,
    repetitions,
    q_heads,
    kv_heads,
    head_dim,
):
    query = torch.randn(tokens, q_heads, head_dim, device="cuda", dtype=torch.float16)
    key = torch.randn(tokens, kv_heads, head_dim, device="cuda", dtype=torch.float16)
    value = torch.randn_like(key)
    positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
    angles = torch.randn(max(tokens, 2048), head_dim // 2, device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
    blocks = max(8, (tokens + 15) // 16)
    key_cache = torch.empty(
        blocks, 16, kv_heads, head_dim, device="cuda", dtype=torch.float16
    )
    value_cache = torch.empty_like(key_cache)

    def launch():
        kernel[(tokens, q_heads)](
            query,
            key,
            value,
            positions,
            cos_sin,
            slots,
            key_cache,
            value_cache,
            query.stride(0),
            query.stride(1),
            key.stride(0),
            key.stride(1),
            value.stride(0),
            value.stride(1),
            key_cache.stride(0),
            key_cache.stride(1),
            key_cache.stride(2),
            value_cache.stride(0),
            value_cache.stride(1),
            value_cache.stride(2),
            cos_sin.stride(0),
            tokens,
            q_heads,
            kv_heads,
            head_dim,
            head_dim,
            key_cache.shape[1],
            BLOCK_SIZE=head_dim,
            num_warps=num_warps,
            num_stages=1,
        )

    latency = triton.testing.do_bench(launch, warmup=100, rep=repetitions)
    return {
        "tokens": tokens,
        "num_warps": num_warps,
        "latency_ms": latency,
    }


def main() -> int:
    args = parse_args()
    import torch
    import triton
    from vllm.v1.attention.ops.l20_rope_kv import _l20_neox_rope_kv_kernel

    if torch.cuda.get_device_name() != "NVIDIA L20":
        raise SystemExit("benchmark requires NVIDIA L20")
    tokens_to_test = tuple(args.tokens)
    samples = [
        benchmark_case(
            torch,
            triton,
            _l20_neox_rope_kv_kernel,
            tokens,
            num_warps,
            args.repetitions,
            args.q_heads,
            args.kv_heads,
            args.head_dim,
        )
        for _ in range(args.rounds)
        for tokens in tokens_to_test
        for num_warps in (1, 2, 4, 8)
    ]
    reports = []
    for tokens in tokens_to_test:
        for num_warps in (1, 2, 4, 8):
            values = [
                sample["latency_ms"]
                for sample in samples
                if sample["tokens"] == tokens
                and sample["num_warps"] == num_warps
            ]
            reports.append(
                {
                    "tokens": tokens,
                    "num_warps": num_warps,
                    "latency_ms": statistics.median(values),
                    "minimum_ms": min(values),
                    "maximum_ms": max(values),
                    "samples_ms": values,
                }
            )
    for tokens in sorted({report["tokens"] for report in reports}):
        rows = [report for report in reports if report["tokens"] == tokens]
        best = min(rows, key=lambda report: report["latency_ms"])
        baseline = next(report for report in rows if report["num_warps"] == 4)
        for report in rows:
            report["speedup_vs_4_warps"] = (
                baseline["latency_ms"] / report["latency_ms"]
            )
            report["selected"] = report is best
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "shape": {
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
        },
        "reports": reports,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
