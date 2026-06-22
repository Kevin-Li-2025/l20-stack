#!/usr/bin/env python3
"""Benchmark the L20 fused RoPE/KV path against vLLM's separate baseline."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=500)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 8, 32, 64, 96, 128, 256, 512, 1024],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import triton
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
    from vllm import _custom_ops
    from vllm.v1.attention.ops.l20_rope_kv import (
        l20_rope_and_cache,
        l20_rope_kv_num_warps,
    )

    if torch.cuda.get_device_name() != "NVIDIA L20":
        raise SystemExit("benchmark requires NVIDIA L20")
    reports = []
    for tokens in args.tokens:
        torch.manual_seed(tokens)
        query = torch.randn(tokens, 32, 128, device="cuda", dtype=torch.float16)
        key = torch.randn(tokens, 8, 128, device="cuda", dtype=torch.float16)
        value = torch.randn_like(key)
        positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
        angles = torch.randn(max(tokens, 2048), 64, device="cuda")
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
        blocks = max(8, (tokens + 15) // 16)
        key_cache = torch.empty(
            blocks, 16, 8, 128, device="cuda", dtype=torch.float16
        )
        value_cache = torch.empty_like(key_cache)
        slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
        scale = torch.ones(1, device="cuda")
        baseline_samples = []
        fused_samples = []
        for _ in range(args.rounds):
            baseline_query = query.clone()
            baseline_key = key.clone()
            fused_query = query.clone()
            fused_key = key.clone()

            def baseline():
                apply_rope_with_cos_sin_cache_inplace(
                    positions,
                    baseline_query,
                    baseline_key,
                    128,
                    cos_sin,
                    True,
                )
                _custom_ops.reshape_and_cache_flash(
                    baseline_key,
                    value,
                    key_cache,
                    value_cache,
                    slots,
                    "auto",
                    scale,
                    scale,
                )

            def fused():
                l20_rope_and_cache(
                    fused_query,
                    fused_key,
                    value,
                    positions,
                    cos_sin,
                    True,
                    key_cache,
                    value_cache,
                    slots,
                )

            baseline_samples.append(
                triton.testing.do_bench(
                    baseline, warmup=100, rep=args.repetitions
                )
            )
            fused_samples.append(
                triton.testing.do_bench(fused, warmup=100, rep=args.repetitions)
            )
        baseline_ms = statistics.median(baseline_samples)
        fused_ms = statistics.median(fused_samples)
        reports.append(
            {
                "tokens": tokens,
                "num_warps": l20_rope_kv_num_warps(tokens, 128),
                "baseline_ms": baseline_ms,
                "fused_ms": fused_ms,
                "speedup": baseline_ms / fused_ms,
                "baseline_samples_ms": baseline_samples,
                "fused_samples_ms": fused_samples,
            }
        )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
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
