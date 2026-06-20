#!/usr/bin/env python3
"""Benchmark fused RoPE + KV-cache writes on an NVIDIA L20."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from l20_stack.operators import OperatorShape, rope_kv_minimum_bytes
from l20_stack.ops.triton_rope_kv import (
    rope_kv_cache_write_triton,
    rope_kv_reference,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--cache-tokens", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--cache-flush-mb", type=int, default=256)
    parser.add_argument("--require-l20", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values, pct):
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * pct))
    return ordered[index]


def cuda_event_timings(torch, function, reset, warmup, iterations, cache_flush):
    for _ in range(warmup):
        reset()
        if cache_flush is not None:
            cache_flush.zero_()
        function()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    for start, end in zip(starts, ends):
        reset()
        if cache_flush is not None:
            cache_flush.zero_()
        start.record()
        function()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def correctness(torch, actual, expected, dtype):
    atol = 1e-5 if dtype == torch.float32 else 5e-3
    correct = True
    max_abs = 0.0
    max_rel = 0.0
    for actual_tensor, expected_tensor in zip(actual, expected):
        difference = (actual_tensor.float() - expected_tensor.float()).abs()
        max_abs = max(max_abs, difference.max().item())
        relative = difference / expected_tensor.float().abs().clamp_min(1e-6)
        max_rel = max(max_rel, relative.max().item())
        correct = correct and torch.allclose(
            actual_tensor.float(), expected_tensor.float(), atol=atol, rtol=1e-3
        )
    return {
        "correct": bool(correct),
        "max_abs_error": max_abs,
        "max_rel_error": max_rel,
        "atol": atol,
        "rtol": 1e-3,
    }


def timing_report(timings_ms, minimum_bytes):
    p50 = percentile(timings_ms, 0.50)
    return {
        "timing_ms": {
            "p50": round(p50, 4),
            "p95": round(percentile(timings_ms, 0.95), 4),
            "mean": round(statistics.mean(timings_ms), 4),
        },
        "minimum_effective_gbps": round(minimum_bytes / p50 / 1_000_000, 2),
    }


def main() -> int:
    args = parse_args()
    if (
        args.tokens <= 0
        or args.kv_heads <= 0
        or args.head_dim <= 0
        or args.cache_tokens < args.tokens
        or args.warmup < 0
        or args.iters <= 0
        or args.cache_flush_mb < 0
    ):
        raise SystemExit("invalid positive dimension, iteration, or cache setting")

    import torch
    import triton

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark")
    gpu_name = torch.cuda.get_device_name()
    compute_capability = torch.cuda.get_device_capability()
    if args.require_l20 and ("L20" not in gpu_name.upper() or compute_capability != (8, 9)):
        actual_target = f"{gpu_name} sm_{compute_capability[0]}{compute_capability[1]}"
        raise SystemExit(f"--require-l20 expected NVIDIA L20 sm_89, got {actual_target}")

    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    k = torch.randn(args.tokens, args.kv_heads, args.head_dim, device="cuda", dtype=dtype)
    v = torch.randn_like(k)
    half = args.head_dim // 2
    angles = torch.randn(args.tokens, half, device="cuda", dtype=torch.float32)
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    cache_positions = torch.arange(args.tokens, device="cuda", dtype=torch.int64)
    cache_flush = None
    if args.cache_flush_mb:
        cache_flush = torch.empty(
            args.cache_flush_mb * 1024 * 1024, device="cuda", dtype=torch.uint8
        )

    expected_k_cache = torch.zeros(
        args.cache_tokens, args.kv_heads, args.head_dim, device="cuda", dtype=dtype
    )
    expected_v_cache = torch.zeros_like(expected_k_cache)
    expected = rope_kv_reference(
        k, v, cos, sin, cache_positions, expected_k_cache, expected_v_cache
    )

    shape = OperatorShape(args.tokens * args.kv_heads, args.head_dim, k.element_size())
    fused_bytes = rope_kv_minimum_bytes(shape, fused=True)
    unfused_bytes = rope_kv_minimum_bytes(shape, fused=False)

    def separate_reset():
        separate_k_cache.zero_()
        separate_v_cache.zero_()

    def separate_provider():
        return rope_kv_reference(
            k, v, cos, sin, cache_positions, separate_k_cache, separate_v_cache
        )

    separate_k_cache = torch.zeros_like(expected_k_cache)
    separate_v_cache = torch.zeros_like(expected_v_cache)
    providers = {"torch_separate": (separate_provider, separate_reset, unfused_bytes)}

    def triton_reset():
        triton_k_cache.zero_()
        triton_v_cache.zero_()

    def triton_provider():
        return rope_kv_cache_write_triton(
            k, v, cos, sin, cache_positions, triton_k_cache, triton_v_cache
        )

    triton_k_cache = torch.zeros_like(expected_k_cache)
    triton_v_cache = torch.zeros_like(expected_v_cache)
    providers["triton_fused"] = (triton_provider, triton_reset, fused_bytes)

    provider_reports = {}
    for name, (provider, reset, minimum_bytes) in providers.items():
        reset()
        actual = provider()
        torch.cuda.synchronize()
        report = correctness(torch, actual, expected, dtype)
        if report["correct"]:
            timings = cuda_event_timings(
                torch, provider, reset, args.warmup, args.iters, cache_flush
            )
            report.update(timing_report(timings, minimum_bytes))
        provider_reports[name] = report

    baseline = provider_reports["torch_separate"].get("timing_ms", {}).get("p50")
    if baseline:
        for report in provider_reports.values():
            provider_p50 = report.get("timing_ms", {}).get("p50")
            if provider_p50:
                report["speedup_vs_torch_separate"] = round(baseline / provider_p50, 3)

    report = {
        "benchmark_version": 1,
        "gpu_name": gpu_name,
        "compute_capability": f"{compute_capability[0]}.{compute_capability[1]}",
        "cuda": torch.version.cuda,
        "torch": torch.__version__,
        "triton": triton.__version__,
        "shape": {
            "tokens": args.tokens,
            "kv_heads": args.kv_heads,
            "head_dim": args.head_dim,
            "dtype": args.dtype,
        },
        "fused_minimum_bytes": fused_bytes,
        "unfused_minimum_bytes": unfused_bytes,
        "minimum_traffic_reduction_pct": round(
            100 * (unfused_bytes - fused_bytes) / unfused_bytes, 2
        ),
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iters,
        "cache_flush_mb": args.cache_flush_mb,
        "providers": provider_reports,
        "all_correct": all(provider["correct"] for provider in provider_reports.values()),
        "note": (
            "The fused provider applies RoPE to K and writes K/V cache in one Triton "
            "launch. The baseline materializes rotated K before assigning K/V cache."
        ),
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if report["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
