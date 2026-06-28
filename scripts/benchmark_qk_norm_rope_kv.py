#!/usr/bin/env python3
"""Benchmark L20 Q/K norm + RoPE + KV write fusion against vLLM."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 8, 16, 32, 64],
        help="Token counts to benchmark. Use a single value for deterministic NCU capture.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import triton
    from vllm import _custom_ops
    try:
        from vllm.v1.attention.ops.l20_qk_norm_rope_kv import (
            l20_qk_norm_rope_and_cache,
        )
        import_source = "vllm"
    except ModuleNotFoundError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "integrations/vllm"))
        from l20_qk_norm_rope_kv import l20_qk_norm_rope_and_cache

        import_source = "repo"

    q_heads, kv_heads, head_dim = 16, 8, 128
    reports = []
    for tokens in args.tokens:
        if tokens <= 0:
            raise ValueError(f"tokens must be positive, got {tokens}")
        torch.manual_seed(tokens)
        width = (q_heads + 2 * kv_heads) * head_dim
        source = torch.randn(tokens, width, device="cuda", dtype=torch.bfloat16)
        positions = torch.arange(tokens, device="cuda", dtype=torch.int64)
        angles = torch.randn(2048, head_dim // 2, device="cuda")
        cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
        q_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
        k_weight = torch.randn(head_dim, device="cuda", dtype=torch.bfloat16)
        blocks = max(8, (tokens + 15) // 16)
        key_cache = torch.zeros(
            blocks, 16, kv_heads, head_dim, device="cuda", dtype=torch.bfloat16
        )
        value_cache = torch.zeros_like(key_cache)
        slots = torch.arange(tokens, device="cuda", dtype=torch.int64)
        scale = torch.ones(1, device="cuda")

        expected = source.clone()
        _custom_ops.fused_qk_norm_rope(
            expected,
            q_heads,
            kv_heads,
            kv_heads,
            head_dim,
            1e-6,
            q_weight,
            k_weight,
            cos_sin,
            True,
            positions,
        )
        q_size = q_heads * head_dim
        kv_size = kv_heads * head_dim
        _, expected_key, expected_value = expected.split(
            [q_size, kv_size, kv_size], dim=-1
        )
        expected_key = expected_key.view(tokens, kv_heads, head_dim)
        expected_value = expected_value.view(tokens, kv_heads, head_dim)
        expected_k_cache = key_cache.clone()
        expected_v_cache = value_cache.clone()
        _custom_ops.reshape_and_cache_flash(
            expected_key,
            expected_value,
            expected_k_cache,
            expected_v_cache,
            slots,
            "auto",
            scale,
            scale,
        )

        actual = source.clone()
        l20_qk_norm_rope_and_cache(
            actual,
            positions,
            cos_sin,
            q_weight,
            k_weight,
            key_cache,
            value_cache,
            slots,
            num_q_heads=q_heads,
            num_kv_heads=kv_heads,
            eps=1e-6,
        )
        torch.cuda.synchronize()
        atol = 2 * torch.finfo(torch.bfloat16).eps
        correct = (
            torch.allclose(actual, expected, rtol=0.0, atol=atol)
            and torch.allclose(key_cache, expected_k_cache, rtol=0.0, atol=atol)
            and torch.equal(value_cache, expected_v_cache)
        )
        diagnostics = {
            "qkv_max_abs_error": float(
                (actual.float() - expected.float()).abs().max().item()
            ),
            "key_cache_max_abs_error": float(
                (key_cache.float() - expected_k_cache.float()).abs().max().item()
            ),
            "value_cache_equal": torch.equal(value_cache, expected_v_cache),
        }

        baseline_samples = []
        fused_samples = []
        for _ in range(args.rounds):
            baseline_qkv = source.clone()
            fused_qkv = source.clone()

            def baseline():
                _custom_ops.fused_qk_norm_rope(
                    baseline_qkv,
                    q_heads,
                    kv_heads,
                    kv_heads,
                    head_dim,
                    1e-6,
                    q_weight,
                    k_weight,
                    cos_sin,
                    True,
                    positions,
                )
                _, key, value = baseline_qkv.split(
                    [q_size, kv_size, kv_size], dim=-1
                )
                _custom_ops.reshape_and_cache_flash(
                    key.view(tokens, kv_heads, head_dim),
                    value.view(tokens, kv_heads, head_dim),
                    key_cache,
                    value_cache,
                    slots,
                    "auto",
                    scale,
                    scale,
                )

            def fused():
                l20_qk_norm_rope_and_cache(
                    fused_qkv,
                    positions,
                    cos_sin,
                    q_weight,
                    k_weight,
                    key_cache,
                    value_cache,
                    slots,
                    num_q_heads=q_heads,
                    num_kv_heads=kv_heads,
                    eps=1e-6,
                )

            baseline_samples.append(
                triton.testing.do_bench(baseline, warmup=100, rep=500)
            )
            fused_samples.append(
                triton.testing.do_bench(fused, warmup=100, rep=500)
            )
        baseline_ms = statistics.median(baseline_samples)
        fused_ms = statistics.median(fused_samples)
        reports.append(
            {
                "tokens": tokens,
                "correct": correct,
                "baseline_ms": baseline_ms,
                "fused_ms": fused_ms,
                "speedup": baseline_ms / fused_ms,
                "diagnostics": diagnostics,
            }
        )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "import_source": import_source,
        "tokens_requested": args.tokens,
        "reports": reports,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if all(report["correct"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
