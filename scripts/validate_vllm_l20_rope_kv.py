#!/usr/bin/env python3
"""Validate the vLLM L20 RoPE/KV kernel across upstream-relevant shapes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--tokens",
        type=int,
        nargs="+",
        default=[1, 8, 17, 64],
        help="Token counts to validate.",
    )
    return parser.parse_args()


def mismatch(torch, actual, expected, atol):
    unequal = (actual.float() - expected.float()).abs() > atol
    if not bool(unequal.any()):
        return None
    coordinate = unequal.nonzero()[0].tolist()
    index = tuple(coordinate)
    return {
        "coordinate": coordinate,
        "actual": float(actual[index].float().item()),
        "expected": float(expected[index].float().item()),
        "mismatch_count": int(unequal.sum().item()),
    }


def validate_case(torch, apply_rope, fused, case):
    torch.manual_seed(case["seed"])
    tokens = case["tokens"]
    q_heads = case["q_heads"]
    kv_heads = case["kv_heads"]
    head_dim = case["head_dim"]
    dtype = getattr(torch, case["dtype"])
    query = torch.randn(tokens, q_heads, head_dim, device="cuda", dtype=dtype)
    key = torch.randn(tokens, kv_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    expected_query = query.clone()
    expected_key = key.clone()
    positions = torch.randperm(2048, device="cuda")[:tokens].long()
    angles = torch.randn(2048, head_dim // 2, device="cuda")
    cos_sin = torch.cat((angles.cos(), angles.sin()), dim=-1)
    apply_rope(
        positions,
        expected_query,
        expected_key,
        head_dim,
        cos_sin,
        case["is_neox"],
    )

    cache_blocks = max(8, (tokens + 7) // 8)
    block_size = 16
    key_cache = torch.randn(
        cache_blocks, block_size, kv_heads, head_dim, device="cuda", dtype=dtype
    )
    value_cache = torch.randn_like(key_cache)
    expected_key_cache = key_cache.clone()
    expected_value_cache = value_cache.clone()
    slots = torch.randperm(cache_blocks * block_size, device="cuda")[:tokens].long()
    if case["invalid_slots"]:
        slots[::5] = -1
    valid = slots >= 0
    expected_key_cache[
        torch.div(slots[valid], block_size, rounding_mode="floor"),
        slots[valid] % block_size,
    ] = expected_key[valid]
    expected_value_cache[
        torch.div(slots[valid], block_size, rounding_mode="floor"),
        slots[valid] % block_size,
    ] = value[valid]

    fused(
        query,
        key,
        value,
        positions,
        cos_sin,
        case["is_neox"],
        key_cache,
        value_cache,
        slots,
    )
    torch.cuda.synchronize()
    atol = 2 * torch.finfo(dtype).eps
    comparisons = {
        "query": torch.allclose(query, expected_query, rtol=0.0, atol=atol),
        "key": torch.allclose(key, expected_key, rtol=0.0, atol=atol),
        "key_cache": torch.allclose(
            key_cache, expected_key_cache, rtol=0.0, atol=atol
        ),
        "value_cache": torch.equal(value_cache, expected_value_cache),
    }
    mismatches = {
        name: detail
        for name, detail in (
            ("query", mismatch(torch, query, expected_query, atol)),
            ("key", mismatch(torch, key, expected_key, atol)),
            (
                "key_cache",
                mismatch(torch, key_cache, expected_key_cache, atol),
            ),
            ("value_cache", mismatch(torch, value_cache, expected_value_cache, 0.0)),
        )
        if detail is not None
    }
    return case | {
        "correct": all(comparisons.values()),
        "comparisons": comparisons,
        "atol": atol,
        "mismatches": mismatches,
        "query_max_abs_error": float(
            (query.float() - expected_query.float()).abs().max().item()
        ),
        "key_max_abs_error": float(
            (key.float() - expected_key.float()).abs().max().item()
        ),
    }


def main() -> int:
    args = parse_args()
    import torch
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
    from vllm.v1.attention.ops.l20_rope_kv import l20_rope_and_cache

    if torch.cuda.get_device_name() != "NVIDIA L20":
        raise SystemExit("validation requires NVIDIA L20")
    cases = []
    seed = 0
    for dtype in ("float16", "bfloat16"):
        for is_neox in (True, False):
            for tokens in args.tokens:
                for q_heads, kv_heads, head_dim in (
                    (14, 2, 64),
                    (12, 2, 128),
                    (32, 4, 128),
                    (32, 8, 128),
                    (16, 4, 256),
                ):
                    cases.append(
                        {
                            "seed": seed,
                            "dtype": dtype,
                            "is_neox": is_neox,
                            "tokens": tokens,
                            "q_heads": q_heads,
                            "kv_heads": kv_heads,
                            "head_dim": head_dim,
                            "invalid_slots": tokens > 1,
                        }
                    )
                    seed += 1
    reports = [
        validate_case(
            torch,
            apply_rope_with_cos_sin_cache_inplace,
            l20_rope_and_cache,
            case,
        )
        for case in cases
    ]
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability())),
        "cases": len(reports),
        "passed": sum(report["correct"] for report in reports),
        "all_correct": all(report["correct"] for report in reports),
        "reports": reports,
    }
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if result["all_correct"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
