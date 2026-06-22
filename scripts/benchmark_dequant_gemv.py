#!/usr/bin/env python3
"""Benchmark fused INT4 dequant + decode GEMV on L20."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--rounds", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import torch
    import triton
    from l20_stack.ops.triton_dequant_gemv import (
        dequantize_int4_reference,
        int4_groupwise_gemv,
    )

    reports = []
    for n, k in ((1024, 1024), (3072, 1024), (1024, 3072), (4096, 4096)):
        torch.manual_seed(n + k)
        x = torch.randn(k, device="cuda", dtype=torch.float16)
        packed = torch.randint(
            0, 256, (n, k // 2), device="cuda", dtype=torch.uint8
        )
        scales = torch.rand(n, k // 128, device="cuda", dtype=torch.float16)
        expected_weight = dequantize_int4_reference(packed, scales)
        expected = torch.mv(expected_weight, x.float()).to(x.dtype)
        actual = int4_groupwise_gemv(x, packed, scales)
        torch.cuda.synchronize()
        correct = torch.allclose(actual, expected, rtol=2e-2, atol=2e-1)
        baseline_samples = []
        fused_samples = []
        for _ in range(args.rounds):
            baseline_samples.append(
                triton.testing.do_bench(
                    lambda: torch.mv(
                        dequantize_int4_reference(packed, scales), x.float()
                    ),
                    warmup=50,
                    rep=200,
                )
            )
            fused_samples.append(
                triton.testing.do_bench(
                    lambda: int4_groupwise_gemv(x, packed, scales),
                    warmup=50,
                    rep=200,
                )
            )
        baseline_ms = statistics.median(baseline_samples)
        fused_ms = statistics.median(fused_samples)
        reports.append(
            {
                "n": n,
                "k": k,
                "correct": correct,
                "max_abs_error": float(
                    (actual.float() - expected.float()).abs().max().item()
                ),
                "baseline_ms": baseline_ms,
                "fused_ms": fused_ms,
                "speedup": baseline_ms / fused_ms,
            }
        )
    result = {"schema_version": 1, "gpu": torch.cuda.get_device_name(), "reports": reports}
    serialized = json.dumps(result, indent=2, sort_keys=True)
    print(serialized)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized + "\n", encoding="utf-8")
    return 0 if all(report["correct"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
