#!/usr/bin/env python3
"""Validate and benchmark the L20 AWQ GEMV against vLLM AWQ kernels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def latency_ms(function, warmup=30, iterations=200):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from vllm import _custom_ops
    from vllm.model_executor.layers.quantization.l20_awq_gemv import l20_awq_gemv

    torch.manual_seed(13)
    reports = []
    for tokens in (1, 4, 8):
        for k, n in ((1024, 3072), (3072, 1024), (4096, 4096)):
            x = torch.randn(tokens, k, device="cuda", dtype=torch.float16)
            qweight = torch.randint(
                -(2**31), 2**31 - 1, (k, n // 8), device="cuda", dtype=torch.int32
            )
            qzeros = torch.randint(
                -(2**31),
                2**31 - 1,
                (k // 128, n // 8),
                device="cuda",
                dtype=torch.int32,
            )
            scales = (
                torch.rand(k // 128, n, device="cuda", dtype=torch.float16) * 0.02
            )
            expected_weight = _custom_ops.awq_dequantize(
                qweight, scales, qzeros, 0, 0, 0
            )
            expected = torch.matmul(x, expected_weight)
            actual = l20_awq_gemv(x, qweight, scales, qzeros, 128)
            reports.append(
                {
                    "tokens": tokens,
                    "k": k,
                    "n": n,
                    "correct": bool(
                        torch.allclose(actual, expected, rtol=2e-2, atol=2e-1)
                    ),
                    "max_abs_error": float(
                        (actual.float() - expected.float()).abs().max()
                    ),
                    "awq_gemm_ms": latency_ms(
                        lambda: _custom_ops.awq_gemm(
                            x, qweight, scales, qzeros, 8
                        )
                    ),
                    "l20_gemv_ms": latency_ms(
                        lambda: l20_awq_gemv(x, qweight, scales, qzeros, 128)
                    ),
                }
            )
            reports[-1]["speedup"] = (
                reports[-1]["awq_gemm_ms"] / reports[-1]["l20_gemv_ms"]
            )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
