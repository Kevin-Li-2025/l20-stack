#!/usr/bin/env python3
"""Benchmark the L20 contiguous GQA decode-attention kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from l20_stack.ops.triton_decode_attention import gqa_decode_attention


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
    torch.manual_seed(7)
    reports = []
    for batch in (1, 8):
        for context in (128, 512, 2048, 4096):
            query = torch.randn(
                batch, 16, 128, device="cuda", dtype=torch.bfloat16
            )
            key = torch.randn(
                batch, context, 8, 128, device="cuda", dtype=torch.bfloat16
            )
            value = torch.randn_like(key)
            expected = reference(query, key, value)
            actual = gqa_decode_attention(query, key, value)
            baseline_ms = latency_ms(lambda: reference(query, key, value))
            fused_ms = latency_ms(
                lambda: gqa_decode_attention(query, key, value)
            )
            reports.append(
                {
                    "batch": batch,
                    "context": context,
                    "correct": bool(
                        torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)
                    ),
                    "max_abs_error": float(
                        (actual.float() - expected.float()).abs().max()
                    ),
                    "baseline_ms": baseline_ms,
                    "fused_ms": fused_ms,
                    "speedup": baseline_ms / fused_ms,
                }
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
