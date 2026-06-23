#!/usr/bin/env python3
"""Benchmark fused FP8 KV dequantization inside L20 decode attention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from l20_stack.ops.triton_decode_attention import (
    gqa_decode_attention_fp8_split_kv,
    gqa_decode_attention_split_kv,
)


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


def quantize_fp8_e4m3(tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = max(float(tensor.float().abs().max()) / finfo.max, 1e-6)
    quantized = torch.clamp(tensor.float() / scale, finfo.min, finfo.max).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def dequant_then_attention(query, key_fp8, value_fp8, k_scale, v_scale):
    key_dequant = (key_fp8.float() * k_scale).to(torch.bfloat16)
    value_dequant = (value_fp8.float() * v_scale).to(torch.bfloat16)
    return gqa_decode_attention_split_kv(
        query,
        key_dequant,
        value_dequant,
        split_size=1024,
        block_t=128,
        num_warps=8,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=25)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required")

    torch.manual_seed(109)
    reports = []
    for batch in (1, 4, 16):
        for context in (512, 2048, 4096):
            query = torch.randn(batch, 16, 128, device="cuda", dtype=torch.bfloat16)
            key_bf16 = torch.randn(
                batch, context, 8, 128, device="cuda", dtype=torch.bfloat16
            )
            value_bf16 = torch.randn_like(key_bf16)
            key_fp8, k_scale = quantize_fp8_e4m3(key_bf16)
            value_fp8, v_scale = quantize_fp8_e4m3(value_bf16)
            key_dequant = (key_fp8.float() * k_scale).to(torch.bfloat16)
            value_dequant = (value_fp8.float() * v_scale).to(torch.bfloat16)

            bf16_expected = reference(query, key_bf16, value_bf16)
            fp8_expected = reference(query, key_dequant, value_dequant)
            bf16_actual = gqa_decode_attention_split_kv(
                query, key_bf16, value_bf16, split_size=1024, block_t=128, num_warps=8
            )
            fp8_nonfused = gqa_decode_attention_split_kv(
                query,
                key_dequant,
                value_dequant,
                split_size=1024,
                block_t=128,
                num_warps=8,
            )
            fp8_fused = gqa_decode_attention_fp8_split_kv(
                query,
                key_fp8,
                value_fp8,
                k_scale,
                v_scale,
                split_size=1024,
                block_t=128,
                num_warps=8,
            )

            bf16_ms = latency_ms(
                lambda: gqa_decode_attention_split_kv(
                    query,
                    key_bf16,
                    value_bf16,
                    split_size=1024,
                    block_t=128,
                    num_warps=8,
                ),
                args.warmup,
                args.iterations,
            )
            fp8_nonfused_ms = latency_ms(
                lambda: gqa_decode_attention_split_kv(
                    query,
                    key_dequant,
                    value_dequant,
                    split_size=1024,
                    block_t=128,
                    num_warps=8,
                ),
                args.warmup,
                args.iterations,
            )
            fp8_materialized_ms = latency_ms(
                lambda: dequant_then_attention(
                    query,
                    key_fp8,
                    value_fp8,
                    k_scale,
                    v_scale,
                ),
                args.warmup,
                args.iterations,
            )
            fp8_fused_ms = latency_ms(
                lambda: gqa_decode_attention_fp8_split_kv(
                    query,
                    key_fp8,
                    value_fp8,
                    k_scale,
                    v_scale,
                    split_size=1024,
                    block_t=128,
                    num_warps=8,
                ),
                args.warmup,
                args.iterations,
            )

            reports.append(
                {
                    "batch": batch,
                    "context": context,
                    "dtype": {
                        "query": str(query.dtype),
                        "bf16_kv": str(key_bf16.dtype),
                        "fp8_kv": str(key_fp8.dtype),
                    },
                    "scales": {"k": k_scale, "v": v_scale},
                    "correctness": {
                        "bf16_vs_reference": bool(
                            torch.allclose(bf16_actual, bf16_expected, rtol=2e-2, atol=2e-2)
                        ),
                        "fp8_nonfused_vs_dequant_reference": bool(
                            torch.allclose(
                                fp8_nonfused, fp8_expected, rtol=2e-2, atol=2e-2
                            )
                        ),
                        "fp8_fused_vs_dequant_reference": bool(
                            torch.allclose(fp8_fused, fp8_expected, rtol=2e-2, atol=2e-2)
                        ),
                        "fp8_fused_max_abs_error": float(
                            (fp8_fused.float() - fp8_expected.float()).abs().max()
                        ),
                        "fp8_quantization_max_abs_delta": float(
                            (fp8_expected.float() - bf16_expected.float()).abs().max()
                        ),
                    },
                    "latency_ms": {
                        "bf16_kv": bf16_ms,
                        "fp8_predequantized_attention": fp8_nonfused_ms,
                        "fp8_materialize_dequant_then_attention": fp8_materialized_ms,
                        "fp8_fused_dequant_attention": fp8_fused_ms,
                    },
                    "ratios": {
                        "fused_fp8_vs_bf16": bf16_ms / fp8_fused_ms,
                        "fused_fp8_vs_predequantized_fp8": fp8_nonfused_ms / fp8_fused_ms,
                        "fused_fp8_vs_materialized_fp8": (
                            fp8_materialized_ms / fp8_fused_ms
                        ),
                    },
                }
            )

    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "benchmark": "contiguous split-KV decode attention; FP8 E4M3 KV scalar dequant",
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
