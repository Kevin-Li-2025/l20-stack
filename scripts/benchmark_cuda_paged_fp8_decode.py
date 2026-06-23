#!/usr/bin/env python3
"""Benchmark the CUDA FP8 E4M3 paged-decode path on L20."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def latency_ms(function, warmup=10, iterations=50):
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


def quantize_fp8_e4m3(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = max(float(tensor.float().abs().max()) / finfo.max, 1.0e-12)
    quantized = (tensor.float() / scale).clamp(finfo.min, finfo.max).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def allocate_workspace(batch, q_heads, context, split_size, dtype=torch.float16):
    num_splits = (context + split_size - 1) // split_size
    partial = torch.empty(batch, q_heads, num_splits, 128, device="cuda", dtype=dtype)
    maxima = torch.empty(batch, q_heads, num_splits, device="cuda", dtype=torch.float32)
    sums = torch.empty_like(maxima)
    return partial, maxima, sums


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp/l20-paged-fp8-cuda"))
    parser.add_argument("--batches", default="1,4,8")
    parser.add_argument("--contexts", default="512,2048,4096")
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--split-size", type=int, default=64)
    args = parser.parse_args()
    if args.q_heads % args.kv_heads:
        raise SystemExit("q-heads must be divisible by kv-heads")
    if not hasattr(torch, "float8_e4m3fn"):
        raise SystemExit("torch.float8_e4m3fn is required")

    root = Path(__file__).resolve().parents[1]
    args.build_dir.mkdir(parents=True, exist_ok=True)
    extension = load(
        "l20_paged_decode_cuda",
        [
            root / "integrations/vllm/cuda/l20_paged_decode.cpp",
            root / "integrations/vllm/cuda/l20_paged_decode.cu",
        ],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_89,code=sm_89"],
        build_directory=args.build_dir,
    )
    torch.ops.load_library(extension.__file__)

    reports = []
    for batch in [int(item) for item in args.batches.split(",") if item]:
        for context in [int(item) for item in args.contexts.split(",") if item]:
            torch.manual_seed(1000 + batch * 17 + context)
            page_size = 16
            pages_per_request = (context + page_size - 1) // page_size
            pages = batch * pages_per_request
            block_table = torch.randperm(
                pages, device="cuda", dtype=torch.int32
            ).reshape(batch, pages_per_request)
            seq_lens = torch.full((batch,), context, device="cuda", dtype=torch.int32)
            query = torch.randn(
                batch, args.q_heads, 128, device="cuda", dtype=torch.float16
            )
            key_bf16 = torch.randn(
                pages,
                page_size,
                args.kv_heads,
                128,
                device="cuda",
                dtype=torch.bfloat16,
            )
            value_bf16 = torch.randn_like(key_bf16)
            key_fp8, k_scale = quantize_fp8_e4m3(key_bf16)
            value_fp8, v_scale = quantize_fp8_e4m3(value_bf16)
            key_dequant = (key_fp8.float() * k_scale).to(torch.float16)
            value_dequant = (value_fp8.float() * v_scale).to(torch.float16)

            bf16_workspace = allocate_workspace(
                batch, args.q_heads, context, args.split_size
            )
            fp8_workspace = allocate_workspace(
                batch, args.q_heads, context, args.split_size
            )
            materialized_workspace = allocate_workspace(
                batch, args.q_heads, context, args.split_size
            )
            bf16_output = torch.empty_like(query)
            fp8_output = torch.empty_like(query)
            materialized_output = torch.empty_like(query)

            extension.paged_decode_split_out(
                query,
                key_dequant,
                value_dequant,
                block_table,
                seq_lens,
                *bf16_workspace,
                bf16_output,
                context,
                args.split_size,
            )
            extension.paged_decode_fp8_e4m3_split_out(
                query,
                key_fp8,
                value_fp8,
                block_table,
                seq_lens,
                *fp8_workspace,
                fp8_output,
                k_scale,
                v_scale,
                context,
                args.split_size,
            )
            torch.cuda.synchronize()
            max_abs_error = float((fp8_output.float() - bf16_output.float()).abs().max())

            bf16_ms = latency_ms(
                lambda: extension.paged_decode_split_out(
                    query,
                    key_dequant,
                    value_dequant,
                    block_table,
                    seq_lens,
                    *bf16_workspace,
                    bf16_output,
                    context,
                    args.split_size,
                )
            )
            fp8_ms = latency_ms(
                lambda: extension.paged_decode_fp8_e4m3_split_out(
                    query,
                    key_fp8,
                    value_fp8,
                    block_table,
                    seq_lens,
                    *fp8_workspace,
                    fp8_output,
                    k_scale,
                    v_scale,
                    context,
                    args.split_size,
                )
            )
            materialized_ms = latency_ms(
                lambda: extension.paged_decode_split_out(
                    query,
                    (key_fp8.float() * k_scale).to(torch.float16),
                    (value_fp8.float() * v_scale).to(torch.float16),
                    block_table,
                    seq_lens,
                    *materialized_workspace,
                    materialized_output,
                    context,
                    args.split_size,
                )
            )
            reports.append(
                {
                    "batch": batch,
                    "context": context,
                    "q_heads": args.q_heads,
                    "kv_heads": args.kv_heads,
                    "split_size": args.split_size,
                    "correct": bool(torch.allclose(fp8_output, bf16_output, rtol=2e-2, atol=2e-2)),
                    "max_abs_error": max_abs_error,
                    "latency_ms": {
                        "cuda_bf16_predequantized": bf16_ms,
                        "cuda_fp8_fused_dequant": fp8_ms,
                        "cuda_fp8_materialize_then_bf16": materialized_ms,
                    },
                    "ratios": {
                        "fp8_fused_vs_bf16_predequantized": bf16_ms / fp8_ms,
                        "fp8_fused_vs_materialized": materialized_ms / fp8_ms,
                    },
                }
            )

    payload = {"gpu": torch.cuda.get_device_name(), "reports": reports}
    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
