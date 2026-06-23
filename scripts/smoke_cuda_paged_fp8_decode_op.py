#!/usr/bin/env python3
"""Correctness smoke for the CUDA FP8 E4M3 paged-decode path."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def quantize_fp8_e4m3(tensor: torch.Tensor) -> tuple[torch.Tensor, float]:
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = max(float(tensor.float().abs().max()) / finfo.max, 1.0e-12)
    quantized = (tensor.float() / scale).clamp(finfo.min, finfo.max).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def reference(query, key_cache, value_cache, table, seq_lens, k_scale, v_scale):
    key_dequant = (key_cache.float() * k_scale).to(torch.float16)
    value_dequant = (value_cache.float() * v_scale).to(torch.float16)
    scale = query.shape[-1] ** -0.5
    outputs = []
    for batch in range(query.shape[0]):
        length = int(seq_lens[batch])
        token = torch.arange(length, device="cuda")
        physical = table[batch, token // 16]
        offsets = token % 16
        keys = key_dequant[physical, offsets]
        values = value_dequant[physical, offsets]
        kv_heads = keys.shape[1]
        group = query.shape[1] // kv_heads
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        scores = torch.einsum("hd,thd->ht", query[batch].float(), keys.float()) * scale
        probs = scores.softmax(dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", probs, values.float()))
    return torch.stack(outputs).to(query.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=257)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--split-size", type=int, default=128)
    args = parser.parse_args()
    if args.q_heads % args.kv_heads:
        raise SystemExit("q-heads must be divisible by kv-heads")
    if not hasattr(torch, "float8_e4m3fn"):
        raise SystemExit("torch.float8_e4m3fn is required")

    root = Path(__file__).resolve().parents[1]
    build = Path("/tmp/l20-paged-fp8-op-smoke")
    build.mkdir(parents=True, exist_ok=True)
    extension = load(
        "l20_paged_decode_cuda",
        [
            root / "integrations/vllm/cuda/l20_paged_decode.cpp",
            root / "integrations/vllm/cuda/l20_paged_decode.cu",
        ],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_89,code=sm_89"],
        build_directory=build,
    )
    torch.ops.load_library(extension.__file__)

    torch.manual_seed(29)
    pages_per_request = (args.context + 15) // 16
    pages = args.batch * pages_per_request
    table = torch.randperm(pages, device="cuda", dtype=torch.int32).reshape(
        args.batch, pages_per_request
    )
    seq_lens = torch.full(
        (args.batch,), args.context, device="cuda", dtype=torch.int32
    )
    query = torch.randn(
        args.batch, args.q_heads, 128, device="cuda", dtype=torch.float16
    )
    key_bf16 = torch.randn(
        pages, 16, args.kv_heads, 128, device="cuda", dtype=torch.bfloat16
    )
    value_bf16 = torch.randn_like(key_bf16)
    key_cache, k_scale = quantize_fp8_e4m3(key_bf16)
    value_cache, v_scale = quantize_fp8_e4m3(value_bf16)

    splits = (args.context + args.split_size - 1) // args.split_size
    partial = torch.empty(
        args.batch, args.q_heads, splits, 128, device="cuda", dtype=torch.float16
    )
    maxima = torch.empty(args.batch, args.q_heads, splits, device="cuda")
    sums = torch.empty_like(maxima)
    output = torch.empty_like(query)
    actual = torch.ops.l20_stack.paged_decode_fp8_e4m3_split_out(
        query,
        key_cache,
        value_cache,
        table,
        seq_lens,
        partial,
        maxima,
        sums,
        output,
        k_scale,
        v_scale,
        args.context,
        args.split_size,
    )
    torch.cuda.synchronize()
    expected = reference(query, key_cache, value_cache, table, seq_lens, k_scale, v_scale)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    print(
        {
            "gpu": torch.cuda.get_device_name(),
            "batch": args.batch,
            "context": args.context,
            "q_heads": args.q_heads,
            "kv_heads": args.kv_heads,
            "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        }
    )


if __name__ == "__main__":
    main()
