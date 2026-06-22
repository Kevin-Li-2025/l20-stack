#!/usr/bin/env python3
"""Minimal correctness target suitable for compute-sanitizer memcheck."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def reference(query, key_cache, value_cache, table, seq_lens):
    scale = query.shape[-1] ** -0.5
    outputs = []
    for batch in range(query.shape[0]):
        length = int(seq_lens[batch])
        physical = table[batch, torch.arange(length, device="cuda") // 16]
        offsets = torch.arange(length, device="cuda") % 16
        keys = key_cache[physical, offsets]
        values = value_cache[physical, offsets]
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
    parser.add_argument("--context", type=int, default=129)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = Path("/tmp/l20-paged-op-smoke")
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

    torch.manual_seed(17)
    pages_per_request = (args.context + 15) // 16
    pages = args.batch * pages_per_request
    table = torch.randperm(pages, device="cuda", dtype=torch.int32).reshape(
        args.batch, pages_per_request
    )
    seq_lens = torch.full(
        (args.batch,), args.context, device="cuda", dtype=torch.int32
    )
    query = torch.randn(args.batch, 12, 128, device="cuda", dtype=torch.float16)
    key_cache = torch.randn(pages, 16, 2, 128, device="cuda", dtype=torch.float16)
    value_cache = torch.randn_like(key_cache)
    splits = (args.context + 63) // 64
    partial = torch.empty(
        args.batch, 12, splits, 128, device="cuda", dtype=torch.float16
    )
    maxima = torch.empty(args.batch, 12, splits, device="cuda", dtype=torch.float32)
    sums = torch.empty_like(maxima)
    output = torch.empty_like(query)
    actual = torch.ops.l20_stack.paged_decode_split_out(
        query,
        key_cache,
        value_cache,
        table,
        seq_lens,
        partial,
        maxima,
        sums,
        output,
        args.context,
        64,
    )
    torch.cuda.synchronize()
    expected = reference(query, key_cache, value_cache, table, seq_lens)
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    print(
        {
            "gpu": torch.cuda.get_device_name(),
            "batch": args.batch,
            "context": args.context,
            "max_abs_error": float((actual.float() - expected.float()).abs().max()),
        }
    )


if __name__ == "__main__":
    main()
