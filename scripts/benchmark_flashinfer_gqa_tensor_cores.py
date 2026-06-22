#!/usr/bin/env python3
"""Measure the L20 FlashInfer tensor-core crossover by GQA ratio and batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def latency_ms(function, warmup=20, iterations=100):
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
    import flashinfer

    reports = []
    page_size = 16
    context = 2048
    for batch in (1, 4, 8, 16):
        for ratio in (1, 2, 4, 8):
            kv_heads = 8
            q_heads = kv_heads * ratio
            pages = context // page_size
            num_pages = batch * pages
            block_table = torch.arange(
                num_pages, device="cuda", dtype=torch.int32
            ).reshape(batch, pages)
            indptr = (
                torch.arange(batch + 1, device="cuda", dtype=torch.int32) * pages
            )
            last_page_len = torch.full(
                (batch,), page_size, device="cuda", dtype=torch.int32
            )
            query = torch.randn(
                batch, q_heads, 128, device="cuda", dtype=torch.float16
            )
            cache = (
                torch.randn(
                    num_pages,
                    page_size,
                    kv_heads,
                    128,
                    device="cuda",
                    dtype=torch.float16,
                ),
                torch.randn(
                    num_pages,
                    page_size,
                    kv_heads,
                    128,
                    device="cuda",
                    dtype=torch.float16,
                ),
            )
            timings = {}
            for tensor_cores in (False, True):
                workspace = torch.empty(
                    128 * 1024 * 1024, device="cuda", dtype=torch.uint8
                )
                wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                    workspace, "NHD", use_tensor_cores=tensor_cores
                )
                wrapper.plan(
                    indptr,
                    block_table.flatten(),
                    last_page_len,
                    q_heads,
                    kv_heads,
                    128,
                    page_size,
                    pos_encoding_mode="NONE",
                    q_data_type=query.dtype,
                    kv_data_type=query.dtype,
                )
                timings[tensor_cores] = latency_ms(lambda: wrapper.run(query, cache))
            reports.append(
                {
                    "batch": batch,
                    "gqa_ratio": ratio,
                    "cuda_core_ms": timings[False],
                    "tensor_core_ms": timings[True],
                    "tensor_core_speedup": timings[False] / timings[True],
                }
            )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "context": context,
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
