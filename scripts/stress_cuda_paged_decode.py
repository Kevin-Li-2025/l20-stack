#!/usr/bin/env python3
"""Stress correctness and CUDA Graph capture for the SM89 paged decoder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.cpp_extension import load


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cases", type=int, default=100)
    parser.add_argument("--graph-replays", type=int, default=1000)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    build = Path("/tmp/l20-paged-stress")
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
    import flashinfer

    torch.manual_seed(31)
    failures = []
    max_error = 0.0
    for case in range(args.cases):
        batch = (1, 2, 4)[case % 3]
        max_seq = 65 + (case * 137) % 2239
        seq_lens = torch.randint(
            max(1, max_seq - 127),
            max_seq + 1,
            (batch,),
            device="cuda",
            dtype=torch.int32,
        )
        page_size = 16
        pages_per_sequence = (max_seq + page_size - 1) // page_size
        num_pages = batch * pages_per_sequence
        block_table = torch.randperm(
            num_pages, device="cuda", dtype=torch.int32
        ).reshape(batch, pages_per_sequence)
        pages_per_request = torch.div(
            seq_lens + page_size - 1, page_size, rounding_mode="floor"
        )
        indptr = torch.cat(
            (
                torch.zeros(1, device="cuda", dtype=torch.int32),
                pages_per_request.cumsum(0).to(torch.int32),
            )
        )
        indices = torch.cat(
            [
                block_table[index, : int(pages_per_request[index])]
                for index in range(batch)
            ]
        ).to(torch.int32)
        last_page_len = seq_lens % page_size
        last_page_len = torch.where(
            last_page_len == 0,
            torch.full_like(last_page_len, page_size),
            last_page_len,
        )
        query = torch.randn(batch, 16, 128, device="cuda", dtype=torch.float16)
        cache = (
            torch.randn(
                num_pages, page_size, 8, 128, device="cuda", dtype=torch.float16
            ),
            torch.randn(
                num_pages, page_size, 8, 128, device="cuda", dtype=torch.float16
            ),
        )
        fi_workspace = torch.empty(
            128 * 1024 * 1024, device="cuda", dtype=torch.uint8
        )
        wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            fi_workspace, "NHD", use_tensor_cores=False
        )
        wrapper.plan(
            indptr,
            indices,
            last_page_len,
            16,
            8,
            128,
            page_size,
            pos_encoding_mode="NONE",
            q_data_type=query.dtype,
            kv_data_type=query.dtype,
        )
        expected = torch.empty_like(query)
        wrapper.run(query, cache, out=expected)
        splits = (max_seq + 63) // 64
        partial = torch.empty(
            batch, 16, splits, 128, device="cuda", dtype=query.dtype
        )
        maxima = torch.empty(batch, 16, splits, device="cuda", dtype=torch.float32)
        sums = torch.empty_like(maxima)
        actual = torch.empty_like(query)
        extension.paged_decode_split_out(
            query,
            cache[0],
            cache[1],
            block_table,
            seq_lens,
            partial,
            maxima,
            sums,
            actual,
            max_seq,
            64,
        )
        torch.cuda.synchronize()
        error = float((actual.float() - expected.float()).abs().max())
        max_error = max(max_error, error)
        if not torch.allclose(actual, expected, rtol=2e-2, atol=2e-2):
            failures.append({"case": case, "batch": batch, "max_seq": max_seq, "error": error})

    # Capture a production-gate boundary shape with fixed addresses.
    batch, max_seq = 1, 2304
    pages = (max_seq + 15) // 16
    query = torch.randn(batch, 16, 128, device="cuda", dtype=torch.float16)
    cache = (
        torch.randn(pages, 16, 8, 128, device="cuda", dtype=torch.float16),
        torch.randn(pages, 16, 8, 128, device="cuda", dtype=torch.float16),
    )
    table = torch.arange(pages, device="cuda", dtype=torch.int32).reshape(1, pages)
    lengths = torch.full((1,), max_seq, device="cuda", dtype=torch.int32)
    splits = (max_seq + 63) // 64
    partial = torch.empty(1, 16, splits, 128, device="cuda", dtype=query.dtype)
    maxima = torch.empty(1, 16, splits, device="cuda", dtype=torch.float32)
    sums = torch.empty_like(maxima)
    output = torch.empty_like(query)

    def launch():
        extension.paged_decode_split_out(
            query,
            cache[0],
            cache[1],
            table,
            lengths,
            partial,
            maxima,
            sums,
            output,
            max_seq,
            64,
        )

    for _ in range(3):
        launch()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    for _ in range(args.graph_replays):
        graph.replay()
    torch.cuda.synchronize()
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "cases": args.cases,
        "failed": len(failures),
        "failures": failures[:10],
        "max_abs_error": max_error,
        "graph_capture": True,
        "graph_replays": args.graph_replays,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n")
    raise SystemExit(1 if failures else 0)


if __name__ == "__main__":
    main()
