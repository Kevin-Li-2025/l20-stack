#!/usr/bin/env python3
"""Trigger the L20 split-KV decode-attention partial kernel for Nsight Compute."""

from __future__ import annotations

import argparse

import torch

from l20_stack.ops.triton_decode_attention import (
    gqa_decode_attention_split_kv,
    gqa_decode_attention_split_kv_tensor_core_candidate,
    gqa_decode_attention_split_kv_tensor_core_dsplit_candidate,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--context", type=int, default=4096)
    parser.add_argument("--q-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--split-size", type=int, default=512)
    parser.add_argument("--block-t", type=int, default=32)
    parser.add_argument("--block-q", type=int, default=2)
    parser.add_argument("--block-d", type=int, default=64)
    parser.add_argument("--num-warps", type=int, default=4)
    parser.add_argument(
        "--path",
        choices=["scalar", "tensor-core-candidate", "tensor-core-dsplit-candidate"],
        default="scalar",
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=20)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(37)
    query = torch.randn(
        args.batch,
        args.q_heads,
        args.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    key = torch.randn(
        args.batch,
        args.context,
        args.kv_heads,
        args.head_dim,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    function = gqa_decode_attention_split_kv
    kwargs = {
        "split_size": args.split_size,
        "block_t": args.block_t,
        "num_warps": args.num_warps,
    }
    if args.path == "tensor-core-candidate":
        function = gqa_decode_attention_split_kv_tensor_core_candidate
        kwargs["block_q"] = args.block_q
    elif args.path == "tensor-core-dsplit-candidate":
        function = gqa_decode_attention_split_kv_tensor_core_dsplit_candidate
        kwargs["block_q"] = args.block_q
        kwargs["block_d"] = args.block_d
    for _ in range(args.warmup):
        function(query, key, value, **kwargs)
    torch.cuda.synchronize()
    for _ in range(args.iterations):
        function(query, key, value, **kwargs)
    torch.cuda.synchronize()
    print(
        {
            "path": args.path,
            "batch": args.batch,
            "context": args.context,
            "split_size": args.split_size,
            "block_t": args.block_t,
            "block_q": (
                args.block_q
                if args.path in {"tensor-core-candidate", "tensor-core-dsplit-candidate"}
                else None
            ),
            "block_d": (
                args.block_d if args.path == "tensor-core-dsplit-candidate" else None
            ),
            "num_warps": args.num_warps,
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
