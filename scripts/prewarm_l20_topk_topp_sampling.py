#!/usr/bin/env python3
"""Prewarm the L20 top-k/top-p Triton sampler compile cache."""

from __future__ import annotations

import argparse
import json
import traceback

import torch

from l20_stack.ops.triton_sampling import topk_topp_sample_from_uniform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=torch.float16)
        uniforms = torch.rand((args.batch,), device="cuda", dtype=torch.float32)
        output = topk_topp_sample_from_uniform(
            logits,
            uniforms,
            top_k=args.top_k,
            top_p=args.top_p,
        )
        torch.cuda.synchronize()
        result = {
            "schema_version": 1,
            "hardware": torch.cuda.get_device_name(),
            "output_shape": list(output.shape),
            "output_dtype": str(output.dtype),
            "status": "ok",
        }
    except Exception as error:
        result = {
            "schema_version": 1,
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback_tail": traceback.format_exc().splitlines()[-40:],
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
