#!/usr/bin/env python3
"""Benchmark fused top-logprobs selection.

The target boundary is token logprob reporting: select top-N token IDs and
normalized logprobs without materializing a full ``[batch, vocab]`` log-softmax
tensor.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from l20_stack.ops.triton_sampling import (
    logprob_topk_launch_config,
    top_logprobs,
    top_logprobs_out,
    top_logprobs_reference,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--block-vocab", type=int)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(samples: list[float]) -> dict[str, object]:
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[round(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[round(0.90 * (len(ordered) - 1))],
        "samples_ms": samples,
    }


def time_gpu(fn, warmup: int, rounds: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    return samples


def time_cpu_wall(fn, warmup: int, rounds: int) -> list[float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(113)
    logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=dtype)
    config = logprob_topk_launch_config(
        args.vocab,
        args.top_n,
        batch=args.batch,
        block_vocab_override=args.block_vocab,
    )
    output_values = torch.empty((args.batch, args.top_n), device="cuda", dtype=torch.float32)
    output_tokens = torch.empty((args.batch, args.top_n), device="cuda", dtype=torch.int64)
    partial_shape = (args.batch, config.blocks_per_row, args.top_n)
    partial_values = torch.empty(partial_shape, device="cuda", dtype=torch.float32)
    partial_tokens = torch.empty(partial_shape, device="cuda", dtype=torch.int64)
    partial_max = torch.empty((args.batch, config.blocks_per_row), device="cuda", dtype=torch.float32)
    partial_sum_exp = torch.empty(
        (args.batch, config.blocks_per_row), device="cuda", dtype=torch.float32
    )

    expected_values, expected_tokens = top_logprobs_reference(
        logits,
        top_n=args.top_n,
        temperature=args.temperature,
    )
    actual_values, actual_tokens = top_logprobs(
        logits,
        top_n=args.top_n,
        temperature=args.temperature,
        block_vocab_override=args.block_vocab,
    )
    torch.cuda.synchronize()
    if not torch.equal(actual_tokens.cpu(), expected_tokens.cpu()):
        raise AssertionError(
            f"top-logprobs token mismatch: actual={actual_tokens.cpu()} "
            f"expected={expected_tokens.cpu()}"
        )
    max_abs_error = float(torch.max(torch.abs(actual_values - expected_values)).item())
    if max_abs_error > 5e-3:
        raise AssertionError(f"top-logprobs value mismatch: max_abs_error={max_abs_error}")

    top_logprobs_out(
        logits,
        output_values,
        output_tokens,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        partial_max=partial_max,
        partial_sum_exp=partial_sum_exp,
        top_n=args.top_n,
        temperature=args.temperature,
        block_vocab_override=args.block_vocab,
    )
    torch.cuda.synchronize()
    if not torch.equal(output_tokens.cpu(), expected_tokens.cpu()):
        raise AssertionError("preallocated top-logprobs token mismatch")

    def triton_preallocated():
        top_logprobs_out(
            logits,
            output_values,
            output_tokens,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            partial_max=partial_max,
            partial_sum_exp=partial_sum_exp,
            top_n=args.top_n,
            temperature=args.temperature,
            block_vocab_override=args.block_vocab,
        )

    def torch_logsoftmax_topk():
        return torch.topk(
            torch.log_softmax(logits.float() / args.temperature, dim=-1),
            args.top_n,
            dim=-1,
        )

    def torch_logsumexp_topk():
        scaled = logits.float() / args.temperature
        values, tokens = torch.topk(scaled, args.top_n, dim=-1)
        return values - torch.logsumexp(scaled, dim=-1, keepdim=True), tokens

    result = {
        "schema_version": 1,
        "hardware": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "vocab": args.vocab,
            "top_n": args.top_n,
            "temperature": args.temperature,
            "dtype": args.dtype,
        },
        "launch": config.to_dict(),
        "rounds": args.rounds,
        "warmup": args.warmup,
        "correctness": {
            "tokens_match": True,
            "max_abs_logprob_error": max_abs_error,
        },
        "triton_top_logprobs_preallocated": summarize(
            time_gpu(triton_preallocated, args.warmup, args.rounds)
        ),
        "torch_logsoftmax_then_topk": summarize(
            time_gpu(torch_logsoftmax_topk, args.warmup, args.rounds)
        ),
        "torch_logsumexp_then_topk": summarize(
            time_gpu(torch_logsumexp_topk, args.warmup, args.rounds)
        ),
        "torch_logsumexp_then_topk_wall": summarize(
            time_cpu_wall(torch_logsumexp_topk, max(1, args.warmup // 4), args.rounds)
        ),
    }
    fused = result["triton_top_logprobs_preallocated"]["median_ms"]
    logsoftmax = result["torch_logsoftmax_then_topk"]["median_ms"]
    logsumexp = result["torch_logsumexp_then_topk"]["median_ms"]
    result["speedups"] = {
        "vs_torch_logsoftmax_then_topk": logsoftmax / fused if fused else 0.0,
        "vs_torch_logsumexp_then_topk": logsumexp / fused if fused else 0.0,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
