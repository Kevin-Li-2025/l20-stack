#!/usr/bin/env python3
"""Benchmark L20 GPU-side greedy sampling against CPU round-trip sampling."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from l20_stack.ops.triton_sampling import (
    greedy_sample,
    greedy_sample_out,
    greedy_sample_reference,
    greedy_sampling_launch_config,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    parser.add_argument("--block-vocab", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def time_gpu(fn, warmup: int, rounds: int):
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


def time_cpu_roundtrip(logits, warmup: int, rounds: int):
    for _ in range(warmup):
        _ = torch.argmax(logits.cpu(), dim=-1)
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        started = time.perf_counter()
        _ = torch.argmax(logits.cpu(), dim=-1)
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": percentile(samples, 10),
        "p90_ms": percentile(samples, 90),
        "samples_ms": samples,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(20)
    logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=dtype)

    expected = greedy_sample_reference(logits)
    actual = greedy_sample(logits)
    torch.cuda.synchronize()
    if not torch.equal(actual.cpu(), expected.cpu()):
        raise AssertionError("Triton greedy sampler does not match torch.argmax")

    config = greedy_sampling_launch_config(args.vocab, block_vocab_override=args.block_vocab)
    output = torch.empty((args.batch,), device="cuda", dtype=torch.int64)
    partial_values = None
    partial_tokens = None
    if config.strategy == "two_stage_block_argmax":
        partial_values = torch.empty(
            (args.batch, config.blocks_per_row), device="cuda", dtype=torch.float32
        )
        partial_tokens = torch.empty(
            (args.batch, config.blocks_per_row), device="cuda", dtype=torch.int64
        )
    greedy_sample_out(
        logits,
        output,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        block_vocab_override=args.block_vocab,
    )
    torch.cuda.synchronize()
    if not torch.equal(output.cpu(), expected.cpu()):
        raise AssertionError("preallocated Triton greedy sampler does not match torch.argmax")

    triton_samples = time_gpu(lambda: greedy_sample(logits), args.warmup, args.rounds)
    triton_prealloc_samples = time_gpu(
        lambda: greedy_sample_out(
            logits,
            output,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            block_vocab_override=args.block_vocab,
        ),
        args.warmup,
        args.rounds,
    )
    torch_samples = time_gpu(lambda: torch.argmax(logits, dim=-1), args.warmup, args.rounds)
    cpu_samples = time_cpu_roundtrip(logits, args.warmup, args.rounds)

    result = {
        "schema_version": 1,
        "hardware": torch.cuda.get_device_name(),
        "shape": {"batch": args.batch, "vocab": args.vocab, "dtype": args.dtype},
        "launch": config.to_dict(),
        "rounds": args.rounds,
        "warmup": args.warmup,
        "triton_gpu_greedy": summarize(triton_samples),
        "triton_gpu_greedy_preallocated": summarize(triton_prealloc_samples),
        "torch_gpu_argmax": summarize(torch_samples),
        "cpu_roundtrip_argmax": summarize(cpu_samples),
    }
    result["speedups"] = {
        "triton_vs_cpu_roundtrip": (
            result["cpu_roundtrip_argmax"]["median_ms"]
            / result["triton_gpu_greedy"]["median_ms"]
        ),
        "triton_preallocated_vs_cpu_roundtrip": (
            result["cpu_roundtrip_argmax"]["median_ms"]
            / result["triton_gpu_greedy_preallocated"]["median_ms"]
        ),
        "triton_vs_torch_gpu_argmax": (
            result["torch_gpu_argmax"]["median_ms"]
            / result["triton_gpu_greedy"]["median_ms"]
        ),
        "triton_preallocated_vs_torch_gpu_argmax": (
            result["torch_gpu_argmax"]["median_ms"]
            / result["triton_gpu_greedy_preallocated"]["median_ms"]
        ),
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
