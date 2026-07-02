#!/usr/bin/env python3
"""Benchmark producer-side LM-head sparse-penalty sampling.

This compares the current post-logits sparse penalty path against the new
producer-side LM-head tile path:

1. materialize ``hidden @ weight.T``, copy/apply sparse penalties, then argmax;
2. compute LM-head vocab tiles, apply sparse penalties in tile, and reduce.

The producer-side path is a correctness/profiling boundary first. It is not a
serving win claim until a vLLM A/B proves the same path in decode.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from l20_stack.ops.triton_lm_head_sampling import (
    lm_head_sample_out,
    lm_head_sampling_launch_config,
)
from l20_stack.ops.triton_sampling import _copy_and_apply_sparse_token_penalties_out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--max-history", type=int, default=32)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--block-vocab", type=int, default=None)
    parser.add_argument("--block-hidden", type=int, default=None)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def percentile(values, pct):
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round((pct / 100) * (len(ordered) - 1))))
    return ordered[index]


def summarize(samples):
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": percentile(samples, 10),
        "p90_ms": percentile(samples, 90),
        "samples_ms": samples,
    }


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


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.batch <= 0 or args.hidden <= 0 or args.vocab <= 0:
        raise ValueError("batch, hidden, and vocab must be positive")
    if args.max_history <= 0 or args.max_history > 256:
        raise ValueError("max-history must be in [1, 256]")

    dtype = getattr(torch, args.dtype)
    torch.manual_seed(2026)
    hidden = torch.randn((args.batch, args.hidden), device="cuda", dtype=dtype)
    weight = torch.randn((args.vocab, args.hidden), device="cuda", dtype=dtype)
    history_tokens = torch.randint(
        low=0,
        high=args.vocab,
        size=(args.batch, args.max_history),
        device="cuda",
        dtype=torch.int64,
    )
    history_lengths = torch.full(
        (args.batch,),
        args.max_history,
        device="cuda",
        dtype=torch.int64,
    )
    frequency = torch.full((args.batch,), 0.05, device="cuda", dtype=torch.float32)
    presence = torch.full((args.batch,), 0.10, device="cuda", dtype=torch.float32)
    repetition = torch.full((args.batch,), 1.10, device="cuda", dtype=torch.float32)

    config = lm_head_sampling_launch_config(
        args.batch,
        args.vocab,
        args.hidden,
        block_vocab=args.block_vocab,
        block_hidden=args.block_hidden,
    )
    baseline_adjusted = torch.empty((args.batch, args.vocab), device="cuda", dtype=torch.float32)
    output_values = torch.empty((args.batch,), device="cuda", dtype=torch.float32)
    output_tokens = torch.empty((args.batch,), device="cuda", dtype=torch.int64)
    partial_values = torch.empty(
        (args.batch, config.blocks_per_row),
        device="cuda",
        dtype=torch.float32,
    )
    partial_tokens = torch.empty(
        (args.batch, config.blocks_per_row),
        device="cuda",
        dtype=torch.int64,
    )

    def baseline():
        logits = hidden @ weight.T
        _copy_and_apply_sparse_token_penalties_out(
            logits,
            history_tokens,
            history_lengths,
            baseline_adjusted,
            frequency_penalties=frequency,
            presence_penalties=presence,
            repetition_penalties=repetition,
        )
        return baseline_adjusted.max(dim=-1)

    def producer_side():
        return lm_head_sample_out(
            hidden,
            weight,
            output_values,
            output_tokens,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            history_tokens=history_tokens,
            history_lengths=history_lengths,
            frequency_penalties=frequency,
            presence_penalties=presence,
            repetition_penalties=repetition,
            use_gumbel=False,
            block_vocab=args.block_vocab,
            block_hidden=args.block_hidden,
        )

    ref_values, ref_tokens = baseline()
    producer_values, producer_tokens = producer_side()
    torch.cuda.synchronize()
    if not torch.equal(ref_tokens.cpu(), producer_tokens.cpu()):
        raise AssertionError("producer-side tokens differ from baseline")
    if not torch.allclose(ref_values.cpu(), producer_values.cpu(), atol=6e-2, rtol=1e-3):
        max_err = (ref_values.cpu() - producer_values.cpu()).abs().max().item()
        raise AssertionError(f"producer-side values differ: max_err={max_err}")

    baseline_samples = time_gpu(baseline, args.warmup, args.rounds)
    producer_samples = time_gpu(producer_side, args.warmup, args.rounds)
    result = {
        "schema_version": 1,
        "hardware": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "max_history": args.max_history,
            "dtype": args.dtype,
        },
        "launch": config.to_dict(),
        "bytes": {
            "materialized_logits_bytes": args.batch
            * args.vocab
            * torch.empty((), dtype=dtype).element_size(),
            "adjusted_logits_bytes": args.batch
            * args.vocab
            * torch.empty((), dtype=torch.float32).element_size(),
            "weight_bytes": args.vocab
            * args.hidden
            * torch.empty((), dtype=dtype).element_size(),
        },
        "baseline_full_logits_sparse_penalty_argmax": summarize(baseline_samples),
        "producer_lm_head_sparse_penalty_argmax": summarize(producer_samples),
    }
    result["ratios"] = {
        "producer_over_baseline": (
            result["producer_lm_head_sparse_penalty_argmax"]["median_ms"]
            / result["baseline_full_logits_sparse_penalty_argmax"]["median_ms"]
        ),
        "baseline_over_producer": (
            result["baseline_full_logits_sparse_penalty_argmax"]["median_ms"]
            / result["producer_lm_head_sparse_penalty_argmax"]["median_ms"]
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
