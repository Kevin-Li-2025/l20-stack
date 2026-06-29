#!/usr/bin/env python3
"""Benchmark the L20 FlashSampling-style LM-head boundary.

This compares full-logits greedy sampling against the experimental LM-head
sampling primitive that computes tile-local candidates without materializing a
full `[batch, vocab]` logits tensor.  The candidate is intentionally limited to
safe greedy / full-vocabulary Gumbel-max semantics; top-k/top-p and production
serving semantics stay on the baseline path.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from l20_stack.epilogue.flash_sampling import (
    FlashSamplingRequest,
    plan_flash_sampling_epilogue,
)
from l20_stack.ops.triton_lm_head_sampling import (
    lm_head_sample_out,
    lm_head_sampling_launch_config,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--sampling-mode", choices=("greedy", "gumbel"), default="greedy")
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--block-vocab", type=int, default=None)
    parser.add_argument("--block-hidden", type=int, default=None)
    parser.add_argument("--include-candidate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


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


def dtype_bytes(dtype_name: str) -> int:
    return {"float16": 2, "bfloat16": 2}[dtype_name]


def base_result(args, cuda_available: bool):
    request = FlashSamplingRequest(
        batch_size=args.batch,
        vocab_size=args.vocab,
        hidden_size=args.hidden,
        sampling_mode=args.sampling_mode,
        top_k=args.top_k,
        top_p=args.top_p,
    )
    decision = plan_flash_sampling_epilogue(request)
    element_size = dtype_bytes(args.dtype)
    result = {
        "schema_version": 1,
        "hardware": torch.cuda.get_device_name() if torch is not None and cuda_available else "no_cuda",
        "cuda_available": cuda_available,
        "shape": {
            "batch": args.batch,
            "hidden": args.hidden,
            "vocab": args.vocab,
            "dtype": args.dtype,
            "sampling_mode": args.sampling_mode,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
        },
        "bytes": {
            "materialized_logits_bytes": args.batch * args.vocab * element_size,
            "weight_bytes": args.vocab * args.hidden * element_size,
            "hidden_bytes": args.batch * args.hidden * element_size,
        },
        "gate": decision.to_dict(),
        "paths": {
            "full_logits_reference": {"status": "pending"},
            "l20_lm_head_sampling_candidate": {"status": "not_requested"},
        },
    }
    if decision.policy is not None:
        result["candidate_policy"] = decision.policy.to_dict()
    return result


def full_logits_reference(hidden, weight, mode: str, temperature: float):
    logits = hidden @ weight.T
    if temperature != 1.0:
        logits = logits / temperature
    logits = logits.float()
    if mode == "greedy":
        return torch.max(logits, dim=-1)
    uniforms = torch.rand(logits.shape, device=logits.device, dtype=torch.float32)
    uniforms = uniforms.clamp_(min=1e-6, max=1.0 - 1e-6)
    gumbels = -torch.log(-torch.log(uniforms))
    return torch.max(logits + gumbels, dim=-1)


def run_cuda(args, result):
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(2026)
    hidden = torch.randn((args.batch, args.hidden), device="cuda", dtype=dtype)
    weight = torch.randn((args.vocab, args.hidden), device="cuda", dtype=dtype)

    reference = full_logits_reference(hidden, weight, args.sampling_mode, args.temperature)
    result["paths"]["full_logits_reference"] = summarize(
        time_gpu(
            lambda: full_logits_reference(hidden, weight, args.sampling_mode, args.temperature),
            args.warmup,
            args.rounds,
        )
    )

    if not args.include_candidate:
        return result
    if not result["gate"]["eligible"]:
        result["paths"]["l20_lm_head_sampling_candidate"] = {
            "status": "skipped_gate",
            "reasons": result["gate"]["reasons"],
        }
        return result
    if args.sampling_mode not in {"greedy", "gumbel"}:
        result["paths"]["l20_lm_head_sampling_candidate"] = {
            "status": "unsupported_sampling_mode"
        }
        return result

    config = lm_head_sampling_launch_config(
        args.batch,
        args.vocab,
        args.hidden,
        block_vocab=args.block_vocab,
        block_hidden=args.block_hidden,
    )
    output_values = torch.empty((args.batch,), device="cuda", dtype=torch.float32)
    output_tokens = torch.empty((args.batch,), device="cuda", dtype=torch.int64)
    partial_values = torch.empty(
        (args.batch, config.blocks_per_row), device="cuda", dtype=torch.float32
    )
    partial_tokens = torch.empty(
        (args.batch, config.blocks_per_row), device="cuda", dtype=torch.int64
    )
    use_gumbel = args.sampling_mode == "gumbel"
    seeds = torch.arange(2026, 2026 + args.batch, device="cuda", dtype=torch.int64)

    lm_head_sample_out(
        hidden,
        weight,
        output_values,
        output_tokens,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        seeds=seeds,
        use_gumbel=use_gumbel,
        temperature=args.temperature,
        block_vocab=args.block_vocab,
        block_hidden=args.block_hidden,
    )
    torch.cuda.synchronize()
    if not use_gumbel:
        ref_values, ref_tokens = reference
        if not torch.equal(output_tokens.cpu(), ref_tokens.cpu()):
            raise AssertionError("candidate greedy tokens differ from full logits reference")
        if not torch.allclose(output_values, ref_values.float(), atol=5e-2, rtol=1e-3):
            max_err = (output_values - ref_values.float()).abs().max().item()
            raise AssertionError(f"candidate greedy values differ: max_err={max_err}")

    candidate = summarize(
        time_gpu(
            lambda: lm_head_sample_out(
                hidden,
                weight,
                output_values,
                output_tokens,
                partial_values=partial_values,
                partial_tokens=partial_tokens,
                seeds=seeds,
                use_gumbel=use_gumbel,
                temperature=args.temperature,
                block_vocab=args.block_vocab,
                block_hidden=args.block_hidden,
            ),
            args.warmup,
            args.rounds,
        )
    )
    candidate["status"] = "measured"
    candidate["launch"] = config.to_dict()
    result["paths"]["l20_lm_head_sampling_candidate"] = candidate
    result["ratios"] = {
        "candidate_over_full_logits_reference": (
            candidate["median_ms"]
            / result["paths"]["full_logits_reference"]["median_ms"]
        ),
        "full_logits_reference_over_candidate": (
            result["paths"]["full_logits_reference"]["median_ms"]
            / candidate["median_ms"]
        ),
    }
    return result


def main() -> int:
    args = parse_args()
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    result = base_result(args, cuda_available)
    if args.dry_run:
        result["paths"]["full_logits_reference"] = {"status": "dry_run"}
        if args.include_candidate:
            result["paths"]["l20_lm_head_sampling_candidate"] = {"status": "dry_run"}
    elif torch is None:
        result["paths"]["full_logits_reference"] = {"status": "not_run_no_torch"}
        if args.include_candidate:
            result["paths"]["l20_lm_head_sampling_candidate"] = {"status": "not_run_no_torch"}
    elif not cuda_available:
        result["paths"]["full_logits_reference"] = {"status": "not_run_no_cuda"}
        if args.include_candidate:
            result["paths"]["l20_lm_head_sampling_candidate"] = {"status": "not_run_no_cuda"}
    else:
        result = run_cuda(args, result)

    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
