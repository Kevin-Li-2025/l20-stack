#!/usr/bin/env python3
"""Benchmark sparse-history penalty + top-k/top-p sampling.

This is the serving-shaped successor to the dense-count prototype. vLLM stores
prior tokens sparsely per request, so this benchmark uses
``history_tokens[batch, max_history]`` plus lengths instead of a dense
``[batch, vocab]`` count matrix.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from l20_stack.ops.triton_sampling import (
    apply_dense_token_penalties_reference,
    topk_topp_penalty_sample_from_uniform_out,
    topk_topp_sample_from_uniform_out,
    topk_topp_sampling_launch_config,
    topk_topp_sparse_penalty_sample_from_uniform_out,
    topk_topp_sparse_penalty_sample_from_uniform_reference,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--frequency-penalty", type=float, default=0.1)
    parser.add_argument("--presence-penalty", type=float, default=0.1)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--history-tokens", type=int, default=128)
    parser.add_argument("--max-history", type=int, default=128)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=40)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--block-vocab", type=int)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def summarize(samples):
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p10_ms": ordered[round(0.10 * (len(ordered) - 1))],
        "p90_ms": ordered[round(0.90 * (len(ordered) - 1))],
        "samples_ms": samples,
    }


def time_gpu(fn, warmup, rounds):
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


def time_cpu(fn, warmup, rounds):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(rounds):
        torch.cuda.synchronize()
        started = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - started) * 1000)
    return samples


def make_sparse_history(batch: int, vocab: int, history_tokens: int, max_history: int):
    if history_tokens > max_history:
        raise ValueError("history_tokens cannot exceed max_history")
    history = torch.full((batch, max_history), vocab, device="cuda", dtype=torch.int64)
    if history_tokens > 0:
        history[:, :history_tokens] = torch.randint(
            0, vocab, (batch, history_tokens), device="cuda"
        )
    lengths = torch.full((batch,), history_tokens, device="cuda", dtype=torch.int32)
    return history, lengths


def make_dense_counts(history_tokens: torch.Tensor, history_lengths: torch.Tensor, vocab: int):
    batch, max_history = history_tokens.shape
    counts = torch.zeros((batch, vocab), device=history_tokens.device, dtype=torch.int16)
    for row in range(batch):
        length = min(int(history_lengths[row].item()), max_history)
        if length <= 0:
            continue
        tokens = history_tokens[row, :length]
        tokens = tokens[(tokens >= 0) & (tokens < vocab)]
        if tokens.numel() == 0:
            continue
        ones = torch.ones_like(tokens, dtype=counts.dtype)
        counts[row].scatter_add_(0, tokens, ones)
    return counts


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.max_history > 256:
        raise ValueError("max_history above 256 is outside the sparse v1 gate")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(113)
    logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=dtype)
    history_tokens, history_lengths = make_sparse_history(
        args.batch, args.vocab, args.history_tokens, args.max_history
    )
    dense_counts = make_dense_counts(history_tokens, history_lengths, args.vocab)
    uniforms = torch.rand((args.batch,), device="cuda", dtype=torch.float32)
    frequency = torch.full(
        (args.batch,), args.frequency_penalty, device="cuda", dtype=torch.float32
    )
    presence = torch.full(
        (args.batch,), args.presence_penalty, device="cuda", dtype=torch.float32
    )
    repetition = torch.full(
        (args.batch,), args.repetition_penalty, device="cuda", dtype=torch.float32
    )
    config = topk_topp_sampling_launch_config(
        args.vocab,
        args.top_k,
        batch=args.batch,
        block_vocab_override=args.block_vocab,
    )
    partial_shape = (args.batch, config.blocks_per_row, args.top_k)
    partial_values = torch.empty(partial_shape, device="cuda", dtype=torch.float32)
    partial_tokens = torch.empty(partial_shape, device="cuda", dtype=torch.int64)
    output = torch.empty((args.batch,), device="cuda", dtype=torch.int64)
    adjusted_logits = torch.empty_like(logits, dtype=torch.float32)

    expected = topk_topp_sparse_penalty_sample_from_uniform_reference(
        logits,
        history_tokens,
        history_lengths,
        uniforms,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        frequency_penalties=frequency,
        presence_penalties=presence,
        repetition_penalties=repetition,
    )
    topk_topp_sparse_penalty_sample_from_uniform_out(
        logits,
        history_tokens,
        history_lengths,
        uniforms,
        output,
        adjusted_logits=adjusted_logits,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        frequency_penalties=frequency,
        presence_penalties=presence,
        repetition_penalties=repetition,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        block_vocab_override=args.block_vocab,
    )
    torch.cuda.synchronize()
    if not torch.equal(output.cpu(), expected.cpu()):
        raise AssertionError(
            f"sparse penalty sampler mismatch: actual={output.cpu()} expected={expected.cpu()}"
        )

    def sparse_path():
        topk_topp_sparse_penalty_sample_from_uniform_out(
            logits,
            history_tokens,
            history_lengths,
            uniforms,
            output,
            adjusted_logits=adjusted_logits,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            frequency_penalties=frequency,
            presence_penalties=presence,
            repetition_penalties=repetition,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            block_vocab_override=args.block_vocab,
        )

    def dense_count_path():
        topk_topp_penalty_sample_from_uniform_out(
            logits,
            dense_counts,
            uniforms,
            output,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            frequency_penalty=args.frequency_penalty,
            presence_penalty=args.presence_penalty,
            repetition_penalty=args.repetition_penalty,
            block_vocab_override=args.block_vocab,
        )

    def apply_then_sample_path():
        adjusted_logits.copy_(
            apply_dense_token_penalties_reference(
                logits,
                dense_counts,
                frequency_penalty=args.frequency_penalty,
                presence_penalty=args.presence_penalty,
                repetition_penalty=args.repetition_penalty,
            )
        )
        topk_topp_sample_from_uniform_out(
            adjusted_logits,
            uniforms,
            output,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            top_k=args.top_k,
            top_p=args.top_p,
            temperature=args.temperature,
            block_vocab_override=args.block_vocab,
        )

    result = {
        "schema_version": 1,
        "stage": "sparse_history_penalty_sampling_microbenchmark",
        "hardware": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "vocab": args.vocab,
            "top_k": args.top_k,
            "top_p": args.top_p,
            "temperature": args.temperature,
            "dtype": args.dtype,
            "history_tokens": args.history_tokens,
            "max_history": args.max_history,
        },
        "penalties": {
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "repetition_penalty": args.repetition_penalty,
        },
        "launch": config.to_dict(),
        "rounds": args.rounds,
        "warmup": args.warmup,
        "sparse_history_penalty_topk_topp": summarize(
            time_gpu(sparse_path, args.warmup, args.rounds)
        ),
        "dense_count_penalty_topk_topp": summarize(
            time_gpu(dense_count_path, args.warmup, args.rounds)
        ),
        "dense_apply_penalty_then_topk_topp": summarize(
            time_gpu(apply_then_sample_path, args.warmup, args.rounds)
        ),
        "cpu_reference": summarize(
            time_cpu(
                lambda: topk_topp_sparse_penalty_sample_from_uniform_reference(
                    logits.cpu(),
                    history_tokens.cpu(),
                    history_lengths.cpu(),
                    uniforms.cpu(),
                    top_k=args.top_k,
                    top_p=args.top_p,
                    temperature=args.temperature,
                    frequency_penalties=frequency.cpu(),
                    presence_penalties=presence.cpu(),
                    repetition_penalties=repetition.cpu(),
                ),
                max(1, args.warmup // 10),
                max(3, args.rounds // 10),
            )
        ),
    }
    sparse = result["sparse_history_penalty_topk_topp"]["median_ms"]
    dense = result["dense_count_penalty_topk_topp"]["median_ms"]
    baseline = result["dense_apply_penalty_then_topk_topp"]["median_ms"]
    result["speedups"] = {
        "vs_dense_count_fused": dense / sparse if sparse > 0 else None,
        "vs_apply_then_sample": baseline / sparse if sparse > 0 else None,
    }

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
