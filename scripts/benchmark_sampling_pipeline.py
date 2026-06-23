#!/usr/bin/env python3
"""Benchmark decode sampling pipelines that motivate fused top-k/top-p kernels."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--vocab", type=int, default=151_936)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--warmup", type=int, default=30)
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


def gpu_topk_multinomial(logits, top_k, temperature):
    values, indices = torch.topk(logits / temperature, k=top_k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    sample = torch.multinomial(probs, num_samples=1)
    return torch.gather(indices, dim=-1, index=sample).squeeze(-1)


def cpu_roundtrip_topk_multinomial(logits, top_k, temperature):
    cpu_logits = logits.cpu()
    values, indices = torch.topk(cpu_logits / temperature, k=top_k, dim=-1)
    probs = torch.softmax(values, dim=-1)
    sample = torch.multinomial(probs, num_samples=1)
    return torch.gather(indices, dim=-1, index=sample).squeeze(-1)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.manual_seed(43)
    logits = torch.randn((args.batch, args.vocab), device="cuda", dtype=torch.float16)
    result = {
        "schema_version": 1,
        "hardware": torch.cuda.get_device_name(),
        "shape": {
            "batch": args.batch,
            "vocab": args.vocab,
            "top_k": args.top_k,
            "temperature": args.temperature,
            "dtype": "float16",
        },
        "gpu_argmax": summarize(
            time_gpu(lambda: torch.argmax(logits, dim=-1), args.warmup, args.rounds)
        ),
        "gpu_topk_softmax_multinomial": summarize(
            time_gpu(
                lambda: gpu_topk_multinomial(logits, args.top_k, args.temperature),
                args.warmup,
                args.rounds,
            )
        ),
        "cpu_roundtrip_topk_softmax_multinomial": summarize(
            time_cpu(
                lambda: cpu_roundtrip_topk_multinomial(logits, args.top_k, args.temperature),
                args.warmup,
                args.rounds,
            )
        ),
    }
    result["ratios"] = {
        "gpu_sampling_pipeline_vs_gpu_argmax": (
            result["gpu_topk_softmax_multinomial"]["median_ms"]
            / result["gpu_argmax"]["median_ms"]
        ),
        "cpu_roundtrip_pipeline_vs_gpu_pipeline": (
            result["cpu_roundtrip_topk_softmax_multinomial"]["median_ms"]
            / result["gpu_topk_softmax_multinomial"]["median_ms"]
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
