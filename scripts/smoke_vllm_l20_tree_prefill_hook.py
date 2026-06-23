#!/usr/bin/env python3
"""Smoke test the FlashInfer native-prefill L20 tree-attention hook."""

from __future__ import annotations

import argparse
import json
import os
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


def make_paged_prefix(prefix_key, prefix_value, page_size: int = 16):
    cached_length, num_kv_heads, head_dim = prefix_key.shape
    pages = cached_length // page_size
    block_table = torch.randperm(pages, device=prefix_key.device, dtype=torch.int32).reshape(
        1, pages
    )
    kv_cache = torch.empty(
        pages,
        2,
        page_size,
        num_kv_heads,
        head_dim,
        device=prefix_key.device,
        dtype=prefix_key.dtype,
    )
    key_pages = prefix_key.reshape(pages, page_size, num_kv_heads, head_dim)
    value_pages = prefix_value.reshape_as(key_pages)
    kv_cache[block_table[0].long(), 0] = key_pages
    kv_cache[block_table[0].long(), 1] = value_pages
    return kv_cache, block_table


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cached", type=int, default=4096)
    parser.add_argument("--draft", type=int, default=16)
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="float16")
    parser.add_argument("--causal-verifier", action="store_true")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    os.environ.setdefault("VLLM_ENABLE_L20_TREE_ATTENTION", "1")
    from vllm.v1.attention.backends.flashinfer import (
        maybe_run_l20_causal_verifier_from_prefill,
        maybe_run_l20_tree_attention_from_prefill,
    )
    from vllm.v1.attention.ops.l20_tree_attention import (
        make_chain_tree_mask,
        torch_tree_attention_reference,
    )

    torch.manual_seed(67)
    num_q_heads = 16
    num_kv_heads = 8
    head_dim = 128
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    hook = (
        maybe_run_l20_causal_verifier_from_prefill
        if args.causal_verifier
        else maybe_run_l20_tree_attention_from_prefill
    )
    prefill_query = torch.randn(
        args.draft,
        num_q_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    prefix_key = torch.randn(
        args.cached,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    prefix_value = torch.randn_like(prefix_key)
    suffix_key = torch.randn(
        args.draft,
        num_kv_heads,
        head_dim,
        device="cuda",
        dtype=dtype,
    )
    suffix_value = torch.randn_like(suffix_key)
    kv_cache, block_table = make_paged_prefix(prefix_key, prefix_value)
    seq_lens = torch.tensor([args.cached + args.draft], device="cuda", dtype=torch.int32)
    out = torch.empty_like(prefill_query)
    ran = hook(
        prefill_query,
        kv_cache,
        suffix_key,
        suffix_value,
        block_table,
        seq_lens,
        out,
        args.cached + args.draft,
    )
    key = torch.cat([prefix_key, suffix_key], dim=0).unsqueeze(0)
    value = torch.cat([prefix_value, suffix_value], dim=0).unsqueeze(0)
    mask = make_chain_tree_mask(args.draft, device="cuda")
    expected = torch_tree_attention_reference(
        prefill_query.unsqueeze(0),
        key,
        value,
        mask,
        args.cached,
    ).squeeze(0)
    runtime_ms = latency_ms(
        lambda: hook(
            prefill_query,
            kv_cache,
            suffix_key,
            suffix_value,
            block_table,
            seq_lens,
            out,
            args.cached + args.draft,
        ),
        iterations=args.iterations,
    )
    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "hook": hook.__name__,
        "dtype": args.dtype,
        "cached_length": args.cached,
        "draft_length": args.draft,
        "ran": bool(ran),
        "correct": bool(torch.allclose(out, expected, rtol=2e-2, atol=2e-2)),
        "max_abs_error": float((out.float() - expected.float()).abs().max()),
        "hook_ms": runtime_ms,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
