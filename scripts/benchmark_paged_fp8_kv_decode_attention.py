#!/usr/bin/env python3
"""Benchmark L20 paged FP8 KV fused-dequant decode attention."""

from __future__ import annotations

import argparse
import json
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


def quantize_fp8_e4m3(tensor):
    finfo = torch.finfo(torch.float8_e4m3fn)
    scale = max(float(tensor.float().abs().max()) / finfo.max, 1e-6)
    quantized = torch.clamp(tensor.float() / scale, finfo.min, finfo.max).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    args = parser.parse_args()

    import flashinfer
    from integrations.vllm.l20_paged_split_kv import (
        allocate_l20_paged_split_kv_workspace,
        l20_paged_split_kv_attention,
        l20_paged_split_kv_attention_fp8,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not hasattr(torch, "float8_e4m3fn"):
        raise RuntimeError("torch.float8_e4m3fn is required")

    reports = []
    torch.manual_seed(293)
    for batch in (1, 4, 8):
        for context in (2048, 4096):
            page_size = 16
            pages_per_sequence = context // page_size
            num_pages = batch * pages_per_sequence
            block_table = torch.randperm(
                num_pages, device="cuda", dtype=torch.int32
            ).reshape(batch, pages_per_sequence)
            indptr = (
                torch.arange(batch + 1, device="cuda", dtype=torch.int32)
                * pages_per_sequence
            )
            indices = block_table.flatten()
            last_page_len = torch.full(
                (batch,), page_size, device="cuda", dtype=torch.int32
            )
            seq_lens = torch.full(
                (batch,), context, device="cuda", dtype=torch.int32
            )
            query = torch.randn(
                batch, 16, 128, device="cuda", dtype=torch.bfloat16
            )
            key_bf16 = torch.randn(
                num_pages, page_size, 8, 128, device="cuda", dtype=torch.bfloat16
            )
            value_bf16 = torch.randn_like(key_bf16)
            key_fp8, k_scale = quantize_fp8_e4m3(key_bf16)
            value_fp8, v_scale = quantize_fp8_e4m3(value_bf16)
            key_dequant = (key_fp8.float() * k_scale).to(torch.bfloat16)
            value_dequant = (value_fp8.float() * v_scale).to(torch.bfloat16)

            flashinfer_workspace = torch.empty(
                128 * 1024 * 1024, device="cuda", dtype=torch.uint8
            )
            wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
                flashinfer_workspace, "NHD"
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
            expected = wrapper.run(query, (key_dequant, value_dequant))
            bf16_workspace = allocate_l20_paged_split_kv_workspace(query, context)
            fp8_workspace = allocate_l20_paged_split_kv_workspace(query, context)
            materialized_workspace = allocate_l20_paged_split_kv_workspace(
                query, context
            )
            bf16_actual = l20_paged_split_kv_attention(
                query,
                key_bf16,
                value_bf16,
                block_table,
                seq_lens,
                workspace=bf16_workspace,
            )
            fp8_predequantized = l20_paged_split_kv_attention(
                query,
                key_dequant,
                value_dequant,
                block_table,
                seq_lens,
                workspace=materialized_workspace,
            )
            fp8_fused = l20_paged_split_kv_attention_fp8(
                query,
                key_fp8,
                value_fp8,
                block_table,
                seq_lens,
                k_scale=k_scale,
                v_scale=v_scale,
                workspace=fp8_workspace,
            )

            flashinfer_ms = latency_ms(
                lambda: wrapper.run(query, (key_dequant, value_dequant)),
                args.warmup,
                args.iterations,
            )
            bf16_ms = latency_ms(
                lambda: l20_paged_split_kv_attention(
                    query,
                    key_bf16,
                    value_bf16,
                    block_table,
                    seq_lens,
                    workspace=bf16_workspace,
                ),
                args.warmup,
                args.iterations,
            )
            fp8_predequantized_ms = latency_ms(
                lambda: l20_paged_split_kv_attention(
                    query,
                    key_dequant,
                    value_dequant,
                    block_table,
                    seq_lens,
                    workspace=materialized_workspace,
                ),
                args.warmup,
                args.iterations,
            )
            fp8_materialized_ms = latency_ms(
                lambda: l20_paged_split_kv_attention(
                    query,
                    (key_fp8.float() * k_scale).to(torch.bfloat16),
                    (value_fp8.float() * v_scale).to(torch.bfloat16),
                    block_table,
                    seq_lens,
                    workspace=materialized_workspace,
                ),
                args.warmup,
                args.iterations,
            )
            fp8_fused_ms = latency_ms(
                lambda: l20_paged_split_kv_attention_fp8(
                    query,
                    key_fp8,
                    value_fp8,
                    block_table,
                    seq_lens,
                    k_scale=k_scale,
                    v_scale=v_scale,
                    workspace=fp8_workspace,
                ),
                args.warmup,
                args.iterations,
            )

            reports.append(
                {
                    "batch": batch,
                    "context": context,
                    "correctness": {
                        "bf16_vs_flashinfer_dequant_reference": bool(
                            torch.allclose(bf16_actual, expected, rtol=2e-2, atol=2e-2)
                        ),
                        "fp8_predequantized_vs_flashinfer_dequant_reference": bool(
                            torch.allclose(
                                fp8_predequantized,
                                expected,
                                rtol=2e-2,
                                atol=2e-2,
                            )
                        ),
                        "fp8_fused_vs_flashinfer_dequant_reference": bool(
                            torch.allclose(fp8_fused, expected, rtol=2e-2, atol=2e-2)
                        ),
                        "fp8_fused_max_abs_error": float(
                            (fp8_fused.float() - expected.float()).abs().max()
                        ),
                    },
                    "latency_ms": {
                        "flashinfer_bf16_on_dequant_kv": flashinfer_ms,
                        "l20_bf16_paged": bf16_ms,
                        "l20_fp8_predequantized_paged": fp8_predequantized_ms,
                        "l20_fp8_materialize_dequant_then_paged": fp8_materialized_ms,
                        "l20_fp8_fused_dequant_paged": fp8_fused_ms,
                    },
                    "ratios": {
                        "fused_fp8_vs_flashinfer_bf16_dequant": (
                            flashinfer_ms / fp8_fused_ms
                        ),
                        "fused_fp8_vs_l20_bf16": bf16_ms / fp8_fused_ms,
                        "fused_fp8_vs_predequantized_fp8": (
                            fp8_predequantized_ms / fp8_fused_ms
                        ),
                        "fused_fp8_vs_materialized_fp8": (
                            fp8_materialized_ms / fp8_fused_ms
                        ),
                    },
                    "scales": {"k": k_scale, "v": v_scale},
                }
            )

    result = {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(),
        "benchmark": "paged NHD split-KV decode attention; FP8 E4M3 KV scalar dequant",
        "reports": reports,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
