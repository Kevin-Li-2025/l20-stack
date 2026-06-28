#!/usr/bin/env python3
"""Classify Nsight Systems CUDA kernels and APIs into serving families."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KERNEL_FAMILIES = (
    "custom_l20",
    "flashinfer_sampling",
    "sampler_other",
    "flashinfer_attention",
    "cutlass_or_cublas_gemm",
    "cublas_gemv",
    "triton_generated",
    "pytorch_fill",
    "pytorch_softmax",
    "pytorch_elementwise",
    "other",
)

API_FAMILIES = (
    "sync",
    "memcpy",
    "launch",
    "graph",
    "alloc_free",
    "library_load",
    "memory_info",
    "other",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--top-rows", type=int, default=8)
    return parser.parse_args()


def parse_number(value) -> float:
    text = str(value or "").strip().replace(",", "")
    if not text or text.lower() in {"n/a", "nan", "none"}:
        return 0.0
    try:
        return float(text)
    except ValueError:
        return 0.0


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return [
            row
            for row in csv.DictReader(line for line in handle if not line.startswith("#"))
        ]


def first_key(row: dict[str, str], names: tuple[str, ...]) -> str | None:
    lowered = {key.lower().strip(): key for key in row}
    for name in names:
        key = lowered.get(name.lower())
        if key is not None:
            return key
    return None


def numeric(row: dict[str, str], names: tuple[str, ...]) -> float:
    key = first_key(row, names)
    return parse_number(row.get(key)) if key is not None else 0.0


def text_value(row: dict[str, str], names: tuple[str, ...]) -> str:
    key = first_key(row, names)
    return row.get(key, "") if key is not None else ""


def normalize_row(row: dict[str, str]) -> dict:
    total_time_ns = numeric(row, ("Total Time (ns)", "Total Time", "Time (ns)"))
    instances = numeric(row, ("Instances", "Num Calls", "Calls", "Count"))
    avg_ns = numeric(row, ("Avg (ns)", "Average (ns)", "Avg"))
    if not total_time_ns and avg_ns and instances:
        total_time_ns = avg_ns * instances
    return {
        "name": text_value(row, ("Name", "Kernel Name", "Range", "Operation")),
        "total_time_ns": total_time_ns,
        "instances": int(instances) if instances else 0,
        "avg_ns": avg_ns,
        "reported_time_pct": numeric(row, ("Time (%)", "% Time", "Total Time (%)")),
    }


def find_report_csv(input_dir: Path, report: str) -> Path:
    candidates = [
        input_dir / f"{report}.csv",
        input_dir / f"{report}_{report}.csv",
    ]
    candidates.extend(sorted(input_dir.glob(f"*{report}*.csv")))
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"missing Nsight Systems CSV report: {report}")


def classify_kernel(name: str) -> str:
    if "_l20_" in name or "l20_" in name:
        return "custom_l20"
    if "flashinfer::sampling::" in name:
        return "flashinfer_sampling"
    if (
        "_topk_topp_kernel" in name
        or "_topk_log_softmax_kernel" in name
        or "_gumbel_sample_kernel" in name
        or "_temperature_kernel" in name
        or "_min_p_kernel" in name
        or "_penalties_kernel" in name
        or "topk" in name.lower()
        or "topp" in name.lower()
        or "sampling" in name.lower()
    ):
        return "sampler_other"
    if "flashinfer::" in name:
        return "flashinfer_attention"
    if "gemvx::kernel" in name:
        return "cublas_gemv"
    if (
        "cutlass::Kernel" in name
        or "ampere_fp16" in name
        or "cublasLt::" in name
        or "s168" in name
    ):
        return "cutlass_or_cublas_gemm"
    if name.startswith("triton_"):
        return "triton_generated"
    if "FillFunctor" in name:
        return "pytorch_fill"
    if "SoftMax" in name or "softmax" in name:
        return "pytorch_softmax"
    if "at::native" in name or "elementwise" in name:
        return "pytorch_elementwise"
    return "other"


def classify_api(name: str) -> str:
    if "Synchronize" in name:
        return "sync"
    if "Memcpy" in name or "Memset" in name:
        return "memcpy"
    if "Graph" in name:
        return "graph"
    if "Launch" in name:
        return "launch"
    if "Malloc" in name or "Free" in name or "HostAlloc" in name:
        return "alloc_free"
    if "LibraryLoad" in name or "ModuleLoad" in name:
        return "library_load"
    if "MemGetInfo" in name:
        return "memory_info"
    return "other"


def aggregate(rows: list[dict], families: tuple[str, ...], classifier) -> dict:
    total_time_ns = sum(row["total_time_ns"] for row in rows)
    by_family = {
        family: {
            "family": family,
            "total_time_ns": 0.0,
            "time_pct": 0.0,
            "instances": 0,
            "unique_rows": 0,
            "top_rows": [],
        }
        for family in families
    }
    for row in rows:
        family = classifier(row["name"])
        bucket = by_family[family]
        bucket["total_time_ns"] += row["total_time_ns"]
        bucket["instances"] += row["instances"]
        bucket["unique_rows"] += 1
        bucket["top_rows"].append(row)
    for bucket in by_family.values():
        bucket["time_pct"] = (
            100.0 * bucket["total_time_ns"] / total_time_ns if total_time_ns else 0.0
        )
        bucket["top_rows"] = sorted(
            bucket["top_rows"], key=lambda row: row["total_time_ns"], reverse=True
        )
    return {
        "total_time_ns": total_time_ns,
        "families": by_family,
        "ordered_families": sorted(
            by_family.values(), key=lambda row: row["total_time_ns"], reverse=True
        ),
    }


def trim_top_rows(aggregate_result: dict, limit: int) -> None:
    for bucket in aggregate_result["families"].values():
        bucket["top_rows"] = bucket["top_rows"][:limit]


def render_markdown(summary: dict) -> str:
    lines = [
        "# Nsight Systems Kernel Family Summary",
        "",
        f"Source: `{summary['source_dir']}`",
        "",
        "## GPU Kernel Families",
        "",
        "| Family | Time share | Total GPU time | Instances | Unique rows |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary["gpu"]["ordered_families"]:
        if row["total_time_ns"] <= 0:
            continue
        lines.append(
            f"| `{row['family']}` | {row['time_pct']:.2f}% | "
            f"{row['total_time_ns'] / 1e6:.3f} ms | {row['instances']} | "
            f"{row['unique_rows']} |"
        )
    lines.extend(
        [
            "",
            "## CUDA API Families",
            "",
            "| Family | Time share | Total API time | Calls | Unique rows |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["api"]["ordered_families"]:
        if row["total_time_ns"] <= 0:
            continue
        lines.append(
            f"| `{row['family']}` | {row['time_pct']:.2f}% | "
            f"{row['total_time_ns'] / 1e6:.3f} ms | {row['instances']} | "
            f"{row['unique_rows']} |"
        )
    lines.extend(["", "## Interpretation", ""])
    lines.append(
        "Use this as a ceiling estimate. A family with a small time share cannot "
        "produce a large end-to-end win unless the change also removes launches, "
        "synchronization, or adjacent work."
    )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir
    kernel_csv = find_report_csv(input_dir, "cuda_gpu_kern_sum")
    api_csv = find_report_csv(input_dir, "cuda_api_sum")
    kernel_rows = [normalize_row(row) for row in read_csv(kernel_csv)]
    api_rows = [normalize_row(row) for row in read_csv(api_csv)]
    gpu = aggregate(kernel_rows, KERNEL_FAMILIES, classify_kernel)
    api = aggregate(api_rows, API_FAMILIES, classify_api)
    trim_top_rows(gpu, args.top_rows)
    trim_top_rows(api, args.top_rows)
    summary = {
        "schema_version": 1,
        "source_dir": str(input_dir),
        "files": {
            "cuda_gpu_kern_sum": str(kernel_csv),
            "cuda_api_sum": str(api_csv),
        },
        "gpu": gpu,
        "api": api,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
