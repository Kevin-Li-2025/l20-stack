#!/usr/bin/env python3
"""Summarize vLLM speculative decoding acceptance evidence from logs/results."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from statistics import median


NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)"
PATTERNS = {
    "acceptance_rate": re.compile(
        rf"(?:acceptance[_ -]?rate|accepted[_ -]?rate)\D+({NUMBER})",
        re.IGNORECASE,
    ),
    "accepted_tokens": re.compile(
        rf"(?:accepted[_ -]?tokens|num[_ -]?accepted[_ -]?tokens)\D+({NUMBER})",
        re.IGNORECASE,
    ),
    "draft_tokens": re.compile(
        rf"(?:draft[_ -]?tokens|num[_ -]?draft[_ -]?tokens|total[_ -]?draft[_ -]?tokens)\D+({NUMBER})",
        re.IGNORECASE,
    ),
}


def collect_numbers(pattern: re.Pattern[str], text: str) -> list[float]:
    values = []
    for match in pattern.finditer(text):
        try:
            values.append(float(match.group(1)))
        except ValueError:
            continue
    return values


def summarize_values(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "min": None, "max": None}
    return {
        "count": len(values),
        "median": median(values),
        "min": min(values),
        "max": max(values),
    }


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    log_text = args.log.read_text(encoding="utf-8", errors="replace")
    log_metrics = {
        name: summarize_values(collect_numbers(pattern, log_text))
        for name, pattern in PATTERNS.items()
    }
    benchmark_reports = []
    for path in sorted(args.result_dir.glob("*.json")):
        if path.name == args.output.name or path.name.endswith("summary.json"):
            continue
        try:
            payload = read_json(path)
        except json.JSONDecodeError:
            continue
        if "completed" not in payload and "request_throughput" not in payload:
            continue
        benchmark_reports.append(
            {
                "path": str(path),
                "completed": payload.get("completed"),
                "failed": payload.get("failed"),
                "request_throughput": payload.get("request_throughput"),
                "output_throughput": payload.get("output_throughput"),
                "median_itl_ms": payload.get("median_itl_ms"),
                "median_ttft_ms": payload.get("median_ttft_ms"),
                "p95_itl_ms": payload.get("p95_itl_ms"),
            }
        )

    accepted = log_metrics["accepted_tokens"]["median"]
    draft = log_metrics["draft_tokens"]["median"]
    inferred_acceptance = None
    if accepted is not None and draft not in (None, 0):
        inferred_acceptance = accepted / draft

    summary = {
        "schema_version": 1,
        "log": str(args.log),
        "result_dir": str(args.result_dir),
        "log_metrics": log_metrics,
        "inferred_acceptance_rate_from_tokens": inferred_acceptance,
        "acceptance_observed": (
            log_metrics["acceptance_rate"]["count"] > 0
            or inferred_acceptance is not None
        ),
        "benchmark_reports": benchmark_reports,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
