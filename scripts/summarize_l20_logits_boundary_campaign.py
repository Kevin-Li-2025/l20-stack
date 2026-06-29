#!/usr/bin/env python3
"""Summarize a vLLM logits-boundary trace serving campaign."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path


METRICS = (
    "request_throughput",
    "output_throughput",
    "median_ttft_ms",
    "p95_ttft_ms",
    "median_itl_ms",
    "p95_itl_ms",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("campaign_dir", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reports(root: Path) -> dict[tuple[int, int], list[dict]]:
    groups = defaultdict(list)
    pattern = re.compile(r"c(\d+)-i(\d+)-r(\d+)\.json")
    for path in sorted(root.glob("*.json")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        report = load_json(path)
        if report.get("failed") not in (0, None):
            raise SystemExit(f"serving report has failures: {path}")
        groups[(int(match.group(1)), int(match.group(2)))].append(report)
    return groups


def median_metric(rows: list[dict], name: str) -> float:
    return round(statistics.median(row[name] for row in rows), 5)


def summarize(root: Path) -> dict:
    groups = load_reports(root)
    trace_summary_path = root / "logits-boundary-summary.json"
    trace_summary = load_json(trace_summary_path) if trace_summary_path.exists() else {}
    shapes = []
    for (concurrency, input_tokens), rows in sorted(groups.items()):
        shapes.append(
            {
                "max_concurrency": concurrency,
                "input_tokens": input_tokens,
                "runs": len(rows),
                "metrics": {name: median_metric(rows, name) for name in METRICS},
            }
        )
    return {
        "schema_version": 1,
        "campaign_dir": str(root),
        "run_config": load_json(root / "run-config.json")
        if (root / "run-config.json").exists()
        else {},
        "serving_report_count": sum(len(rows) for rows in groups.values()),
        "shapes": shapes,
        "trace_summary": trace_summary,
    }


def render_markdown(summary: dict) -> str:
    trace = summary.get("trace_summary", {})
    lines = [
        "# L20 vLLM Logits Boundary Campaign",
        "",
        f"- Campaign: `{summary['campaign_dir']}`",
        f"- Serving reports: `{summary['serving_report_count']}`",
        f"- Trace events: `{trace.get('total_events', 0)}`",
        f"- Eligible fraction: `{trace.get('eligible_fraction', 0.0):.4f}`",
        f"- Eligible events: `{trace.get('eligible_events', 0)}`",
        f"- Fallback events: `{trace.get('fallback_events', 0)}`",
        f"- Eligible logits materialization: `{trace.get('eligible_logits_mib', 0.0):.2f} MiB`",
        f"- Total logits materialization: `{trace.get('total_logits_mib', 0.0):.2f} MiB`",
        f"- Events without logits byte estimate: `{trace.get('logits_unknown_bytes_events', 0)}`",
        f"- Shadow epilogue events: `{trace.get('shadow_events', 0)}`",
        f"- Shadow epilogue eligible: `{trace.get('shadow_eligible_events', 0)}`",
        (
            "- Shadow avoidable logits materialization: "
            f"`{trace.get('shadow_avoidable_logits_mib', 0.0):.2f} MiB`"
        ),
        "",
        "## Serving Shapes",
        "",
    ]
    if summary["shapes"]:
        lines.extend(
            [
                (
                    "| Concurrency | Input Tokens | Runs | Median TTFT ms | "
                    "Median ITL ms | Output tok/s |"
                ),
                "| ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for shape in summary["shapes"]:
            metrics = shape["metrics"]
            lines.append(
                f"| {shape['max_concurrency']} | {shape['input_tokens']} | "
                f"{shape['runs']} | {metrics['median_ttft_ms']:.5f} | "
                f"{metrics['median_itl_ms']:.5f} | "
                f"{metrics['output_throughput']:.5f} |"
            )
    else:
        lines.append("No serving reports found.")
    lines.extend(["", "## Logits Materialization Budget", ""])
    shape_budget = trace.get("shape_budget", [])
    if shape_budget:
        lines.extend(
            [
                "| Logits shape | Events | Eligible | Eligible logits MiB | Total logits MiB |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in shape_budget:
            lines.append(
                f"| `{row['shape']}` | {row['events']} | "
                f"{row['eligible_events']} | "
                f"{row['eligible_logits_mib']:.2f} | "
                f"{row['total_logits_mib']:.2f} |"
            )
    else:
        lines.append("No logits materialization budget recorded.")
    lines.extend(["", "## Shadow Epilogue Fallback Reasons", ""])
    shadow_reason_counts = trace.get("shadow_reason_counts", {})
    if shadow_reason_counts:
        lines.extend(["| Reason | Count |", "| --- | ---: |"])
        for reason, count in shadow_reason_counts.items():
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("No shadow epilogue fallback reasons recorded.")
    lines.extend(["", "## Fallback Reasons", ""])
    reason_counts = trace.get("reason_counts", {})
    if reason_counts:
        lines.extend(["| Reason | Count |", "| --- | ---: |"])
        for reason, count in reason_counts.items():
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("No fallback reasons recorded.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = summarize(args.campaign_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
