#!/usr/bin/env python3
"""Summarize L20 logits-boundary trace JSONL emitted by the vLLM hook."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def read_events(path: Path) -> list[dict]:
    events = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        events.append(json.loads(line))
    return events


def summarize(events: list[dict]) -> dict:
    reason_counts = Counter()
    logits_shapes = Counter()
    hidden_shapes = Counter()
    eligible = 0
    for event in events:
        if event.get("eligible"):
            eligible += 1
        for reason in event.get("reasons", []):
            reason_counts[reason] += 1
        metadata = event.get("metadata", {})
        logits_shape = metadata.get("logits_shape")
        hidden_shape = metadata.get("hidden_shape")
        if logits_shape is not None:
            logits_shapes["x".join(str(dim) for dim in logits_shape)] += 1
        if hidden_shape is not None:
            hidden_shapes["x".join(str(dim) for dim in hidden_shape)] += 1
    total = len(events)
    return {
        "schema_version": 1,
        "total_events": total,
        "eligible_events": eligible,
        "fallback_events": total - eligible,
        "eligible_fraction": eligible / total if total else 0.0,
        "reason_counts": dict(reason_counts.most_common()),
        "logits_shape_counts": dict(logits_shapes.most_common()),
        "hidden_shape_counts": dict(hidden_shapes.most_common()),
    }


def render_markdown(summary: dict, trace: Path) -> str:
    lines = [
        "# L20 vLLM Logits Boundary Trace Summary",
        "",
        f"- Trace: `{trace}`",
        f"- Total events: `{summary['total_events']}`",
        f"- Eligible events: `{summary['eligible_events']}`",
        f"- Fallback events: `{summary['fallback_events']}`",
        f"- Eligible fraction: `{summary['eligible_fraction']:.4f}`",
        "",
        "## Fallback Reasons",
        "",
    ]
    if summary["reason_counts"]:
        lines.extend(["| Reason | Count |", "| --- | ---: |"])
        for reason, count in summary["reason_counts"].items():
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("No fallback reasons recorded.")
    lines.extend(["", "## Logits Shapes", ""])
    if summary["logits_shape_counts"]:
        lines.extend(["| Shape | Count |", "| --- | ---: |"])
        for shape, count in summary["logits_shape_counts"].items():
            lines.append(f"| `{shape}` | {count} |")
    else:
        lines.append("No logits shapes recorded.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    events = read_events(args.trace)
    summary = summarize(events)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(summary, args.trace), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
