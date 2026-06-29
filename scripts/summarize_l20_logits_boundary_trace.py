#!/usr/bin/env python3
"""Summarize L20 logits-boundary trace JSONL emitted by the vLLM hook."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

MIB = 1024 * 1024


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


def _shape_key(shape: list[int] | None) -> str | None:
    if shape is None:
        return None
    return "x".join(str(dim) for dim in shape)


def _dtype_nbytes(dtype: object) -> int | None:
    if dtype is None:
        return None
    text = str(dtype).lower()
    if "float8" in text or "int8" in text or "uint8" in text or "bool" in text:
        return 1
    if "bfloat16" in text or "float16" in text or "half" in text or "int16" in text:
        return 2
    if "float32" in text or text.endswith(".float") or "int32" in text:
        return 4
    if "float64" in text or "double" in text or "int64" in text:
        return 8
    return None


def _shape_numel(shape: list[int] | None) -> int | None:
    if shape is None:
        return None
    numel = 1
    for dim in shape:
        numel *= int(dim)
    return numel


def _tensor_bytes(metadata: dict, prefix: str) -> int | None:
    value = metadata.get(f"{prefix}_bytes")
    if value is not None:
        return int(value)
    element_bytes = metadata.get(f"{prefix}_element_bytes")
    if element_bytes is None:
        element_bytes = _dtype_nbytes(metadata.get(f"{prefix}_dtype"))
    shape = metadata.get(f"{prefix}_shape")
    numel = _shape_numel(shape)
    if element_bytes is None or numel is None:
        return None
    return int(element_bytes) * numel


def _mib(value: int) -> float:
    return value / MIB


def summarize(events: list[dict]) -> dict:
    reason_counts = Counter()
    shadow_reason_counts = Counter()
    logits_shapes = Counter()
    hidden_shapes = Counter()
    shape_budget: dict[str, dict] = {}
    eligible = 0
    shadow_events = 0
    shadow_eligible_events = 0
    total_logits_bytes = 0
    eligible_logits_bytes = 0
    shadow_avoidable_logits_bytes = 0
    total_hidden_bytes = 0
    eligible_hidden_bytes = 0
    logits_unknown_bytes_events = 0
    hidden_unknown_bytes_events = 0
    for event in events:
        is_eligible = bool(event.get("eligible"))
        if is_eligible:
            eligible += 1
        for reason in event.get("reasons", []):
            reason_counts[reason] += 1
        metadata = event.get("metadata", {})
        shadow = metadata.get("shadow_epilogue")
        if shadow is not None:
            shadow_events += 1
            if shadow.get("would_use_epilogue"):
                shadow_eligible_events += 1
            for reason in shadow.get("fallback_reasons", []):
                shadow_reason_counts[reason] += 1
            shadow_bytes = shadow.get("avoidable_logits_materialization_bytes")
            if shadow_bytes is not None:
                shadow_avoidable_logits_bytes += int(shadow_bytes)
        logits_shape = metadata.get("logits_shape")
        hidden_shape = metadata.get("hidden_shape")
        logits_key = _shape_key(logits_shape)
        hidden_key = _shape_key(hidden_shape)
        if logits_key is not None:
            logits_shapes[logits_key] += 1
            entry = shape_budget.setdefault(
                logits_key,
                {
                    "events": 0,
                    "eligible_events": 0,
                    "fallback_events": 0,
                    "total_logits_bytes": 0,
                    "eligible_logits_bytes": 0,
                },
            )
            entry["events"] += 1
            if is_eligible:
                entry["eligible_events"] += 1
            else:
                entry["fallback_events"] += 1
        if hidden_key is not None:
            hidden_shapes[hidden_key] += 1

        logits_bytes = _tensor_bytes(metadata, "logits")
        if logits_bytes is None:
            logits_unknown_bytes_events += 1
        else:
            total_logits_bytes += logits_bytes
            if is_eligible:
                eligible_logits_bytes += logits_bytes
            if logits_key is not None:
                shape_budget[logits_key]["total_logits_bytes"] += logits_bytes
                if is_eligible:
                    shape_budget[logits_key]["eligible_logits_bytes"] += logits_bytes

        hidden_bytes = _tensor_bytes(metadata, "hidden")
        if hidden_bytes is None:
            hidden_unknown_bytes_events += 1
        else:
            total_hidden_bytes += hidden_bytes
            if is_eligible:
                eligible_hidden_bytes += hidden_bytes

    shape_budget_rows = []
    for shape, row in shape_budget.items():
        enriched = dict(row)
        enriched["shape"] = shape
        enriched["total_logits_mib"] = _mib(enriched["total_logits_bytes"])
        enriched["eligible_logits_mib"] = _mib(enriched["eligible_logits_bytes"])
        shape_budget_rows.append(enriched)
    shape_budget_rows.sort(
        key=lambda row: (row["eligible_logits_bytes"], row["eligible_events"]),
        reverse=True,
    )
    total = len(events)
    return {
        "schema_version": 3,
        "total_events": total,
        "eligible_events": eligible,
        "fallback_events": total - eligible,
        "eligible_fraction": eligible / total if total else 0.0,
        "reason_counts": dict(reason_counts.most_common()),
        "shadow_events": shadow_events,
        "shadow_eligible_events": shadow_eligible_events,
        "shadow_fallback_events": shadow_events - shadow_eligible_events,
        "shadow_eligible_fraction": shadow_eligible_events / shadow_events
        if shadow_events
        else 0.0,
        "shadow_reason_counts": dict(shadow_reason_counts.most_common()),
        "shadow_avoidable_logits_bytes": shadow_avoidable_logits_bytes,
        "shadow_avoidable_logits_mib": _mib(shadow_avoidable_logits_bytes),
        "logits_shape_counts": dict(logits_shapes.most_common()),
        "hidden_shape_counts": dict(hidden_shapes.most_common()),
        "total_logits_bytes": total_logits_bytes,
        "eligible_logits_bytes": eligible_logits_bytes,
        "total_logits_mib": _mib(total_logits_bytes),
        "eligible_logits_mib": _mib(eligible_logits_bytes),
        "total_hidden_bytes": total_hidden_bytes,
        "eligible_hidden_bytes": eligible_hidden_bytes,
        "total_hidden_mib": _mib(total_hidden_bytes),
        "eligible_hidden_mib": _mib(eligible_hidden_bytes),
        "logits_unknown_bytes_events": logits_unknown_bytes_events,
        "hidden_unknown_bytes_events": hidden_unknown_bytes_events,
        "shape_budget": shape_budget_rows,
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
        f"- Eligible logits materialization: `{summary['eligible_logits_mib']:.2f} MiB`",
        f"- Total logits materialization: `{summary['total_logits_mib']:.2f} MiB`",
        f"- Events without logits byte estimate: `{summary['logits_unknown_bytes_events']}`",
        f"- Shadow epilogue events: `{summary['shadow_events']}`",
        f"- Shadow epilogue eligible: `{summary['shadow_eligible_events']}`",
        (
            "- Shadow avoidable logits materialization: "
            f"`{summary['shadow_avoidable_logits_mib']:.2f} MiB`"
        ),
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
    lines.extend(["", "## Shadow Epilogue Fallback Reasons", ""])
    if summary["shadow_reason_counts"]:
        lines.extend(["| Reason | Count |", "| --- | ---: |"])
        for reason, count in summary["shadow_reason_counts"].items():
            lines.append(f"| `{reason}` | {count} |")
    else:
        lines.append("No shadow epilogue fallback reasons recorded.")
    lines.extend(["", "## Logits Materialization Budget", ""])
    if summary["shape_budget"]:
        lines.extend(
            [
                "| Logits shape | Events | Eligible | Eligible logits MiB | Total logits MiB |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in summary["shape_budget"]:
            lines.append(
                (
                    "| `{shape}` | {events} | {eligible_events} | "
                    "{eligible:.2f} | {total:.2f} |"
                ).format(
                    shape=row["shape"],
                    events=row["events"],
                    eligible_events=row["eligible_events"],
                    eligible=row["eligible_logits_mib"],
                    total=row["total_logits_mib"],
                )
            )
    else:
        lines.append("No logits materialization budget recorded.")
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
