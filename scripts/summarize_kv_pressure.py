#!/usr/bin/env python3
"""Summarize L20 multi-turn KV-pressure benchmark directories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def median_or_none(values: list[float]) -> float | None:
    return median(values) if values else None


def summarize_success(path: Path, payload: dict, metadata: dict) -> dict:
    reports = payload.get("reports", [])
    ttft = [row["ttft_ms"] for row in reports if row.get("ttft_ms") is not None]
    e2e = [row["e2e_ms"] for row in reports if row.get("e2e_ms") is not None]
    first_ttft = ttft[0] if ttft else None
    last_ttft = ttft[-1] if ttft else None
    late_over_first = None
    if first_ttft not in (None, 0) and last_ttft is not None:
        late_over_first = last_ttft / first_ttft
    return {
        "path": str(path),
        "status": "ok",
        "metadata": metadata,
        "turns_completed": len(reports),
        "first_turn_ttft_ms": first_ttft,
        "last_turn_ttft_ms": last_ttft,
        "late_over_first_ttft": late_over_first,
        "median_ttft_ms": median_or_none(ttft),
        "median_e2e_ms": median_or_none(e2e),
        "max_ttft_ms": max(ttft) if ttft else None,
    }


def summarize_run_dir(run_dir: Path) -> dict:
    metadata_path = run_dir / "kv-pressure-run.json"
    metadata = load_json(metadata_path) if metadata_path.exists() else {}
    failure_path = run_dir / "kv-pressure-failure.json"
    if failure_path.exists():
        failure = load_json(failure_path)
        return {
            "path": str(run_dir),
            "status": failure.get("status", "failed"),
            "reason": failure.get("reason"),
            "oom_suspected": failure.get("oom_suspected"),
            "flashinfer_observed": failure.get("flashinfer_observed"),
            "metadata": failure.get("metadata", metadata),
        }

    candidates = sorted(run_dir.glob("kv-pressure-prefix-cache-*.json"))
    if not candidates:
        return {
            "path": str(run_dir),
            "status": "missing_result",
            "metadata": metadata,
        }
    return summarize_success(run_dir, load_json(candidates[0]), metadata)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    reports = []
    for path in args.paths:
        if path.is_file():
            payload = load_json(path)
            metadata = {}
            reports.append(summarize_success(path.parent, payload, metadata))
        elif path.is_dir():
            child_runs = [
                child for child in sorted(path.iterdir())
                if child.is_dir() and (
                    (child / "kv-pressure-run.json").exists()
                    or (child / "kv-pressure-failure.json").exists()
                    or list(child.glob("kv-pressure-prefix-cache-*.json"))
                )
            ]
            if child_runs:
                reports.extend(summarize_run_dir(child) for child in child_runs)
            else:
                reports.append(summarize_run_dir(path))

    ok_reports = [row for row in reports if row["status"] == "ok"]
    result = {
        "schema_version": 1,
        "reports": reports,
        "summary": {
            "total_runs": len(reports),
            "ok_runs": len(ok_reports),
            "failed_runs": len(reports) - len(ok_reports),
            "best_median_ttft_ms": min(
                (row["median_ttft_ms"] for row in ok_reports if row["median_ttft_ms"] is not None),
                default=None,
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
