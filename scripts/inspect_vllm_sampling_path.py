#!/usr/bin/env python3
"""Inspect a vLLM server log for sampler-path evidence."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERNS = {
    "flashinfer": re.compile(r"flashinfer", re.IGNORECASE),
    "sampling": re.compile(r"sampl", re.IGNORECASE),
    "cpu": re.compile(r"\bcpu\b", re.IGNORECASE),
    "fallback": re.compile(r"fallback|fall back", re.IGNORECASE),
    "cuda13": re.compile(r"cuda\s*13|cu13|nvcc", re.IGNORECASE),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-lines", type=int, default=80)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lines = args.log.read_text(encoding="utf-8", errors="replace").splitlines()
    matches = {name: [] for name in PATTERNS}
    for lineno, line in enumerate(lines, start=1):
        for name, pattern in PATTERNS.items():
            if pattern.search(line):
                matches[name].append({"line": lineno, "text": line[-500:]})

    result = {
        "schema_version": 1,
        "log": str(args.log),
        "line_count": len(lines),
        "match_counts": {name: len(rows) for name, rows in matches.items()},
        "matches": {name: rows[: args.max_lines] for name, rows in matches.items()},
        "cpu_fallback_suspected": bool(matches["fallback"] and matches["cpu"]),
        "notes": [
            "This is log evidence only. Absence of CPU/fallback strings does not prove "
            "that no CPU synchronization happened inside lower-level kernels.",
            "Use paired stochastic serving latency and FlashInfer prewarm status as the "
            "primary service-level signal.",
        ],
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
