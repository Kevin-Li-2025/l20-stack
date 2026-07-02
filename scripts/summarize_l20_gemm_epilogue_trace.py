#!/usr/bin/env python3
"""Summarize fallback-first GEMM epilogue trace JSONL.

The hook records two different notions:

* event eligibility: the request is at a decode step where the hook can call the
  fallback-first API before ``compute_logits``;
* semantic-candidate eligibility: the sampling semantics match a producer-side
  target such as top-k/top-p plus sparse penalties.

This summarizer keeps them separate so prefill/profile events are not counted
as decode wins.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _counter_dict(counter: collections.Counter[str]) -> dict[str, int]:
    return dict(counter.most_common())


def summarize_trace(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "schema_version": 1,
        "events": 0,
        "event_eligible": 0,
        "api_called": 0,
        "fallback_to_compute_logits": 0,
        "semantic_candidate_eligible": 0,
        "decode_semantic_candidate_eligible": 0,
        "estimated_logits_bytes_fp32": 0,
        "decode_estimated_logits_bytes_fp32": 0,
    }
    targets: collections.Counter[str] = collections.Counter()
    priorities: collections.Counter[str] = collections.Counter()
    features: collections.Counter[str] = collections.Counter()
    semantic_reasons: collections.Counter[str] = collections.Counter()
    event_reasons: collections.Counter[str] = collections.Counter()
    history_sources: collections.Counter[str] = collections.Counter()
    hidden_shapes: collections.Counter[str] = collections.Counter()

    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            summary["events"] += 1
            event_eligible = bool(event.get("eligible"))
            if event_eligible:
                summary["event_eligible"] += 1
            for reason in event.get("reasons", []):
                event_reasons[str(reason)] += 1

            metadata = event.get("metadata", {})
            api = metadata.get("api", {})
            if api.get("api_called"):
                summary["api_called"] += 1
            if api.get("fallback_to_compute_logits"):
                summary["fallback_to_compute_logits"] += 1

            shape = metadata.get("hidden_shape")
            if shape:
                hidden_shapes["x".join(map(str, shape))] += 1

            semantic = metadata.get("semantic_candidate", {})
            semantic_eligible = bool(semantic.get("eligible"))
            target = str(semantic.get("target", "missing"))
            priority = str(semantic.get("priority", "missing"))
            targets[target] += 1
            priorities[priority] += 1
            logits_bytes = int(semantic.get("estimated_logits_bytes_fp32") or 0)
            if semantic_eligible:
                summary["semantic_candidate_eligible"] += 1
                summary["estimated_logits_bytes_fp32"] += logits_bytes
            if semantic_eligible and event_eligible:
                summary["decode_semantic_candidate_eligible"] += 1
                summary["decode_estimated_logits_bytes_fp32"] += logits_bytes
            for feature in semantic.get("features", []):
                features[str(feature)] += 1
            for reason in semantic.get("reasons", []):
                semantic_reasons[str(reason)] += 1
            history = semantic.get("history", {})
            history_sources[str(history.get("source"))] += 1

    events = int(summary["events"])
    summary["event_eligible_fraction"] = (
        summary["event_eligible"] / events if events else 0.0
    )
    summary["semantic_candidate_eligible_fraction"] = (
        summary["semantic_candidate_eligible"] / events if events else 0.0
    )
    summary["decode_semantic_candidate_eligible_fraction"] = (
        summary["decode_semantic_candidate_eligible"] / events if events else 0.0
    )
    summary["estimated_logits_mib_fp32"] = (
        summary["estimated_logits_bytes_fp32"] / 1048576
    )
    summary["decode_estimated_logits_mib_fp32"] = (
        summary["decode_estimated_logits_bytes_fp32"] / 1048576
    )
    summary["targets"] = _counter_dict(targets)
    summary["priorities"] = _counter_dict(priorities)
    summary["features"] = _counter_dict(features)
    summary["semantic_reasons"] = _counter_dict(semantic_reasons)
    summary["event_reasons"] = _counter_dict(event_reasons)
    summary["history_sources"] = _counter_dict(history_sources)
    summary["hidden_shapes"] = _counter_dict(hidden_shapes)
    return summary


def main() -> int:
    args = parse_args()
    summary = summarize_trace(args.trace)
    rendered = json.dumps(summary, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
