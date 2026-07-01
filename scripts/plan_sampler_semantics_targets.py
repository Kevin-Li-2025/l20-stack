#!/usr/bin/env python3
"""Plan next sampler/logits optimization targets from semantics probe results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from l20_stack.epilogue import SamplerConfig, plan_sampler_optimization


CASE_CONFIGS = {
    "greedy_no_penalty": SamplerConfig(temperature=0.0, top_k=-1, top_p=1.0),
    "greedy_default_repetition": SamplerConfig(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        has_penalties=True,
    ),
    "sample_topk_topp": SamplerConfig(temperature=0.8, top_k=50, top_p=0.9),
    "sample_topk_topp_penalty": SamplerConfig(
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        has_penalties=True,
    ),
    "greedy_token_logprobs": SamplerConfig(
        temperature=0.0,
        top_k=-1,
        top_p=1.0,
        num_logprobs=5,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    return parser.parse_args()


def _median_itl(case: dict) -> float | None:
    try:
        return float(case["itl_ms"]["median"])
    except (KeyError, TypeError, ValueError):
        return None


def build_plan(summary: dict) -> dict:
    cases = summary.get("cases", [])
    median_by_case = {case.get("case"): _median_itl(case) for case in cases}
    baseline = median_by_case.get("greedy_no_penalty")
    rows = []
    for case in cases:
        name = case.get("case")
        config = CASE_CONFIGS.get(name)
        if config is None:
            continue
        plan = plan_sampler_optimization(config)
        median = median_by_case.get(name)
        delta_pct = None
        if baseline and median is not None:
            delta_pct = 100.0 * (median - baseline) / baseline
        rows.append(
            {
                "case": name,
                "median_itl_ms": median,
                "observed_itl_delta_vs_greedy_pct": delta_pct,
                "plan": plan.to_dict(),
            }
        )
    p0 = [
        row
        for row in rows
        if row["plan"]["priority"] == "p0"
        and row["plan"]["eligible_for_next_prototype"]
    ]
    p0.sort(
        key=lambda row: row["observed_itl_delta_vs_greedy_pct"] or 0.0,
        reverse=True,
    )
    return {
        "schema_version": 1,
        "source_model": summary.get("model"),
        "baseline_case": "greedy_no_penalty",
        "baseline_median_itl_ms": baseline,
        "rows": rows,
        "recommended_next_target": p0[0] if p0 else None,
        "decision": (
            "target_sampling_semantics_boundary"
            if p0
            else "no_p0_sampling_semantics_target"
        ),
    }


def render_markdown(plan: dict) -> str:
    lines = [
        "# Sampler Semantics Target Plan",
        "",
        "| Case | Median ITL | Delta vs greedy | Target | Priority |",
        "| --- | ---: | ---: | --- | --- |",
    ]
    for row in plan["rows"]:
        delta = row["observed_itl_delta_vs_greedy_pct"]
        lines.append(
            f"| `{row['case']}` | "
            f"{(row['median_itl_ms'] or 0.0):.3f} ms | "
            f"{(delta or 0.0):+.2f}% | "
            f"`{row['plan']['target']}` | "
            f"`{row['plan']['priority']}` |"
        )
    target = plan.get("recommended_next_target")
    lines.extend(["", "## Recommendation", ""])
    if target:
        lines.append(
            "Start with "
            f"`{target['plan']['target']}` because it is a P0 semantics path "
            "with the largest observed ITL delta in the probe."
        )
    else:
        lines.append("No P0 semantics target was found in this probe.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    plan = build_plan(summary)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(render_markdown(plan), encoding="utf-8")
    print(json.dumps(plan, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
