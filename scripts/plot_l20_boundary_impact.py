#!/usr/bin/env python3
"""Generate the L20 boundary-impact table and a small dependency-free SVG."""

from __future__ import annotations

import argparse
from pathlib import Path

from l20_stack.epilogue.compare import (
    BoundaryImpact,
    build_boundary_impacts,
    render_markdown,
    write_csv,
    write_json,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/results/l20-boundary-impact"),
    )
    return parser.parse_args()


def _bar(width: float, max_width: int) -> float:
    return max(0.0, min(float(max_width), width))


def render_svg(rows: list[BoundaryImpact]) -> str:
    width = 1100
    micro_x = 430
    serving_x = 660
    budget_x = 900
    bar_width = 160
    row_h = 58
    top = 86
    height = top + row_h * len(rows) + 52
    max_micro = max((row.micro_speedup_x or 0.0) for row in rows) or 1.0
    max_serving = max(abs(row.serving_impact_pct or 0.0) for row in rows) or 1.0
    max_budget = max((row.materialization_mib or 0.0) for row in rows) or 1.0

    lines = [
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
            f'height="{height}" viewBox="0 0 {width} {height}">'
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        (
            "<style>text{font-family:Inter,Arial,sans-serif;fill:#172033}"
            ".small{font-size:13px}.label{font-size:14px;font-weight:600}"
            ".title{font-size:22px;font-weight:700}.muted{fill:#667085}"
            ".axis{stroke:#d0d5dd;stroke-width:1}.good{fill:#137333}"
            ".bad{fill:#b42318}.budget{fill:#175cd3}.micro{fill:#7a5af8}</style>"
        ),
        '<text x="28" y="36" class="title">Where L20 inference optimizations stop mattering</text>',
        (
            '<text x="28" y="60" class="small muted">Micro speedups only matter '
            "when the boundary survives vLLM serving and Amdahl dilution.</text>"
        ),
        f'<text x="{micro_x}" y="78" class="small muted">micro speedup</text>',
        f'<text x="{serving_x}" y="78" class="small muted">serving impact</text>',
        f'<text x="{budget_x}" y="78" class="small muted">logits budget</text>',
    ]
    for index, row in enumerate(rows):
        y = top + index * row_h
        lines.append(f'<line x1="28" y1="{y - 18}" x2="{width - 28}" y2="{y - 18}" class="axis"/>')
        lines.append(f'<text x="28" y="{y + 4}" class="label">{row.boundary}</text>')
        lines.append(f'<text x="28" y="{y + 24}" class="small muted">{row.decision}</text>')

        if row.micro_speedup_x is not None:
            w = _bar((row.micro_speedup_x / max_micro) * bar_width, bar_width)
            lines.append(
                f'<rect x="{micro_x}" y="{y - 5}" width="{w:.1f}" '
                'height="16" rx="2" class="micro"/>'
            )
            lines.append(
                f'<text x="{micro_x + w + 8:.1f}" y="{y + 8}" '
                f'class="small">{row.micro_speedup_x:.2f}x</text>'
            )
        else:
            lines.append(f'<text x="{micro_x}" y="{y + 8}" class="small muted">not measured</text>')

        if row.serving_impact_pct is not None:
            value = row.serving_impact_pct
            w = _bar((abs(value) / max_serving) * bar_width, bar_width)
            klass = "good" if value >= 0 else "bad"
            lines.append(
                f'<rect x="{serving_x}" y="{y - 5}" width="{w:.1f}" '
                f'height="16" rx="2" class="{klass}"/>'
            )
            lines.append(
                f'<text x="{serving_x + w + 8:.1f}" y="{y + 8}" '
                f'class="small">{value:+.2f}%</text>'
            )
        else:
            lines.append(f'<text x="{serving_x}" y="{y + 8}" class="small muted">pending</text>')

        if row.materialization_mib is not None:
            w = _bar((row.materialization_mib / max_budget) * bar_width, bar_width)
            lines.append(
                f'<rect x="{budget_x}" y="{y - 5}" width="{w:.1f}" '
                'height="16" rx="2" class="budget"/>'
            )
            lines.append(
                f'<text x="{budget_x + w + 8:.1f}" y="{y + 8}" class="small">'
                f'{row.eligible_fraction_pct:.1f}%, {row.materialization_mib:.1f} MiB</text>'
            )
        else:
            lines.append(f'<text x="{budget_x}" y="{y + 8}" class="small muted">n/a</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows = build_boundary_impacts(args.repo_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(rows, args.output_dir / "boundary-impact.json")
    write_csv(rows, args.output_dir / "boundary-impact.csv")
    (args.output_dir / "README.md").write_text(render_markdown(rows), encoding="utf-8")
    (args.output_dir / "boundary-impact.svg").write_text(render_svg(rows), encoding="utf-8")
    print(f"wrote {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
