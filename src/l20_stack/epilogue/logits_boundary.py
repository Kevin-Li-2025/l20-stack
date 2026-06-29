"""Utilities for the L20 LM-head/logits boundary trace."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class LogitsBoundaryBudget:
    """Aggregated opportunity size for a future logits/sampling epilogue."""

    total_events: int
    eligible_events: int
    fallback_events: int
    eligible_fraction: float
    eligible_logits_mib: float
    total_logits_mib: float
    unknown_byte_events: int
    top_shape: str | None = None
    top_shape_eligible_logits_mib: float | None = None

    @property
    def ineligible_fraction(self) -> float:
        return 1.0 - self.eligible_fraction

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_events": self.total_events,
            "eligible_events": self.eligible_events,
            "fallback_events": self.fallback_events,
            "eligible_fraction": self.eligible_fraction,
            "eligible_logits_mib": self.eligible_logits_mib,
            "total_logits_mib": self.total_logits_mib,
            "unknown_byte_events": self.unknown_byte_events,
            "top_shape": self.top_shape,
            "top_shape_eligible_logits_mib": self.top_shape_eligible_logits_mib,
        }


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _campaign_summary_path(path: Path) -> Path:
    if path.is_dir():
        return path / "campaign-summary.json"
    return path


def _trace_summary(payload: dict[str, Any]) -> dict[str, Any]:
    return payload.get("trace_summary", payload)


def budget_from_summary(payload: dict[str, Any]) -> LogitsBoundaryBudget:
    """Build a budget object from a trace summary or campaign summary."""

    trace = _trace_summary(payload)
    shape_budget = trace.get("shape_budget", [])
    top_shape = shape_budget[0] if shape_budget else {}
    return LogitsBoundaryBudget(
        total_events=int(trace.get("total_events", 0)),
        eligible_events=int(trace.get("eligible_events", 0)),
        fallback_events=int(trace.get("fallback_events", 0)),
        eligible_fraction=float(trace.get("eligible_fraction", 0.0)),
        eligible_logits_mib=float(trace.get("eligible_logits_mib", 0.0)),
        total_logits_mib=float(trace.get("total_logits_mib", 0.0)),
        unknown_byte_events=int(trace.get("logits_unknown_bytes_events", 0)),
        top_shape=top_shape.get("shape"),
        top_shape_eligible_logits_mib=(
            float(top_shape["eligible_logits_mib"])
            if "eligible_logits_mib" in top_shape
            else None
        ),
    )


def load_logits_boundary_budget(path: str | Path) -> LogitsBoundaryBudget:
    """Load a logits-boundary campaign directory or JSON summary."""

    return budget_from_summary(_load_json(_campaign_summary_path(Path(path))))
