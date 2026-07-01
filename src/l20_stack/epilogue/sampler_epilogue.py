"""Semantic gates and priorities for future sampler/logits epilogues."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class SamplerConfig:
    """Only the fields that affect the first safe epilogue boundary."""

    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0
    num_logprobs: int = 0
    has_grammar: bool = False
    has_structured_output: bool = False
    has_penalties: bool = False
    has_bad_words: bool = False
    has_logit_bias: bool = False
    has_allowed_token_ids: bool = False
    per_request_generators: bool = False
    speculative_decode: bool = False
    prefill: bool = False
    tensor_parallel_size: int = 1


@dataclass(frozen=True)
class SamplerOptimizationPlan:
    """A CPU-safe plan for the next sampler/logits optimization boundary."""

    target: str
    priority: Literal["control", "p0", "p1", "defer"]
    eligible_for_next_prototype: bool
    reasons: tuple[str, ...]
    expected_itl_delta_vs_greedy_pct: float | None = None

    def to_dict(self) -> dict:
        return {
            "target": self.target,
            "priority": self.priority,
            "eligible_for_next_prototype": self.eligible_for_next_prototype,
            "reasons": list(self.reasons),
            "expected_itl_delta_vs_greedy_pct": self.expected_itl_delta_vs_greedy_pct,
        }


_DEFER_REASONS = {
    "tensor_parallel_not_1",
    "prefill",
    "spec_decode",
    "grammar_or_structured_output",
    "bad_words",
    "logit_bias_or_min_tokens",
    "per_request_generators",
    "min_p",
    "invalid_temperature",
    "invalid_top_k",
    "invalid_top_p",
}


def sampler_gate_reasons(config: SamplerConfig) -> list[str]:
    """Return reasons a request should not use the first epilogue prototype."""

    reasons: list[str] = []
    if config.tensor_parallel_size != 1:
        reasons.append("tensor_parallel_not_1")
    if config.prefill:
        reasons.append("prefill")
    if config.speculative_decode:
        reasons.append("spec_decode")
    if config.has_grammar or config.has_structured_output:
        reasons.append("grammar_or_structured_output")
    if config.num_logprobs not in (-1, 0):
        reasons.append("token_logprobs")
    if config.min_p != 0.0:
        reasons.append("min_p")
    if config.has_penalties:
        reasons.append("penalties")
    if config.has_bad_words:
        reasons.append("bad_words")
    if config.has_logit_bias or config.has_allowed_token_ids:
        reasons.append("logit_bias_or_min_tokens")
    if config.per_request_generators:
        reasons.append("per_request_generators")
    if config.temperature <= 0.0:
        reasons.append("invalid_temperature")
    if config.top_k == 0:
        reasons.append("invalid_top_k")
    if not 0.0 < config.top_p <= 1.0:
        reasons.append("invalid_top_p")
    return sorted(set(reasons))


def _semantic_features(config: SamplerConfig) -> set[str]:
    features: set[str] = set()
    if config.num_logprobs not in (-1, 0):
        features.add("token_logprobs")
    if config.has_penalties:
        features.add("penalties")
    if config.temperature > 0.0 and (config.top_k not in (-1, 1) or config.top_p < 1.0):
        features.add("topk_topp")
    elif config.temperature > 0.0:
        features.add("full_vocab_sampling")
    return features


def plan_sampler_optimization(config: SamplerConfig) -> SamplerOptimizationPlan:
    """Prioritize the next sampler/logits boundary after the A100 semantics probe.

    This is not a support gate for the existing prototype. It is a research
    planner: the latest A100 probe shows plain greedy/no-penalty decode is the
    fast control, while top-k/top-p, token logprobs, and penalties add roughly
    37-42% median ITL. The planner therefore marks those semantics as the next
    optimization targets, while still deferring high-risk features that change
    masking, batching, or distributed behavior.
    """

    reasons = set(sampler_gate_reasons(config))
    # vLLM represents greedy decode with temperature=0 in request payloads, and
    # sometimes with -1 in internal tensors. That is invalid for stochastic
    # sampling kernels, but it is the fast control for the semantics planner.
    if config.temperature <= 0.0:
        reasons.discard("invalid_temperature")
    blocking = sorted(reasons & _DEFER_REASONS)
    if blocking:
        return SamplerOptimizationPlan(
            target="unsupported_semantics",
            priority="defer",
            eligible_for_next_prototype=False,
            reasons=tuple(blocking),
            expected_itl_delta_vs_greedy_pct=None,
        )

    features = _semantic_features(config)
    if "topk_topp" in features:
        suffix = "+penalty" if "penalties" in features else ""
        return SamplerOptimizationPlan(
            target=f"fused_topk_topp{suffix}",
            priority="p0",
            eligible_for_next_prototype=True,
            reasons=tuple(sorted(features)),
            expected_itl_delta_vs_greedy_pct=42.0,
        )
    if "token_logprobs" in features:
        suffix = "+penalty" if "penalties" in features else ""
        return SamplerOptimizationPlan(
            target=f"fused_token_logprobs{suffix}",
            priority="p0",
            eligible_for_next_prototype=True,
            reasons=tuple(sorted(features)),
            expected_itl_delta_vs_greedy_pct=39.0,
        )
    if "penalties" in features:
        return SamplerOptimizationPlan(
            target="fused_repetition_penalty",
            priority="p1",
            eligible_for_next_prototype=True,
            reasons=tuple(sorted(features)),
            expected_itl_delta_vs_greedy_pct=37.0,
        )
    if "full_vocab_sampling" in features:
        return SamplerOptimizationPlan(
            target="full_vocab_sampling_control",
            priority="defer",
            eligible_for_next_prototype=False,
            reasons=tuple(sorted(features)),
            expected_itl_delta_vs_greedy_pct=None,
        )
    return SamplerOptimizationPlan(
        target="greedy_no_penalty_control",
        priority="control",
        eligible_for_next_prototype=False,
        reasons=(),
        expected_itl_delta_vs_greedy_pct=0.0,
    )
