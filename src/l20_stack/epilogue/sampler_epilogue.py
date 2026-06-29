"""Conservative semantic gate for a future sampler/logits epilogue."""

from __future__ import annotations

from dataclasses import dataclass


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
