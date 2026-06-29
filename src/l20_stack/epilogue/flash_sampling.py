"""CPU-safe planning helpers for a future L20 FlashSampling epilogue.

This module only models the semantic gate and launch shape policy.  It does
not import PyTorch/Triton and does not contain CUDA kernels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Optional


MAX_BATCH_SIZE = 4
MAX_VOCAB_SIZE = 262_144
HIDDEN_ALIGNMENT = 64
SUPPORTED_SAMPLING_MODES = frozenset({"greedy", "gumbel"})

REASON_NOT_DECODE_ONLY = "not_decode_only"
REASON_BATCH_GT_4 = "batch_gt_4"
REASON_VOCAB_GT_262144 = "vocab_gt_262144"
REASON_HIDDEN_NOT_DIVISIBLE_BY_64 = "hidden_not_divisible_by_64"
REASON_SAMPLING_MODE_UNSUPPORTED = "sampling_mode_unsupported"
REASON_LOGPROBS_UNSUPPORTED = "logprobs_unsupported"
REASON_PENALTIES_UNSUPPORTED = "penalties_unsupported"
REASON_BAD_WORDS_UNSUPPORTED = "bad_words_unsupported"
REASON_STRUCTURED_OUTPUT_UNSUPPORTED = "structured_output_unsupported"
REASON_SPEC_DECODE_UNSUPPORTED = "spec_decode_unsupported"
REASON_TOP_K_TOP_P_UNSUPPORTED = "top_k_top_p_unsupported"


@dataclass(frozen=True)
class FlashSamplingRequest:
    """Shape and semantics relevant to the first FlashSampling epilogue gate."""

    batch_size: int
    vocab_size: int
    hidden_size: int
    decode_only: bool = True
    sampling_mode: str = "greedy"
    top_k: Optional[int] = None
    top_p: Optional[float] = None
    num_logprobs: int = 0
    has_penalties: bool = False
    has_bad_words: bool = False
    has_structured_output: bool = False
    speculative_decode: bool = False

    def __post_init__(self) -> None:
        for name in ("batch_size", "vocab_size", "hidden_size"):
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_logprobs < -1:
            raise ValueError("num_logprobs must be -1 or greater")


@dataclass(frozen=True)
class FlashSamplingLaunchPolicy:
    """Static launch-shape plan for the epilogue prototype."""

    block_vocab: int
    block_hidden: int
    blocks_per_row: int
    reduce_block: int
    num_warps: int
    num_stages: int
    strategy: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FlashSamplingGateDecision:
    """Gate result with stable fallback reasons and an optional launch policy."""

    eligible: bool
    reasons: tuple[str, ...]
    policy: Optional[FlashSamplingLaunchPolicy] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "eligible": self.eligible,
            "reasons": list(self.reasons),
            "policy": self.policy.to_dict() if self.policy is not None else None,
        }


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def _uses_top_k_top_p(request: FlashSamplingRequest) -> bool:
    # vLLM normalizes full-vocabulary sampling to top_k == vocab_size.
    top_k_enabled = request.top_k not in (None, -1, 0) and request.top_k < request.vocab_size
    top_p_enabled = request.top_p is not None and request.top_p != 1.0
    return top_k_enabled or top_p_enabled


def flash_sampling_gate_reasons(request: FlashSamplingRequest) -> list[str]:
    """Return all reasons the conservative epilogue prototype should fall back."""

    reasons: list[str] = []
    if not request.decode_only:
        reasons.append(REASON_NOT_DECODE_ONLY)
    if request.batch_size > MAX_BATCH_SIZE:
        reasons.append(REASON_BATCH_GT_4)
    if request.vocab_size > MAX_VOCAB_SIZE:
        reasons.append(REASON_VOCAB_GT_262144)
    if request.hidden_size % HIDDEN_ALIGNMENT != 0:
        reasons.append(REASON_HIDDEN_NOT_DIVISIBLE_BY_64)
    if request.sampling_mode not in SUPPORTED_SAMPLING_MODES:
        reasons.append(REASON_SAMPLING_MODE_UNSUPPORTED)
    if request.num_logprobs not in (-1, 0):
        reasons.append(REASON_LOGPROBS_UNSUPPORTED)
    if request.has_penalties:
        reasons.append(REASON_PENALTIES_UNSUPPORTED)
    if request.has_bad_words:
        reasons.append(REASON_BAD_WORDS_UNSUPPORTED)
    if request.has_structured_output:
        reasons.append(REASON_STRUCTURED_OUTPUT_UNSUPPORTED)
    if request.speculative_decode:
        reasons.append(REASON_SPEC_DECODE_UNSUPPORTED)
    if _uses_top_k_top_p(request):
        reasons.append(REASON_TOP_K_TOP_P_UNSUPPORTED)
    return reasons


def should_use_flash_sampling_epilogue(request: FlashSamplingRequest) -> bool:
    return not flash_sampling_gate_reasons(request)


def _build_launch_policy(request: FlashSamplingRequest) -> FlashSamplingLaunchPolicy:
    if request.batch_size == 1:
        block_vocab = 32
        block_hidden = 64
    else:
        block_vocab = 64
        block_hidden = 128

    blocks_per_row = (request.vocab_size + block_vocab - 1) // block_vocab
    return FlashSamplingLaunchPolicy(
        block_vocab=block_vocab,
        block_hidden=block_hidden,
        blocks_per_row=blocks_per_row,
        reduce_block=next_power_of_2(blocks_per_row),
        num_warps=4 if block_vocab <= 32 else 8,
        num_stages=3,
        strategy="two_stage_lm_head_flash_sampling_epilogue_plan",
    )


def flash_sampling_launch_policy(
    request: FlashSamplingRequest,
) -> FlashSamplingLaunchPolicy:
    """Return the launch policy, or raise if the request is outside the gate."""

    reasons = flash_sampling_gate_reasons(request)
    if reasons:
        raise ValueError("request is outside FlashSampling epilogue gate: " + ", ".join(reasons))
    return _build_launch_policy(request)


def plan_flash_sampling_epilogue(request: FlashSamplingRequest) -> FlashSamplingGateDecision:
    """Plan the epilogue gate and launch policy without requiring CUDA imports."""

    reasons = tuple(flash_sampling_gate_reasons(request))
    return FlashSamplingGateDecision(
        eligible=not reasons,
        reasons=reasons,
        policy=None if reasons else _build_launch_policy(request),
    )


__all__ = [
    "HIDDEN_ALIGNMENT",
    "MAX_BATCH_SIZE",
    "MAX_VOCAB_SIZE",
    "REASON_BAD_WORDS_UNSUPPORTED",
    "REASON_BATCH_GT_4",
    "REASON_HIDDEN_NOT_DIVISIBLE_BY_64",
    "REASON_LOGPROBS_UNSUPPORTED",
    "REASON_NOT_DECODE_ONLY",
    "REASON_PENALTIES_UNSUPPORTED",
    "REASON_SAMPLING_MODE_UNSUPPORTED",
    "REASON_SPEC_DECODE_UNSUPPORTED",
    "REASON_STRUCTURED_OUTPUT_UNSUPPORTED",
    "REASON_TOP_K_TOP_P_UNSUPPORTED",
    "REASON_VOCAB_GT_262144",
    "SUPPORTED_SAMPLING_MODES",
    "FlashSamplingGateDecision",
    "FlashSamplingLaunchPolicy",
    "FlashSamplingRequest",
    "flash_sampling_gate_reasons",
    "flash_sampling_launch_policy",
    "next_power_of_2",
    "plan_flash_sampling_epilogue",
    "should_use_flash_sampling_epilogue",
]
