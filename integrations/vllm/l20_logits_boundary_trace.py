"""Trace-only gate for an L20 logits-boundary sampling fast path.

This helper is intentionally behavior-preserving. It records when a future
LM-head epilogue / sampled-token boundary would be safe to try, but it never
changes logits or sampler outputs.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

TRACE_ENV = "VLLM_L20_LOGITS_BOUNDARY_TRACE"
TRACE_LIMIT_ENV = "VLLM_L20_LOGITS_BOUNDARY_TRACE_LIMIT"
ALLOW_NON_L20_ENV = "VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20"

_TRACE_COUNT = 0


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def _as_shape(value: Any) -> list[int] | None:
    shape = getattr(value, "shape", None)
    if shape is None:
        return None
    return [int(dim) for dim in shape]


def _safe_int(value: Any, default: int | None = None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _array_any(value: Any) -> bool | None:
    if value is None:
        return None
    try:
        return bool(value.any())
    except AttributeError:
        return any(bool(item) for item in value)
    except Exception:
        return None


def _array_max(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value.max())
    except AttributeError:
        try:
            return float(max(value))
        except ValueError:
            return 0.0
    except Exception:
        return None


def _array_min(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value.min())
    except AttributeError:
        try:
            return float(min(value))
        except ValueError:
            return 0.0
    except Exception:
        return None


def _mapped_np(state: Any, field: str, idx_mapping_np: Any) -> Any:
    owner = getattr(state, field, None)
    data = getattr(owner, "np", owner)
    if data is None:
        return None
    try:
        return data[idx_mapping_np]
    except Exception:
        return data


def _device_reason(tensor: Any) -> str | None:
    device = getattr(tensor, "device", None)
    if getattr(device, "type", None) != "cuda":
        return "not_cuda"
    if _env_flag(ALLOW_NON_L20_ENV):
        return None
    try:
        import torch

        capability = torch.cuda.get_device_capability(device)
        name = torch.cuda.get_device_name(device)
    except Exception as exc:  # pragma: no cover - defensive runtime path.
        return f"device_query_failed:{type(exc).__name__}"
    if capability != (8, 9):
        return f"not_sm89:{capability[0]}{capability[1]}"
    if "L20" not in name:
        return f"not_l20:{name}"
    return None


def _sampling_metadata(model_runner: Any, input_batch: Any) -> dict[str, float | None]:
    sampler = getattr(model_runner, "sampler", None)
    sampling_states = getattr(sampler, "sampling_states", None)
    idx_mapping_np = getattr(input_batch, "idx_mapping_np", None)
    result = {}
    for field in ("temperature", "top_k", "top_p", "min_p", "num_logprobs"):
        values = _mapped_np(sampling_states, field, idx_mapping_np)
        result[f"{field}_min"] = _array_min(values)
        result[f"{field}_max"] = _array_max(values)
    return result


def _sampling_state_reasons(model_runner: Any, input_batch: Any) -> list[str]:
    sampler = getattr(model_runner, "sampler", None)
    if sampler is None:
        return ["missing_sampler"]
    idx_mapping_np = getattr(input_batch, "idx_mapping_np", None)
    reasons = []

    sampling_states = getattr(sampler, "sampling_states", None)
    num_logprobs = _mapped_np(sampling_states, "num_logprobs", idx_mapping_np)
    if _array_max(num_logprobs) not in (-1.0, None):
        reasons.append("token_logprobs")
    min_p = _mapped_np(sampling_states, "min_p", idx_mapping_np)
    if _array_max(min_p) not in (0.0, None):
        reasons.append("min_p")

    logprob_token_ids_state = getattr(sampler, "logprob_token_ids_state", None)
    token_ids = _mapped_np(logprob_token_ids_state, "num_token_ids", idx_mapping_np)
    if _array_max(token_ids) not in (0.0, None):
        reasons.append("logprob_token_ids")

    penalties_state = getattr(sampler, "penalties_state", None)
    if _array_any(_mapped_np(penalties_state, "use_penalty", idx_mapping_np)):
        reasons.append("penalties")

    logit_bias_state = getattr(sampler, "logit_bias_state", None)
    if _array_any(_mapped_np(logit_bias_state, "use_logit_bias", idx_mapping_np)):
        reasons.append("logit_bias_or_min_tokens")

    bad_words_state = getattr(sampler, "bad_words_state", None)
    bad_words = _mapped_np(bad_words_state, "num_bad_words", idx_mapping_np)
    if _array_max(bad_words) not in (0.0, None):
        reasons.append("bad_words")

    return reasons


def l20_logits_boundary_gate(
    model_runner: Any,
    input_batch: Any,
    grammar_output: Any,
    sample_hidden_states: Any,
    logits: Any,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Return whether a future fused logits-boundary path should be eligible."""
    reasons = []
    metadata = {
        "num_reqs": _safe_int(getattr(input_batch, "num_reqs", None)),
        "num_tokens": _safe_int(getattr(input_batch, "num_tokens", None)),
        "num_draft_tokens": _safe_int(getattr(input_batch, "num_draft_tokens", None)),
        "hidden_shape": _as_shape(sample_hidden_states),
        "logits_shape": _as_shape(logits),
        "sampling": _sampling_metadata(model_runner, input_batch),
    }

    device_reason = _device_reason(sample_hidden_states)
    if device_reason is not None:
        reasons.append(device_reason)

    parallel_config = getattr(model_runner, "parallel_config", None)
    tp_size = _safe_int(getattr(parallel_config, "tensor_parallel_size", None))
    metadata["tensor_parallel_size"] = tp_size
    if tp_size != 1:
        reasons.append("tensor_parallel_not_1")

    if grammar_output is not None or getattr(input_batch, "has_structured_output_reqs", False):
        reasons.append("grammar_or_structured_output")
    if getattr(input_batch, "num_draft_tokens", 0) != 0:
        reasons.append("spec_decode")
    if _array_any(getattr(input_batch, "is_prefilling_np", None)):
        reasons.append("prefill")
    if getattr(input_batch, "num_tokens", None) != getattr(input_batch, "num_reqs", None):
        reasons.append("not_single_token_decode")

    logits_shape = _as_shape(logits)
    num_reqs = metadata["num_reqs"]
    if logits_shape is None or len(logits_shape) != 2:
        reasons.append("unexpected_logits_rank")
    elif num_reqs is not None and logits_shape[0] != num_reqs:
        reasons.append("logits_rows_not_num_reqs")

    reasons.extend(_sampling_state_reasons(model_runner, input_batch))
    return len(reasons) == 0, reasons, metadata


def maybe_trace_l20_logits_boundary(
    model_runner: Any,
    input_batch: Any,
    grammar_output: Any,
    sample_hidden_states: Any,
    logits: Any,
) -> None:
    path = os.environ.get(TRACE_ENV)
    if not path:
        return
    global _TRACE_COUNT
    limit = int(os.environ.get(TRACE_LIMIT_ENV, "4096"))
    if _TRACE_COUNT >= limit:
        return
    eligible, reasons, metadata = l20_logits_boundary_gate(
        model_runner,
        input_batch,
        grammar_output,
        sample_hidden_states,
        logits,
    )
    event = {
        "ts": time.time(),
        "event": "l20_logits_boundary_gate",
        "eligible": eligible,
        "reasons": reasons,
        "metadata": metadata,
    }
    _TRACE_COUNT += 1
    try:
        trace_path = Path(path)
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        with trace_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    except Exception:
        return
