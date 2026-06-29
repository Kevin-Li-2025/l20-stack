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


def _dtype_text(value: Any) -> str | None:
    dtype = getattr(value, "dtype", None)
    return str(dtype) if dtype is not None else None


def _dtype_nbytes(dtype: Any) -> int | None:
    if dtype is None:
        return None
    text = str(dtype).lower()
    if "float8" in text or "int8" in text or "uint8" in text or "bool" in text:
        return 1
    if "bfloat16" in text or "float16" in text or "half" in text or "int16" in text:
        return 2
    if "float32" in text or text.endswith(".float") or "int32" in text:
        return 4
    if "float64" in text or "double" in text or "int64" in text:
        return 8
    return None


def _shape_numel(shape: list[int] | None) -> int | None:
    if shape is None:
        return None
    numel = 1
    for dim in shape:
        numel *= dim
    return numel


def _tensor_nbytes(shape: list[int] | None, dtype_text: str | None) -> int | None:
    element_bytes = _dtype_nbytes(dtype_text)
    numel = _shape_numel(shape)
    if element_bytes is None or numel is None:
        return None
    return numel * element_bytes


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


def _active_num_reqs(input_batch: Any) -> int | None:
    return _safe_int(getattr(input_batch, "num_reqs", None))


def _slice_active(value: Any, num_reqs: int | None) -> Any:
    if value is None or num_reqs is None:
        return value
    try:
        return value[:num_reqs]
    except Exception:
        return value


def _mapped_np(
    state: Any,
    field: str,
    idx_mapping_np: Any,
    num_reqs: int | None = None,
) -> Any:
    owner = getattr(state, field, None)
    data = getattr(owner, "np", owner)
    if data is None:
        return None
    if idx_mapping_np is not None:
        try:
            return data[idx_mapping_np]
        except Exception:
            return data
    return _slice_active(data, num_reqs)


def _active_input_array(input_batch: Any, field: str) -> Any:
    return _slice_active(
        getattr(input_batch, field, None),
        _active_num_reqs(input_batch),
    )


def _active_dict_or_set_any(input_batch: Any, field: str) -> bool:
    value = getattr(input_batch, field, None)
    if value is None:
        return False
    return bool(value)


def _active_array_non_default(input_batch: Any, field: str, default: float) -> bool:
    values = _active_input_array(input_batch, field)
    if values is None:
        return False
    try:
        return bool((values != default).any())
    except Exception:
        return any(item != default for item in values)


def _scheduled_token_metadata(scheduler_output: Any) -> dict[str, int | None]:
    counts = getattr(scheduler_output, "num_scheduled_tokens", None)
    total = _safe_int(getattr(scheduler_output, "total_num_scheduled_tokens", None))
    if not counts:
        return {
            "scheduler_num_reqs": None,
            "scheduler_total_num_scheduled_tokens": total,
            "scheduler_max_scheduled_tokens": None,
        }
    values = list(counts.values())
    return {
        "scheduler_num_reqs": len(values),
        "scheduler_total_num_scheduled_tokens": total
        if total is not None
        else sum(_safe_int(value, 0) or 0 for value in values),
        "scheduler_max_scheduled_tokens": max(
            (_safe_int(value, 0) or 0) for value in values
        ),
    }


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
    num_reqs = _active_num_reqs(input_batch)
    result = {}
    if sampling_states is not None:
        for field in ("temperature", "top_k", "top_p", "min_p", "num_logprobs"):
            values = _mapped_np(sampling_states, field, idx_mapping_np, num_reqs)
            result[f"{field}_min"] = _array_min(values)
            result[f"{field}_max"] = _array_max(values)
        return result

    v2_fields = {
        "temperature": "temperature_cpu",
        "top_k": "top_k_cpu",
        "top_p": "top_p_cpu",
        "min_p": "min_p_cpu",
    }
    for field, input_field in v2_fields.items():
        values = _active_input_array(input_batch, input_field)
        result[f"{field}_min"] = _array_min(values)
        result[f"{field}_max"] = _array_max(values)
    num_logprobs = getattr(input_batch, "num_logprobs", None)
    num_logprobs_values = list(num_logprobs.values()) if num_logprobs else []
    result["num_logprobs_min"] = (
        float(min(num_logprobs_values)) if num_logprobs_values else 0.0
    )
    result["num_logprobs_max"] = (
        float(max(num_logprobs_values)) if num_logprobs_values else 0.0
    )
    return result


def _sampling_state_reasons(model_runner: Any, input_batch: Any) -> list[str]:
    sampler = getattr(model_runner, "sampler", None)
    has_v2_sampling = hasattr(input_batch, "temperature_cpu") and hasattr(
        input_batch, "top_k_cpu"
    )
    if sampler is None and not has_v2_sampling:
        return ["missing_sampler"]
    idx_mapping_np = getattr(input_batch, "idx_mapping_np", None)
    num_reqs = _active_num_reqs(input_batch)
    reasons = []

    if has_v2_sampling:
        if _active_dict_or_set_any(input_batch, "num_logprobs"):
            reasons.append("token_logprobs")
        if _active_dict_or_set_any(input_batch, "logprob_token_ids"):
            reasons.append("logprob_token_ids")
        if _active_array_non_default(input_batch, "frequency_penalties_cpu", 0.0):
            reasons.append("penalties")
        if _active_array_non_default(input_batch, "presence_penalties_cpu", 0.0):
            reasons.append("penalties")
        if _active_array_non_default(input_batch, "repetition_penalties_cpu", 1.0):
            reasons.append("penalties")
        if _active_dict_or_set_any(input_batch, "has_allowed_token_ids") or bool(
            _array_any(
                _active_input_array(input_batch, "logits_processing_needs_token_ids")
            )
        ):
            reasons.append("logit_bias_or_min_tokens")
        if _active_dict_or_set_any(input_batch, "bad_words_token_ids"):
            reasons.append("bad_words")
        if _active_dict_or_set_any(input_batch, "generators"):
            reasons.append("per_request_generators")
        return sorted(set(reasons))

    sampling_states = getattr(sampler, "sampling_states", None)
    num_logprobs = _mapped_np(sampling_states, "num_logprobs", idx_mapping_np, num_reqs)
    if _array_max(num_logprobs) not in (-1.0, None):
        reasons.append("token_logprobs")
    min_p = _mapped_np(sampling_states, "min_p", idx_mapping_np, num_reqs)
    if _array_max(min_p) not in (0.0, None):
        reasons.append("min_p")

    logprob_token_ids_state = getattr(sampler, "logprob_token_ids_state", None)
    token_ids = _mapped_np(
        logprob_token_ids_state,
        "num_token_ids",
        idx_mapping_np,
        num_reqs,
    )
    if _array_max(token_ids) not in (0.0, None):
        reasons.append("logprob_token_ids")

    penalties_state = getattr(sampler, "penalties_state", None)
    if _array_any(_mapped_np(penalties_state, "use_penalty", idx_mapping_np, num_reqs)):
        reasons.append("penalties")

    logit_bias_state = getattr(sampler, "logit_bias_state", None)
    if _array_any(
        _mapped_np(logit_bias_state, "use_logit_bias", idx_mapping_np, num_reqs)
    ):
        reasons.append("logit_bias_or_min_tokens")

    bad_words_state = getattr(sampler, "bad_words_state", None)
    bad_words = _mapped_np(bad_words_state, "num_bad_words", idx_mapping_np, num_reqs)
    if _array_max(bad_words) not in (0.0, None):
        reasons.append("bad_words")

    return reasons


def l20_logits_boundary_gate(
    model_runner: Any,
    input_batch: Any,
    grammar_output: Any,
    sample_hidden_states: Any,
    logits: Any,
    scheduler_output: Any = None,
) -> tuple[bool, list[str], dict[str, Any]]:
    """Return whether a future fused logits-boundary path should be eligible."""
    scheduled = _scheduled_token_metadata(scheduler_output)
    reasons = []
    hidden_shape = _as_shape(sample_hidden_states)
    logits_shape = _as_shape(logits)
    hidden_dtype = _dtype_text(sample_hidden_states)
    logits_dtype = _dtype_text(logits)
    hidden_element_bytes = _dtype_nbytes(hidden_dtype)
    logits_element_bytes = _dtype_nbytes(logits_dtype)
    metadata = {
        "num_reqs": _safe_int(getattr(input_batch, "num_reqs", None)),
        "num_tokens": _safe_int(getattr(input_batch, "num_tokens", None)),
        "num_draft_tokens": _safe_int(getattr(input_batch, "num_draft_tokens", None)),
        "hidden_shape": hidden_shape,
        "logits_shape": logits_shape,
        "hidden_dtype": hidden_dtype,
        "logits_dtype": logits_dtype,
        "hidden_element_bytes": hidden_element_bytes,
        "logits_element_bytes": logits_element_bytes,
        "hidden_bytes": _tensor_nbytes(hidden_shape, hidden_dtype),
        "logits_bytes": _tensor_nbytes(logits_shape, logits_dtype),
        "sampling": _sampling_metadata(model_runner, input_batch),
        **scheduled,
    }
    if hidden_shape:
        metadata["hidden_dim"] = hidden_shape[-1]
    if logits_shape:
        metadata["vocab_size"] = logits_shape[-1]

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

    scheduled_total = scheduled["scheduler_total_num_scheduled_tokens"]
    scheduled_max = scheduled["scheduler_max_scheduled_tokens"]
    if scheduled_max is not None and scheduled_max > 1:
        reasons.append("prefill")
    if scheduled_total is not None and scheduled_total != metadata["num_reqs"]:
        reasons.append("not_single_token_decode")
    elif scheduled_total is None and getattr(input_batch, "num_tokens", None) != getattr(
        input_batch,
        "num_reqs",
        None,
    ):
        reasons.append("not_single_token_decode")

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
    scheduler_output: Any = None,
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
        scheduler_output,
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
