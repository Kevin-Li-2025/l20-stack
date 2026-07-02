"""Opt-in vLLM token-logprobs hook for fused top-logprobs selection.

The hook is deliberately narrow: it only replaces RAW_LOGPROBS generated-token
gathering when the request asks for a small top-N list. Unsupported shapes
return ``None`` so the patched sampler can fall back to vLLM's native
``log_softmax`` + ``topk`` path.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import torch

from l20_stack.ops.triton_sampling import (
    logprob_topk_launch_config,
    should_use_l20_logprob_topk,
    vllm_top_logprobs_out,
)
from vllm.v1.outputs import LogprobsTensors

ENABLE_ENV = "VLLM_L20_TOP_LOGPROBS"
TRACE_ENV = "VLLM_L20_TOP_LOGPROBS_TRACE"
ALLOW_NON_L20_ENV = "VLLM_L20_TOP_LOGPROBS_ALLOW_NON_L20"

_TRACE_COUNT = 0
_WORKSPACE_CACHE: dict[tuple[Any, ...], tuple[torch.Tensor, ...]] = {}


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "0").lower() in {"1", "true", "yes", "on"}


def l20_top_logprobs_enabled() -> bool:
    return _env_flag(ENABLE_ENV)


def _trace(event: dict[str, Any]) -> None:
    global _TRACE_COUNT
    path = os.environ.get(TRACE_ENV)
    if not path:
        return
    _TRACE_COUNT += 1
    trace_path = Path(path)
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    with trace_path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(
                {
                    "schema_version": 1,
                    "timestamp_ns": time.time_ns(),
                    "sequence": _TRACE_COUNT,
                    **event,
                },
                sort_keys=True,
            )
            + "\n"
        )


def _device_reason(logits: torch.Tensor) -> str | None:
    if not logits.is_cuda:
        return "not_cuda"
    if _env_flag(ALLOW_NON_L20_ENV):
        return None
    capability = torch.cuda.get_device_capability(logits.device)
    name = torch.cuda.get_device_name(logits.device)
    if capability != (8, 9):
        return f"not_sm89:{capability[0]}{capability[1]}"
    if "L20" not in name:
        return f"not_l20:{name}"
    return None


def _workspace(
    logits: torch.Tensor,
    *,
    top_n: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    batch, vocab = int(logits.shape[0]), int(logits.shape[1])
    config = logprob_topk_launch_config(vocab, top_n, batch=batch)
    key = (
        logits.device.type,
        int(logits.device.index or 0),
        str(logits.dtype),
        batch,
        vocab,
        top_n,
        config.block_vocab,
        "vllm_top_logprobs",
    )
    cached = _WORKSPACE_CACHE.get(key)
    partial_shape = (batch, config.blocks_per_row, top_n)
    block_shape = (batch, config.blocks_per_row)
    if cached is not None and cached[0].shape == partial_shape:
        return cached
    partial_values = torch.empty(partial_shape, device=logits.device, dtype=torch.float32)
    partial_tokens = torch.empty(partial_shape, device=logits.device, dtype=torch.int64)
    partial_max = torch.empty(block_shape, device=logits.device, dtype=torch.float32)
    partial_sum_exp = torch.empty(block_shape, device=logits.device, dtype=torch.float32)
    partial_ranks = torch.empty(block_shape, device=logits.device, dtype=torch.int32)
    cached = (partial_values, partial_tokens, partial_max, partial_sum_exp, partial_ranks)
    _WORKSPACE_CACHE[key] = cached
    return cached


def maybe_l20_gather_logprobs(
    logits: torch.Tensor,
    num_logprobs: int,
    *,
    token_ids: torch.Tensor,
) -> LogprobsTensors | None:
    """Return vLLM ``LogprobsTensors`` or ``None`` when ineligible."""

    reasons: list[str] = []
    metadata: dict[str, Any] = {
        "logits_shape": list(logits.shape) if hasattr(logits, "shape") else None,
        "logits_dtype": str(getattr(logits, "dtype", None)),
        "num_logprobs": int(num_logprobs),
        "token_ids_shape": list(token_ids.shape) if hasattr(token_ids, "shape") else None,
    }
    if not _env_flag(ENABLE_ENV):
        reasons.append("disabled")
    if logits.ndim != 2:
        reasons.append("not_2d_logits")
    if token_ids.ndim != 1:
        reasons.append("token_ids_not_1d")
    elif token_ids.shape[0] != logits.shape[0]:
        reasons.append("token_ids_batch_mismatch")
    if token_ids.dtype != torch.int64:
        reasons.append("token_ids_not_int64")
    if not token_ids.is_cuda:
        reasons.append("token_ids_not_cuda")
    device_reason = _device_reason(logits)
    if device_reason is not None:
        reasons.append(device_reason)
    if num_logprobs <= 0:
        reasons.append("non_positive_num_logprobs")

    if not reasons:
        batch, vocab = int(logits.shape[0]), int(logits.shape[1])
        metadata["batch"] = batch
        metadata["vocab"] = vocab
        if not should_use_l20_logprob_topk(batch, vocab, int(num_logprobs)):
            reasons.append("outside_l20_logprob_gate")

    if reasons:
        _trace({"eligible": False, "reasons": reasons, "metadata": metadata})
        return None

    top_n = int(num_logprobs)
    batch = int(logits.shape[0])
    output_token_ids = torch.empty(
        (batch, top_n + 1), device=logits.device, dtype=torch.int32
    )
    output_logprobs = torch.empty(
        (batch, top_n + 1), device=logits.device, dtype=torch.float32
    )
    output_ranks = torch.empty((batch,), device=logits.device, dtype=torch.int32)
    partial_values, partial_tokens, partial_max, partial_sum_exp, partial_ranks = _workspace(
        logits, top_n=top_n
    )
    vllm_top_logprobs_out(
        logits,
        token_ids,
        output_token_ids,
        output_logprobs,
        output_ranks,
        partial_values=partial_values,
        partial_tokens=partial_tokens,
        partial_max=partial_max,
        partial_sum_exp=partial_sum_exp,
        partial_ranks=partial_ranks,
        top_n=top_n,
        temperature=1.0,
    )
    _trace({"eligible": True, "reasons": [], "metadata": metadata})
    return LogprobsTensors(output_token_ids, output_logprobs, output_ranks)
