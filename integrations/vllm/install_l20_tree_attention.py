#!/usr/bin/env python3
"""Install the experimental L20 tree-attention op into a local vLLM checkout."""

from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
from pathlib import Path

import vllm


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise RuntimeError(f"cannot find patch point: {label}")
    return source.replace(old, new, 1)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    package = Path(next(iter(vllm.__path__)))
    op_dir = package / "v1" / "attention" / "ops"
    backend_spec = importlib.util.find_spec("vllm.v1.attention.backends.flashinfer")
    source_tree = os.getenv("VLLM_SOURCE_TREE")
    if backend_spec is not None and backend_spec.origin is not None:
        backend = Path(backend_spec.origin)
    elif source_tree:
        backend = Path(source_tree) / "vllm" / "v1" / "attention" / "backends" / "flashinfer.py"
    else:
        raise RuntimeError(
            "cannot locate vllm.v1.attention.backends.flashinfer; "
            "set VLLM_SOURCE_TREE to a vLLM checkout"
        )
    if not backend.exists():
        raise RuntimeError(f"cannot find FlashInfer backend: {backend}")
    target = op_dir / "l20_tree_attention.py"
    dispatch_target = op_dir / "l20_tree_attention_dispatch.py"
    backup = target.with_suffix(".py.l20-tree-backup")
    dispatch_backup = dispatch_target.with_suffix(".py.l20-tree-backup")
    backend_backup = backend.with_suffix(".py.l20-tree-backup")
    if args.uninstall:
        if backup.exists():
            shutil.copy2(backup, target)
        elif target.exists():
            target.unlink()
        if dispatch_backup.exists():
            shutil.copy2(dispatch_backup, dispatch_target)
        elif dispatch_target.exists():
            dispatch_target.unlink()
        if backend_backup.exists():
            shutil.copy2(backend_backup, backend)
        return 0
    if target.exists() and not backup.exists():
        shutil.copy2(target, backup)
    if dispatch_target.exists() and not dispatch_backup.exists():
        shutil.copy2(dispatch_target, dispatch_backup)
    if backend.exists() and not backend_backup.exists():
        shutil.copy2(backend, backend_backup)
    root = Path(__file__).resolve().parents[2]
    install_dirs = [op_dir]
    if source_tree:
        source_op_dir = Path(source_tree) / "vllm" / "v1" / "attention" / "ops"
        if source_op_dir.exists() and source_op_dir != op_dir:
            install_dirs.append(source_op_dir)
    for install_dir in install_dirs:
        shutil.copy2(
            root / "src/l20_stack/ops/triton_tree_attention.py",
            install_dir / "l20_tree_attention.py",
        )
        shutil.copy2(
            root / "integrations/vllm/l20_tree_attention_dispatch.py",
            install_dir / "l20_tree_attention_dispatch.py",
        )
    source = backend.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "import torch\n",
        (
            "import json\n"
            "import os\n"
            "import time\n"
            "import torch\n"
            "from vllm.v1.attention.ops.l20_tree_attention_dispatch import "
            "maybe_l20_causal_verifier_attention, maybe_l20_tree_attention\n"
        ),
        "l20 tree attention dispatch import",
    )
    source = replace_once(
        source,
        "logger = init_logger(__name__)\n",
        """logger = init_logger(__name__)


def _l20_tree_trace(event, **fields):
    path = os.getenv("VLLM_L20_TREE_ATTENTION_TRACE")
    if not path:
        return
    payload = {"event": event, "ts": time.time(), **fields}
    try:
        with open(path, "a", encoding="utf-8") as trace_file:
            trace_file.write(json.dumps(payload, sort_keys=True) + "\\n")
    except OSError:
        logger.warning("Failed to write L20 tree-attention trace to %s", path)


def _l20_tree_timing_enabled():
    return os.getenv("VLLM_L20_TREE_ATTENTION_TIMING", "0") == "1"


def _l20_tree_cuda_event_ms(function):
    if not _l20_tree_timing_enabled():
        function()
        return None
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    function()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def maybe_run_l20_tree_attention(
    query,
    key_cache,
    value_cache,
    suffix_key,
    suffix_value,
    block_table,
    ancestor_mask,
    cached_length,
    *,
    workspace=None,
):
    return maybe_l20_tree_attention(
        query,
        key_cache,
        value_cache,
        suffix_key,
        suffix_value,
        block_table,
        ancestor_mask,
        cached_length,
        workspace=workspace,
    )


def maybe_run_l20_causal_verifier_from_prefill(
    prefill_query,
    kv_cache,
    suffix_key,
    suffix_value,
    block_tables,
    seq_lens,
    out,
    max_seq_len=None,
):
    if prefill_query.ndim != 3 or block_tables.shape[0] != 1:
        _l20_tree_trace(
            "causal_verifier_skip",
            reason="shape",
            prefill_query_ndim=prefill_query.ndim,
            block_tables_shape=list(block_tables.shape),
        )
        return False
    draft_length = prefill_query.shape[0]
    seq_len = max_seq_len if max_seq_len is not None else int(seq_lens[0].item())
    cached_length = int(seq_len) - draft_length
    if cached_length <= 0:
        _l20_tree_trace(
            "causal_verifier_skip",
            reason="cached_length",
            draft_length=int(draft_length),
            seq_len=int(seq_len),
            cached_length=int(cached_length),
        )
        return False
    key_cache, value_cache = kv_cache.unbind(1)
    result_holder = [None]
    elapsed_ms = _l20_tree_cuda_event_ms(
        lambda: result_holder.__setitem__(
            0,
            maybe_l20_causal_verifier_attention(
                prefill_query.unsqueeze(0),
                key_cache,
                value_cache,
                suffix_key.unsqueeze(0),
                suffix_value.unsqueeze(0),
                block_tables,
                cached_length,
            ),
        )
    )
    result = result_holder[0]
    if result is None:
        _l20_tree_trace(
            "causal_verifier_skip",
            reason="dispatch_gate",
            draft_length=int(draft_length),
            seq_len=int(seq_len),
            cached_length=int(cached_length),
        )
        return False
    out.copy_(result.squeeze(0))
    if elapsed_ms is not None:
        _l20_tree_trace(
            "causal_verifier_timing",
            elapsed_ms=elapsed_ms,
            draft_length=int(draft_length),
            seq_len=int(seq_len),
            cached_length=int(cached_length),
        )
    _l20_tree_trace(
        "causal_verifier_run",
        draft_length=int(draft_length),
        seq_len=int(seq_len),
        cached_length=int(cached_length),
    )
    return True


def maybe_run_l20_tree_attention_from_prefill(
    prefill_query,
    kv_cache,
    suffix_key,
    suffix_value,
    block_tables,
    seq_lens,
    out,
    max_seq_len=None,
):
    if prefill_query.ndim != 3 or block_tables.shape[0] != 1:
        _l20_tree_trace(
            "prefill_hook_skip",
            reason="shape",
            prefill_query_ndim=prefill_query.ndim,
            block_tables_shape=list(block_tables.shape),
        )
        return False
    draft_length = prefill_query.shape[0]
    seq_len = max_seq_len if max_seq_len is not None else int(seq_lens[0].item())
    cached_length = int(seq_len) - draft_length
    if cached_length <= 0:
        _l20_tree_trace(
            "prefill_hook_skip",
            reason="cached_length",
            draft_length=int(draft_length),
            seq_len=int(seq_len),
            cached_length=int(cached_length),
        )
        return False
    ancestor_positions = torch.arange(draft_length, device=prefill_query.device)
    ancestor_mask = ancestor_positions[None, :] <= ancestor_positions[:, None]
    key_cache, value_cache = kv_cache.unbind(1)
    result_holder = [None]
    elapsed_ms = _l20_tree_cuda_event_ms(
        lambda: result_holder.__setitem__(
            0,
            maybe_run_l20_tree_attention(
                prefill_query.unsqueeze(0),
                key_cache,
                value_cache,
                suffix_key.unsqueeze(0),
                suffix_value.unsqueeze(0),
                block_tables,
                ancestor_mask,
                cached_length,
            ),
        )
    )
    result = result_holder[0]
    if result is None:
        _l20_tree_trace(
            "prefill_hook_skip",
            reason="dispatch_gate",
            draft_length=int(draft_length),
            seq_len=int(seq_len),
            cached_length=int(cached_length),
        )
        return False
    out.copy_(result.squeeze(0))
    if elapsed_ms is not None:
        _l20_tree_trace(
            "prefill_hook_timing",
            elapsed_ms=elapsed_ms,
            draft_length=int(draft_length),
            seq_len=int(seq_len),
            cached_length=int(cached_length),
        )
    _l20_tree_trace(
        "prefill_hook_run",
        draft_length=int(draft_length),
        seq_len=int(seq_len),
        cached_length=int(cached_length),
    )
    return True
""",
        "l20 tree attention backend hook",
    )
    source = replace_once(
        source,
        '''class FIPrefill:
    """Metadata for the native FlashInfer prefill pathway (non-TRTLLM)."""

    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper
''',
        '''class FIPrefill:
    """Metadata for the native FlashInfer prefill pathway (non-TRTLLM)."""

    wrapper: BatchPrefillWithPagedKVCacheWrapper | BatchDCPPrefillWrapper
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    max_seq_len: int
''',
        "FIPrefill metadata",
    )
    source = replace_once(
        source,
        "attn_metadata.prefill = FIPrefill(wrapper=prefill_wrapper)",
        """attn_metadata.prefill = FIPrefill(
                    wrapper=prefill_wrapper,
                    block_tables=block_table_tensor[prefill_start:],
                    seq_lens=seq_lens[prefill_start:],
                    max_seq_len=max_seq_len,
                )""",
        "FIPrefill construction",
    )
    source = replace_once(
        source,
        """                    prefill_wrapper.run(
                        prefill_query,
                        kv_cache_permute,
                        k_scale=layer._k_scale_float,
                        v_scale=layer._v_scale_float,
                        out=out_prefill,
                        kv_cache_sf=kv_cache_sf,
                    )
""",
        """                    _l20_tree_trace(
                        "native_prefill_site",
                        causal=bool(attn_metadata.causal),
                        is_kvcache_nvfp4=bool(self.is_kvcache_nvfp4),
                        prefill_tokens=int(prefill_query.shape[0]),
                        num_decode_tokens=int(num_decode_tokens),
                        max_seq_len=int(attn_metadata.prefill.max_seq_len),
                    )
                    l20_tree_ran = (
                        not self.is_kvcache_nvfp4
                        and (
                            (
                                attn_metadata.causal
                                and maybe_run_l20_causal_verifier_from_prefill(
                                    prefill_query,
                                    kv_cache_permute,
                                    key[num_decode_tokens:],
                                    value[num_decode_tokens:],
                                    attn_metadata.prefill.block_tables,
                                    attn_metadata.prefill.seq_lens,
                                    out_prefill,
                                    attn_metadata.prefill.max_seq_len,
                                )
                            )
                            or (
                                not attn_metadata.causal
                                and maybe_run_l20_tree_attention_from_prefill(
                                    prefill_query,
                                    kv_cache_permute,
                                    key[num_decode_tokens:],
                                    value[num_decode_tokens:],
                                    attn_metadata.prefill.block_tables,
                                    attn_metadata.prefill.seq_lens,
                                    out_prefill,
                                    attn_metadata.prefill.max_seq_len,
                                )
                            )
                        )
                    )
                    if not l20_tree_ran:
                        native_elapsed_ms = _l20_tree_cuda_event_ms(
                            lambda: prefill_wrapper.run(
                                prefill_query,
                                kv_cache_permute,
                                k_scale=layer._k_scale_float,
                                v_scale=layer._v_scale_float,
                                out=out_prefill,
                                kv_cache_sf=kv_cache_sf,
                            )
                        )
                        if native_elapsed_ms is not None:
                            _l20_tree_trace(
                                "native_prefill_timing",
                                elapsed_ms=native_elapsed_ms,
                                causal=bool(attn_metadata.causal),
                                prefill_tokens=int(prefill_query.shape[0]),
                                num_decode_tokens=int(num_decode_tokens),
                                max_seq_len=int(attn_metadata.prefill.max_seq_len),
                            )
""",
        "native prefill dispatch",
    )
    backend.write_text(source, encoding="utf-8")
    print(target)
    print(dispatch_target)
    print(backend)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
