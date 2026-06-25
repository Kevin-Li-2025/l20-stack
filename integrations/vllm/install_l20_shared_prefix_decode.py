#!/usr/bin/env python3
"""Install the experimental L20 shared-prefix decode ops into vLLM."""

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
        backend = None
    targets = {
        "l20_decode_attention.py": Path("src/l20_stack/ops/triton_decode_attention.py"),
        "l20_shared_prefix_decode_dispatch.py": Path(
            "integrations/vllm/l20_shared_prefix_decode_dispatch.py"
        ),
    }
    backups = {
        name: (op_dir / name).with_suffix(".py.l20-shared-prefix-backup")
        for name in targets
    }
    backend_backup = (
        backend.with_suffix(".py.l20-shared-prefix-backup")
        if backend is not None
        else None
    )
    if args.uninstall:
        for name, backup in backups.items():
            target = op_dir / name
            if backup.exists():
                shutil.copy2(backup, target)
            elif target.exists():
                target.unlink()
        if backend is not None and backend_backup is not None and backend_backup.exists():
            shutil.copy2(backend_backup, backend)
        return 0

    root = Path(__file__).resolve().parents[2]
    install_dirs = [op_dir]
    if source_tree:
        source_op_dir = Path(source_tree) / "vllm" / "v1" / "attention" / "ops"
        if source_op_dir.exists() and source_op_dir != op_dir:
            install_dirs.append(source_op_dir)
    for install_dir in install_dirs:
        install_dir.mkdir(parents=True, exist_ok=True)
        for name, source in targets.items():
            target = install_dir / name
            backup = target.with_suffix(".py.l20-shared-prefix-backup")
            if target.exists() and not backup.exists():
                shutil.copy2(target, backup)
            shutil.copy2(root / source, target)
    if backend is None or not backend.exists():
        return 0
    if backend_backup is not None and not backend_backup.exists():
        shutil.copy2(backend, backend_backup)
    source = backend.read_text(encoding="utf-8")
    source = replace_once(
        source,
        "import torch\n",
        (
            "import torch\n"
            "from vllm.v1.attention.ops.l20_shared_prefix_decode_dispatch import "
            "trace_l20_shared_prefix_decode_candidate\n"
        ),
        "shared-prefix decode import",
    )
    source = replace_once(
        source,
        """                else:
                    l20_batch = decode_query.shape[0]
                    l20_max_seq = attn_metadata.decode.max_seq_len
""",
        """                else:
                    trace_l20_shared_prefix_decode_candidate(
                        decode_query,
                        kv_cache_permute,
                        attn_metadata.decode.block_tables,
                        attn_metadata.decode.seq_lens,
                        page_size=kv_cache_permute.shape[2],
                    )
                    l20_batch = decode_query.shape[0]
                    l20_max_seq = attn_metadata.decode.max_seq_len
""",
        "native decode trace hook",
    )
    backend.write_text(source, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
