#!/usr/bin/env python3
"""Install the opt-in L20 fused token-logprobs hook into a vLLM checkout."""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path

HELPER_NAME = "l20_top_logprobs.py"

IMPORT_MARKER = "from vllm.v1.sample.ops.logprobs import batched_count_greater_than\n"
IMPORT_PATCHED = """from vllm.v1.sample.ops.logprobs import batched_count_greater_than
from vllm.v1.sample.ops.l20_top_logprobs import (
    l20_top_logprobs_enabled,
    maybe_l20_gather_logprobs,
)
"""

RAW_LOGPROBS_MARKER = """        num_logprobs = sampling_metadata.max_num_logprobs
        if num_logprobs is not None:
            if self.logprobs_mode == LogprobsMode.RAW_LOGPROBS:
                raw_logprobs = self.compute_logprobs(logits)
            elif self.logprobs_mode == LogprobsMode.RAW_LOGITS:
                raw_logprobs = logits.clone()
"""
RAW_LOGPROBS_PATCHED = """        num_logprobs = sampling_metadata.max_num_logprobs
        l20_raw_logits_for_logprobs = None
        if num_logprobs is not None:
            if (
                self.logprobs_mode == LogprobsMode.RAW_LOGPROBS
                and l20_top_logprobs_enabled()
            ):
                l20_raw_logits_for_logprobs = logits.clone()
                raw_logprobs = None
            elif self.logprobs_mode == LogprobsMode.RAW_LOGPROBS:
                raw_logprobs = self.compute_logprobs(logits)
            elif self.logprobs_mode == LogprobsMode.RAW_LOGITS:
                raw_logprobs = logits.clone()
"""

GATHER_MARKER = """        logprobs_tensors = None if num_logprobs is None else \\
            self.gather_logprobs(raw_logprobs, num_logprobs, token_ids=sampled)
"""
GATHER_PATCHED = """        logprobs_tensors = None
        if num_logprobs is not None:
            if l20_raw_logits_for_logprobs is not None:
                logprobs_tensors = maybe_l20_gather_logprobs(
                    l20_raw_logits_for_logprobs,
                    num_logprobs,
                    token_ids=sampled,
                )
                if logprobs_tensors is None:
                    raw_logprobs = self.compute_logprobs(
                        l20_raw_logits_for_logprobs
                    )
            if logprobs_tensors is None:
                logprobs_tensors = self.gather_logprobs(
                    raw_logprobs,
                    num_logprobs,
                    token_ids=sampled,
                )
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-source",
        type=Path,
        help="Path to a vLLM source checkout root. Defaults to imported package.",
    )
    parser.add_argument("--uninstall", action="store_true")
    return parser.parse_args()


def resolve_package(vllm_source: Path | None) -> Path:
    if vllm_source is not None:
        return vllm_source.expanduser().resolve() / "vllm"
    import vllm

    return Path(inspect.getfile(vllm)).parent


def replace_once(source: str, old: str, new: str, label: str) -> str:
    if new in source:
        return source
    if old not in source:
        raise RuntimeError(f"cannot find patch point: {label}")
    return source.replace(old, new, 1)


def patch_sampler(source: str) -> str:
    source = replace_once(source, IMPORT_MARKER, IMPORT_PATCHED, "sampler import")
    source = replace_once(
        source,
        RAW_LOGPROBS_MARKER,
        RAW_LOGPROBS_PATCHED,
        "raw logprobs deferral",
    )
    return replace_once(source, GATHER_MARKER, GATHER_PATCHED, "logprobs gather")


def install(package: Path) -> None:
    helper = package / "v1" / "sample" / "ops" / HELPER_NAME
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name(HELPER_NAME), helper)

    sampler = package / "v1" / "sample" / "sampler.py"
    if not sampler.exists():
        raise RuntimeError(f"missing supported vLLM sampler: {sampler}")
    backup = sampler.with_suffix(".py.l20-top-logprobs-backup")
    if not backup.exists():
        shutil.copy2(sampler, backup)
    sampler.write_text(patch_sampler(sampler.read_text(encoding="utf-8")), encoding="utf-8")


def uninstall(package: Path) -> None:
    sampler = package / "v1" / "sample" / "sampler.py"
    backup = sampler.with_suffix(".py.l20-top-logprobs-backup")
    if backup.exists():
        shutil.copy2(backup, sampler)
    helper = package / "v1" / "sample" / "ops" / HELPER_NAME
    helper.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    package = resolve_package(args.vllm_source)
    if args.uninstall:
        uninstall(package)
        print(f"uninstalled L20 top-logprobs hook from {package}")
    else:
        install(package)
        print(f"installed L20 top-logprobs hook into {package}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
