#!/usr/bin/env python3
"""Install the opt-in L20 FlashSampling candidate path into vLLM."""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path

HELPER_SOURCE = Path(__file__).with_name("l20_flashsampling_candidate.py")
HELPER_NAME = "l20_flashsampling_candidate.py"
BACKUP_SUFFIX = ".l20-flashsampling-candidate-backup"

IMPORT_LINE = (
    "from vllm.v1.worker.gpu.l20_flashsampling_candidate import "
    "maybe_l20_flashsampling_compute_logits_or_sample, "
    "maybe_take_l20_flashsampling_sampler_output\n"
)
V2_IMPORT_MARKER = "from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch\n"
NATIVE_IMPORT_MARKER = "from vllm.v1.worker.gpu.sample.output import SamplerOutput\n"
COMPUTE_PATCH_POINT = """                sample_hidden_states = hidden_states[logits_indices]
                logits = self.model.compute_logits(sample_hidden_states)
"""
COMPUTE_PATCHED = """                sample_hidden_states = hidden_states[logits_indices]
                logits = maybe_l20_flashsampling_compute_logits_or_sample(
                    self,
                    self.input_batch,
                    scheduler_output,
                    spec_decode_metadata,
                    sample_hidden_states,
                    self.model.compute_logits,
                )
"""
SAMPLE_PATCH_POINT = """        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )

        with record_function_or_nullcontext("gpu_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)
"""
SAMPLE_PATCHED = """        sampler_output = maybe_take_l20_flashsampling_sampler_output(
            self,
            grammar_output,
        )
        if sampler_output is None:
            # Apply structured output bitmasks if present.
            if grammar_output is not None:
                apply_grammar_bitmask(
                    scheduler_output, grammar_output, self.input_batch, logits
                )

            with record_function_or_nullcontext("gpu_model_runner: sample"):
                sampler_output = self._sample(logits, spec_decode_metadata)
"""
NATIVE_COMPUTE_PATCH_POINT = """        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
"""
NATIVE_COMPUTE_PATCHED = """        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = maybe_l20_flashsampling_compute_logits_or_sample(
            self,
            input_batch,
            None,
            None,
            sample_hidden_states,
            self.model.compute_logits,
        )
"""
NATIVE_SAMPLE_PATCH_POINT = """        if grammar_output is not None:
            # Apply grammar bitmask to the logits in-place.
            assert self.structured_outputs_worker is not None
            self.structured_outputs_worker.apply_grammar_bitmask(
                logits,
                input_batch,
                grammar_output.structured_output_request_ids,
                grammar_output.grammar_bitmask,
            )

        if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
            assert self.sampler is not None
            sampler_output = self.sampler(logits, input_batch)
        else:
            # Rejection sampling for spec decoding.
            assert self.rejection_sampler is not None
            assert self.speculator is not None
            sampler_output = self.rejection_sampler(
                logits,
                input_batch,
                # Draft logits are needed for probabilistic rejection sampling.
                self.speculator.draft_logits,
            )
"""
NATIVE_SAMPLE_PATCHED = """        sampler_output = maybe_take_l20_flashsampling_sampler_output(
            self,
            grammar_output,
        )
        if sampler_output is None:
            if grammar_output is not None:
                # Apply grammar bitmask to the logits in-place.
                assert self.structured_outputs_worker is not None
                self.structured_outputs_worker.apply_grammar_bitmask(
                    logits,
                    input_batch,
                    grammar_output.structured_output_request_ids,
                    grammar_output.grammar_bitmask,
                )

            if input_batch.num_draft_tokens == 0 or self.rejection_sampler is None:
                assert self.sampler is not None
                sampler_output = self.sampler(logits, input_batch)
            else:
                # Rejection sampling for spec decoding.
                assert self.rejection_sampler is not None
                assert self.speculator is not None
                sampler_output = self.rejection_sampler(
                    logits,
                    input_batch,
                    # Draft logits are needed for probabilistic rejection sampling.
                    self.speculator.draft_logits,
                )
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-source", type=Path)
    parser.add_argument("--uninstall", action="store_true")
    return parser.parse_args()


def resolve_package(vllm_source: Path | None) -> Path:
    if vllm_source is not None:
        return vllm_source.expanduser().resolve() / "vllm"
    import vllm

    return Path(inspect.getfile(vllm)).parent


def _backup_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + BACKUP_SUFFIX)


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"cannot find patch point: {label}")
    return text.replace(old, new, 1)


def patch_gpu_model_runner(text: str) -> str:
    if IMPORT_LINE not in text:
        text = replace_once(text, V2_IMPORT_MARKER, V2_IMPORT_MARKER + IMPORT_LINE, "candidate import")
    text = replace_once(text, COMPUTE_PATCH_POINT, COMPUTE_PATCHED, "candidate compute_logits")
    text = replace_once(text, SAMPLE_PATCH_POINT, SAMPLE_PATCHED, "candidate sampler output")
    return text


def patch_native_gpu_model_runner(text: str) -> str:
    if IMPORT_LINE not in text:
        text = replace_once(
            text,
            NATIVE_IMPORT_MARKER,
            NATIVE_IMPORT_MARKER + IMPORT_LINE,
            "native candidate import",
        )
    text = replace_once(
        text,
        NATIVE_COMPUTE_PATCH_POINT,
        NATIVE_COMPUTE_PATCHED,
        "native candidate compute_logits",
    )
    text = replace_once(
        text,
        NATIVE_SAMPLE_PATCH_POINT,
        NATIVE_SAMPLE_PATCHED,
        "native candidate sampler output",
    )
    return text


def _patch_file(target: Path, patcher) -> None:
    if not target.exists():
        return
    original = target.read_text(encoding="utf-8")
    patched = patcher(original)
    if patched != original:
        backup = _backup_path(target)
        if not backup.exists():
            shutil.copy2(target, backup)
        target.write_text(patched, encoding="utf-8")


def install(package: Path) -> None:
    if not HELPER_SOURCE.exists():
        raise RuntimeError(f"missing helper source: {HELPER_SOURCE}")
    target = package / "v1" / "worker" / "gpu_model_runner.py"
    if not target.exists():
        raise RuntimeError(f"missing vLLM gpu_model_runner: {target}")
    helper_target = package / "v1" / "worker" / "gpu" / HELPER_NAME
    helper_target.parent.mkdir(parents=True, exist_ok=True)
    if helper_target.exists() and helper_target.read_bytes() != HELPER_SOURCE.read_bytes():
        backup = _backup_path(helper_target)
        if not backup.exists():
            shutil.copy2(helper_target, backup)
    shutil.copy2(HELPER_SOURCE, helper_target)
    _patch_file(target, patch_gpu_model_runner)
    _patch_file(package / "v1" / "worker" / "gpu" / "model_runner.py", patch_native_gpu_model_runner)


def uninstall(package: Path) -> None:
    for target in (
        package / "v1" / "worker" / "gpu_model_runner.py",
        package / "v1" / "worker" / "gpu" / "model_runner.py",
    ):
        backup = _backup_path(target)
        if backup.exists():
            shutil.copy2(backup, target)
    helper_target = package / "v1" / "worker" / "gpu" / HELPER_NAME
    helper_backup = _backup_path(helper_target)
    if helper_backup.exists():
        shutil.copy2(helper_backup, helper_target)
    else:
        helper_target.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    package = resolve_package(args.vllm_source)
    if args.uninstall:
        uninstall(package)
    else:
        install(package)
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
