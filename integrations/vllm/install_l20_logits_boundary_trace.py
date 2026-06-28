#!/usr/bin/env python3
"""Install a trace-only L20 logits-boundary gate into vLLM.

The patch is disabled unless ``VLLM_L20_LOGITS_BOUNDARY_TRACE`` points to a
JSONL file. It does not change sampling behavior; it only records whether a
future LM-head epilogue / logits-boundary fast path would be eligible.
"""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path


IMPORT_LINE = (
    "from vllm.v1.worker.gpu.l20_logits_boundary_trace import "
    "maybe_trace_l20_logits_boundary\n"
)

IMPORT_MARKER = "from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker\n"
V2_IMPORT_MARKER = "from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch\n"

SAMPLE_PATCH_POINT = """        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
"""

SAMPLE_PATCHED = """        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        maybe_trace_l20_logits_boundary(
            self,
            input_batch,
            grammar_output,
            sample_hidden_states,
            logits,
        )
        if grammar_output is not None:
"""

V2_SAMPLE_PATCH_POINT = """        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
"""

V2_SAMPLE_PATCHED = """        # Clear ephemeral state.
        self.execute_model_state = None

        maybe_trace_l20_logits_boundary(
            self,
            self.input_batch,
            grammar_output,
            sample_hidden_states,
            logits,
            scheduler_output,
        )

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
"""


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--vllm-source",
        type=Path,
        help="Path to a vLLM source checkout root. Defaults to the imported package.",
    )
    parser.add_argument("--uninstall", action="store_true")
    return parser.parse_args()


def resolve_package(vllm_source: Path | None) -> Path:
    if vllm_source is not None:
        return vllm_source.expanduser().resolve() / "vllm"
    import vllm

    return Path(inspect.getfile(vllm)).parent


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if new in text:
        return text
    if old not in text:
        raise RuntimeError(f"cannot find patch point: {label}")
    return text.replace(old, new, 1)


def patch_model_runner(source: str) -> str:
    source = replace_once(
        source,
        IMPORT_MARKER,
        IMPORT_MARKER + IMPORT_LINE,
        "model_runner trace import",
    )
    return replace_once(
        source,
        SAMPLE_PATCH_POINT,
        SAMPLE_PATCHED,
        "GPUModelRunner.sample logits boundary",
    )


def patch_gpu_model_runner(source: str) -> str:
    source = replace_once(
        source,
        V2_IMPORT_MARKER,
        V2_IMPORT_MARKER + IMPORT_LINE,
        "v2 gpu_model_runner trace import",
    )
    return replace_once(
        source,
        V2_SAMPLE_PATCH_POINT,
        V2_SAMPLE_PATCHED,
        "GPUModelRunner.sample_tokens logits boundary",
    )


def _install_target(path: Path, patcher) -> bool:
    if not path.exists():
        return False
    backup = path.with_suffix(".py.l20-logits-boundary-trace-backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patcher(path.read_text(encoding="utf-8")), encoding="utf-8")
    return True


def _restore_target(path: Path) -> bool:
    backup = path.with_suffix(".py.l20-logits-boundary-trace-backup")
    if not backup.exists():
        return False
    shutil.copy2(backup, path)
    return True


def install(package: Path) -> None:
    targets = [
        (package / "v1" / "worker" / "gpu" / "model_runner.py", patch_model_runner),
        (package / "v1" / "worker" / "gpu_model_runner.py", patch_gpu_model_runner),
    ]
    helper = package / "v1" / "worker" / "gpu" / "l20_logits_boundary_trace.py"
    shutil.copy2(Path(__file__).with_name("l20_logits_boundary_trace.py"), helper)
    patched = [path for path, patcher in targets if _install_target(path, patcher)]
    if not patched:
        raise RuntimeError(f"missing supported vLLM model runner under: {package}")


def uninstall(package: Path) -> None:
    targets = [
        package / "v1" / "worker" / "gpu" / "model_runner.py",
        package / "v1" / "worker" / "gpu_model_runner.py",
    ]
    helper = package / "v1" / "worker" / "gpu" / "l20_logits_boundary_trace.py"
    for target in targets:
        _restore_target(target)
    helper.unlink(missing_ok=True)


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
