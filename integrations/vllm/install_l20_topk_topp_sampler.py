#!/usr/bin/env python3
"""Install the opt-in L20 top-k/top-p sampler hook into a vLLM checkout."""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path


IMPORT_LINE = (
    "from vllm.v1.sample.ops.l20_topk_topp_sampling import "
    "maybe_l20_topk_topp_sample\n"
)

TOPK_IMPORT_MARKER = "from vllm.triton_utils import HAS_TRITON\n"
FLASHINFER_PATCH_POINT = """    assert not (k is None and p is None)
    if k is None:
"""
FLASHINFER_PATCHED = """    assert not (k is None and p is None)
    if k is None:
"""

TOPK_FORWARD_SIGNATURE = """    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
"""
TOPK_FORWARD_SIGNATURE_PATCHED = """    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
        *,
        l20_expanded_idx_mapping: torch.Tensor | None = None,
        l20_seeds: torch.Tensor | None = None,
        l20_positions: torch.Tensor | None = None,
        l20_history_tokens: torch.Tensor | None = None,
        l20_history_lengths: torch.Tensor | None = None,
        l20_defer_penalties: bool = False,
        l20_frequency_penalties: torch.Tensor | None = None,
        l20_presence_penalties: torch.Tensor | None = None,
        l20_repetition_penalties: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
"""
TOPK_FLASHINFER_RETURN = """        return flashinfer_sample(logits.contiguous(), k, p, generators), None
"""
TOPK_FLASHINFER_RETURN_PATCHED = """        contiguous_logits = logits.contiguous()
        l20_sampled = maybe_l20_topk_topp_sample(
            contiguous_logits,
            k,
            p,
            generators,
            expanded_idx_mapping=l20_expanded_idx_mapping,
            seeds=l20_seeds,
            positions=l20_positions,
            history_tokens=l20_history_tokens,
            history_lengths=l20_history_lengths,
            frequency_penalties=l20_frequency_penalties,
            presence_penalties=l20_presence_penalties,
            repetition_penalties=l20_repetition_penalties,
            defer_penalties=l20_defer_penalties,
        )
        if l20_sampled is not None:
            return l20_sampled, None
        return flashinfer_sample(contiguous_logits, k, p, generators), None
"""

METADATA_GENERATORS = """    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
"""
METADATA_GENERATORS_PATCHED = """    generators: dict[int, torch.Generator]

    # L20 experimental sampler state. These are None for upstream/default paths.
    l20_expanded_idx_mapping: torch.Tensor | None
    l20_seeds: torch.Tensor | None
    l20_positions: torch.Tensor | None
    l20_history_tokens: torch.Tensor | None
    l20_history_lengths: torch.Tensor | None
    l20_defer_penalties: bool

    # None means no logprobs, 0 means sampled token logprobs only
"""

SAMPLER_TOPK_CALL = """        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )
"""
SAMPLER_TOPK_CALL_PATCHED = """        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
            l20_expanded_idx_mapping=sampling_metadata.l20_expanded_idx_mapping,
            l20_seeds=sampling_metadata.l20_seeds,
            l20_positions=sampling_metadata.l20_positions,
            l20_history_tokens=sampling_metadata.l20_history_tokens,
            l20_history_lengths=sampling_metadata.l20_history_lengths,
            l20_defer_penalties=sampling_metadata.l20_defer_penalties,
            l20_frequency_penalties=sampling_metadata.frequency_penalties,
            l20_presence_penalties=sampling_metadata.presence_penalties,
            l20_repetition_penalties=sampling_metadata.repetition_penalties,
        )
"""

DUMMY_METADATA_MARKER = """            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
"""
DUMMY_METADATA_PATCHED = """            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            l20_expanded_idx_mapping=torch.arange(
                num_reqs, dtype=torch.int64, device=self.device
            ),
            l20_seeds=torch.full(
                (num_reqs,), 1, dtype=torch.int64, device=self.device
            ),
            l20_positions=torch.arange(
                num_reqs, dtype=torch.int64, device=self.device
            ),
            l20_history_tokens=None,
            l20_history_lengths=None,
            l20_defer_penalties=False,
"""

INPUT_BATCH_TOPK_REQS = """        self.top_k_reqs: set[str] = set()

        # Frequency penalty related data structures
"""
INPUT_BATCH_TOPK_REQS_PATCHED = """        self.top_k_reqs: set[str] = set()

        self.l20_sampler_seeds = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device=device
        )
        self.l20_sampler_seeds_cpu_tensor = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device=\"cpu\", pin_memory=PIN_MEMORY
        )
        self.l20_sampler_seeds_cpu = self.l20_sampler_seeds_cpu_tensor.numpy()
        self.l20_sampler_positions = torch.empty(
            (max_num_reqs,), dtype=torch.int64, device=device
        )
        self.l20_sampler_indices = torch.arange(
            max_num_reqs, dtype=torch.int64, device=device
        )

        # Frequency penalty related data structures
"""

INPUT_BATCH_TOPK_CPU = """            self.top_k_cpu[req_index] = top_k
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
"""
INPUT_BATCH_TOPK_CPU_PATCHED = """            self.top_k_cpu[req_index] = top_k
            l20_seed = sampling_params.seed
            if l20_seed is None:
                l20_seed = np.random.randint(
                    np.iinfo(np.int64).min, np.iinfo(np.int64).max
                )
            self.l20_sampler_seeds_cpu[req_index] = l20_seed
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
"""

INPUT_BATCH_COPY_TOPK = """        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        if not self.no_penalties:
"""
INPUT_BATCH_COPY_TOPK_PATCHED = """        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)
        copy_slice(self.l20_sampler_seeds_cpu_tensor, self.l20_sampler_seeds, num_reqs)
        copy_slice(
            self.num_tokens_no_spec_cpu_tensor, self.l20_sampler_positions, num_reqs
        )

        if not self.no_penalties:
"""

INPUT_BATCH_METADATA_GENERATORS = """            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
"""
INPUT_BATCH_METADATA_GENERATORS_PATCHED = """            generators=self.generators,
            l20_expanded_idx_mapping=self.l20_sampler_indices[:num_reqs],
            l20_seeds=self.l20_sampler_seeds[:num_reqs],
            l20_positions=self.l20_sampler_positions[:num_reqs],
            l20_history_tokens=None,
            l20_history_lengths=None,
            l20_defer_penalties=False,
            max_num_logprobs=self.max_num_logprobs,
"""

WORKER_IMPORT_MARKER = "from vllm.v1.sample.ops.topk_topp_sampler import (\n"
WORKER_PATCH_POINT = """        if use_flashinfer:
            sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
"""
WORKER_PATCHED = """        if use_flashinfer:
            l20_top_k_values = self.sampling_states.top_k.np[idx_mapping_np]
            l20_top_p_values = self.sampling_states.top_p.np[idx_mapping_np]
            l20_top_k_uniform = bool((l20_top_k_values == l20_top_k_values[0]).all())
            l20_top_p_uniform = bool((l20_top_p_values == l20_top_p_values[0]).all())
            l20_sampled = None
            if l20_top_k_uniform and l20_top_p_uniform:
                l20_sampled = maybe_l20_topk_topp_sample(
                    processed_logits,
                    top_k,
                    top_p,
                    expanded_idx_mapping=expanded_idx_mapping,
                    seeds=self.sampling_states.seeds.gpu,
                    positions=pos,
                    top_k_value=int(l20_top_k_values[0]),
                    top_p_value=float(l20_top_p_values[0]),
                )
            if l20_sampled is not None:
                sampled = l20_sampled.to(torch.int64)
            else:
                sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
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


def patch_topk_topp_sampler(source: str) -> str:
    source = replace_once(
        source,
        TOPK_IMPORT_MARKER,
        TOPK_IMPORT_MARKER + IMPORT_LINE,
        "topk_topp_sampler import",
    )
    source = replace_once(
        source,
        FLASHINFER_PATCH_POINT,
        FLASHINFER_PATCHED,
        "flashinfer_sample hook",
    )
    source = replace_once(
        source,
        TOPK_FORWARD_SIGNATURE,
        TOPK_FORWARD_SIGNATURE_PATCHED,
        "topk_topp forward_cuda signature",
    )
    return replace_once(
        source,
        TOPK_FLASHINFER_RETURN,
        TOPK_FLASHINFER_RETURN_PATCHED,
        "topk_topp forward_cuda l20 hook",
    )


def patch_sampling_metadata(source: str) -> str:
    return replace_once(
        source,
        METADATA_GENERATORS,
        METADATA_GENERATORS_PATCHED,
        "sampling metadata l20 state",
    )


def patch_active_sampler(source: str) -> str:
    return replace_once(
        source,
        SAMPLER_TOPK_CALL,
        SAMPLER_TOPK_CALL_PATCHED,
        "active sampler topk_topp state pass",
    )


def patch_gpu_model_runner(source: str) -> str:
    return replace_once(
        source,
        DUMMY_METADATA_MARKER,
        DUMMY_METADATA_PATCHED,
        "dummy sampler metadata l20 state",
    )


def patch_gpu_input_batch(source: str) -> str:
    source = replace_once(
        source,
        INPUT_BATCH_TOPK_REQS,
        INPUT_BATCH_TOPK_REQS_PATCHED,
        "gpu input batch l20 buffers",
    )
    source = replace_once(
        source,
        INPUT_BATCH_TOPK_CPU,
        INPUT_BATCH_TOPK_CPU_PATCHED,
        "gpu input batch l20 seed init",
    )
    source = replace_once(
        source,
        INPUT_BATCH_COPY_TOPK,
        INPUT_BATCH_COPY_TOPK_PATCHED,
        "gpu input batch l20 copy",
    )
    return replace_once(
        source,
        INPUT_BATCH_METADATA_GENERATORS,
        INPUT_BATCH_METADATA_GENERATORS_PATCHED,
        "gpu input batch metadata l20 state",
    )


def patch_worker_sampler(source: str) -> str:
    source = replace_once(
        source,
        WORKER_IMPORT_MARKER,
        WORKER_IMPORT_MARKER + "    maybe_l20_topk_topp_sample,\n",
        "worker sampler import",
    )
    return replace_once(
        source,
        WORKER_PATCH_POINT,
        WORKER_PATCHED,
        "worker native sampler hook",
    )


def _install_target(path: Path, patcher) -> bool:
    if not path.exists():
        return False
    backup = path.with_suffix(".py.l20-topk-topp-backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(patcher(path.read_text(encoding="utf-8")), encoding="utf-8")
    return True


def _restore_target(path: Path) -> bool:
    backup = path.with_suffix(".py.l20-topk-topp-backup")
    if not backup.exists():
        return False
    shutil.copy2(backup, path)
    return True


def install(package: Path) -> None:
    helper = package / "v1" / "sample" / "ops" / "l20_topk_topp_sampling.py"
    helper.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("l20_topk_topp_sampling.py"), helper)
    patched = [
        _install_target(
            package / "v1" / "sample" / "metadata.py",
            patch_sampling_metadata,
        ),
        _install_target(
            package / "v1" / "sample" / "sampler.py",
            patch_active_sampler,
        ),
        _install_target(
            package / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
            patch_topk_topp_sampler,
        ),
        _install_target(
            package / "v1" / "worker" / "gpu_input_batch.py",
            patch_gpu_input_batch,
        ),
        _install_target(
            package / "v1" / "worker" / "gpu_model_runner.py",
            patch_gpu_model_runner,
        ),
        _install_target(
            package / "v1" / "worker" / "gpu" / "sample" / "sampler.py",
            patch_worker_sampler,
        ),
    ]
    if not any(patched):
        raise RuntimeError(f"missing supported vLLM sampler under: {package}")


def uninstall(package: Path) -> None:
    paths = [
        package / "v1" / "sample" / "metadata.py",
        package / "v1" / "sample" / "sampler.py",
        package / "v1" / "sample" / "ops" / "topk_topp_sampler.py",
        package / "v1" / "worker" / "gpu_input_batch.py",
        package / "v1" / "worker" / "gpu_model_runner.py",
        package / "v1" / "worker" / "gpu" / "sample" / "sampler.py",
    ]
    for path in paths:
        _restore_target(path)
    (package / "v1" / "sample" / "ops" / "l20_topk_topp_sampling.py").unlink(
        missing_ok=True
    )


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
