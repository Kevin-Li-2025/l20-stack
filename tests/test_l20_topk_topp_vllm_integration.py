import importlib.util
from pathlib import Path


def load_installer():
    path = Path("integrations/vllm/install_l20_topk_topp_sampler.py")
    spec = importlib.util.spec_from_file_location("install_l20_topk_topp_sampler", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_l20_topk_topp_installer_patches_vllm_sampler_points():
    module = load_installer()
    topk_source = """
from vllm.triton_utils import HAS_TRITON
def flashinfer_sample(logits, k, p, generators={}):
    assert not (k is None and p is None)
    if k is None:
        return None
class TopKTopPSampler:
    def forward_cuda(
        self,
        logits: torch.Tensor,
        generators: dict[int, torch.Generator],
        k: torch.Tensor | None,
        p: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return flashinfer_sample(logits.contiguous(), k, p, generators), None
"""
    metadata_source = """
class SamplingMetadata:
    generators: dict[int, torch.Generator]

    # None means no logprobs, 0 means sampled token logprobs only
"""
    active_sampler_source = """
        random_sampled, processed_logprobs = self.topk_topp_sampler(
            logits,
            sampling_metadata.generators,
            sampling_metadata.top_k,
            sampling_metadata.top_p,
        )
"""
    gpu_input_batch_source = """
        self.top_k_reqs: set[str] = set()

        # Frequency penalty related data structures
            self.top_k_cpu[req_index] = top_k
            self.frequency_penalties_cpu[req_index] = sampling_params.frequency_penalty
        if not self.no_top_k:
            copy_slice(self.top_k_cpu_tensor, self.top_k, num_reqs)

        if not self.no_penalties:
            pass
            generators=self.generators,
            max_num_logprobs=self.max_num_logprobs,
"""
    gpu_model_runner_source = """
            top_k=dummy_tensors(logits.size(1) - 1),
            generators={},
            max_num_logprobs=None,
"""
    worker_source = """
from vllm.v1.sample.ops.topk_topp_sampler import (
    apply_top_k_top_p,
    flashinfer_sample,
    flashinfer_sampler_supported,
)
def sample():
        if use_flashinfer:
            sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)
        else:
            processed_logits = apply_top_k_top_p(processed_logits, top_k, top_p)
"""

    patched_topk = module.patch_topk_topp_sampler(topk_source)
    patched_metadata = module.patch_sampling_metadata(metadata_source)
    patched_active_sampler = module.patch_active_sampler(active_sampler_source)
    patched_gpu_input_batch = module.patch_gpu_input_batch(gpu_input_batch_source)
    patched_gpu_model_runner = module.patch_gpu_model_runner(gpu_model_runner_source)
    patched_worker = module.patch_worker_sampler(worker_source)

    assert "maybe_l20_topk_topp_sample" in patched_topk
    assert "l20_expanded_idx_mapping: torch.Tensor | None = None" in patched_topk
    assert "l20_seeds: torch.Tensor | None = None" in patched_topk
    assert "l20_positions: torch.Tensor | None = None" in patched_topk
    assert "expanded_idx_mapping=l20_expanded_idx_mapping" in patched_topk
    assert "return l20_sampled, None" in patched_topk
    assert "l20_expanded_idx_mapping: torch.Tensor | None" in patched_metadata
    assert "l20_seeds: torch.Tensor | None" in patched_metadata
    assert "l20_positions: torch.Tensor | None" in patched_metadata
    assert "sampling_metadata.l20_expanded_idx_mapping" in patched_active_sampler
    assert "sampling_metadata.l20_seeds" in patched_active_sampler
    assert "sampling_metadata.l20_positions" in patched_active_sampler
    assert "self.l20_sampler_seeds" in patched_gpu_input_batch
    assert "self.l20_sampler_positions" in patched_gpu_input_batch
    assert "self.l20_sampler_indices[:num_reqs]" in patched_gpu_input_batch
    assert "l20_expanded_idx_mapping=torch.arange" in patched_gpu_model_runner
    assert "maybe_l20_topk_topp_sample" in patched_worker
    assert "expanded_idx_mapping=expanded_idx_mapping" in patched_worker
    assert "seeds=self.sampling_states.seeds.gpu" in patched_worker
    assert "positions=pos" in patched_worker
    assert "top_k_value=int(l20_top_k_values[0])" in patched_worker
    assert "sampled = l20_sampled.to(torch.int64)" in patched_worker
    assert "sampled = flashinfer_sample(processed_logits, top_k, top_p).to(torch.int64)" in patched_worker


def test_l20_topk_topp_helper_uses_vllm_rng_state():
    source = Path("integrations/vllm/l20_topk_topp_sampling.py").read_text()

    assert "VLLM_L20_TOPK_TOPP_SAMPLER" in source
    assert "VLLM_L20_TOPK_TOPP_SAMPLER_TRACE" in source
    assert "should_prefer_l20_topk_topp_sampling" in source
    assert "topk_topp_sample_with_vllm_rng_out" in source
    assert "per_request_generators" in source
    assert "missing_vllm_rng_state" in source
    assert "torch.rand" not in source
