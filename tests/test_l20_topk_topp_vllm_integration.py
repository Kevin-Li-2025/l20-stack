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
    patched_worker = module.patch_worker_sampler(worker_source)

    assert "maybe_l20_topk_topp_sample" in patched_topk
    assert "l20_sampled = maybe_l20_topk_topp_sample(logits, k, p, generators)" in patched_topk
    assert "return l20_sampled" in patched_topk
    assert "maybe_l20_topk_topp_sample" in patched_worker
    assert "return l20_sampled, processed_logits" in patched_worker


def test_l20_topk_topp_helper_keeps_documented_env_gate():
    source = Path("integrations/vllm/l20_topk_topp_sampling.py").read_text()

    assert "VLLM_L20_TOPK_TOPP_SAMPLER" in source
    assert "VLLM_L20_TOPK_TOPP_SAMPLER_TRACE" in source
    assert "should_prefer_l20_topk_topp_sampling" in source
    assert "topk_topp_sample_from_uniform_out" in source
    assert "per_request_generators" in source
