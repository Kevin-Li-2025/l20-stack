from pathlib import Path


def test_sampling_campaign_switches_flashinfer_sampler():
    source = Path("scripts/run_vllm_l20_sampling_campaign.sh").read_text()

    assert "SAMPLER_MODE: flashinfer|torch" in source
    assert "VLLM_USE_FLASHINFER_SAMPLER=\"$use_flashinfer_sampler\"" in source
    assert "export CUDA_HOME=" in source
    assert "export CUDACXX=" in source
    assert "scripts/prewarm_flashinfer_sampling.py" in source
    assert "--temperature \"$temperature\"" in source
    assert "--top-p \"$top_p\"" in source
    assert "--top-k \"$top_k\"" in source
    assert "--percentile-metrics ttft,tpot,itl,e2el" in source
    assert "scripts/inspect_vllm_sampling_path.py" in source
    assert "PYTHONPATH=\"$extra_vllm_pythonpath" in source


def test_sampling_path_inspector_reports_cpu_fallback_evidence():
    source = Path("scripts/inspect_vllm_sampling_path.py").read_text()

    assert '"flashinfer"' in source
    assert '"fallback"' in source
    assert '"cpu"' in source
    assert "cpu_fallback_suspected" in source
