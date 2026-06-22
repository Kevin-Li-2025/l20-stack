from pathlib import Path


def test_gqa_tensor_core_benchmark_covers_policy_matrix():
    source = Path("scripts/benchmark_flashinfer_gqa_tensor_cores.py").read_text()
    assert "for batch in (1, 4, 8, 16)" in source
    assert "for ratio in (1, 2, 4, 8)" in source
    assert "use_tensor_cores=tensor_cores" in source
