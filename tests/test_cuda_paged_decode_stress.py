from pathlib import Path


def test_stress_script_covers_graph_and_randomized_pages():
    source = Path("scripts/stress_cuda_paged_decode.py").read_text()
    assert "torch.cuda.CUDAGraph" in source
    assert "torch.randperm" in source
    assert "graph_replays" in source
    assert "torch.allclose" in source
