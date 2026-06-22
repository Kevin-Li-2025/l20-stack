from pathlib import Path


def test_cuda_prototype_is_l20_specialized_and_checked():
    source = Path("integrations/vllm/cuda/l20_paged_decode.cu").read_text()
    benchmark = Path("scripts/benchmark_cuda_paged_decode.py").read_text()
    assert "threadIdx.x" in source
    assert "C10_CUDA_KERNEL_LAUNCH_CHECK" in source
    assert "code=sm_89" in benchmark
    assert "torch.allclose" in benchmark
    assert "paged_decode_partial_kernel" in source
    assert "paged_decode_merge_kernel" in source
    assert "split_size must be a multiple of 16 from 64 through 1024" in source
    smoke = Path("scripts/smoke_cuda_paged_decode_op.py").read_text()
    assert "torch.ops.l20_stack.paged_decode_split_out" in smoke
    assert "torch.testing.assert_close" in smoke
