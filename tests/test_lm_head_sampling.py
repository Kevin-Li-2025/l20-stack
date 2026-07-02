from pathlib import Path

import pytest

from l20_stack.ops.triton_lm_head_sampling import (
    lm_head_sample,
    lm_head_sampling_launch_config,
    should_use_l20_lm_head_sparse_penalty_sampling,
    should_use_l20_lm_head_sampling,
)


def test_lm_head_sampling_policy_uses_tensor_core_compatible_batch_tile():
    batch1 = lm_head_sampling_launch_config(1, 151_936, 1536)
    batch4 = lm_head_sampling_launch_config(4, 151_936, 1536)

    assert (batch1.block_batch, batch1.block_vocab, batch1.block_hidden) == (16, 32, 256)
    assert (batch4.block_batch, batch4.block_vocab, batch4.block_hidden) == (16, 64, 256)
    assert batch4.blocks_per_row == 2374
    assert batch4.reduce_block == 4096
    assert batch4.num_warps == 8
    assert batch4.strategy == "two_stage_lm_head_gumbel_max"


def test_lm_head_sampling_explicit_blocks_override_policy():
    config = lm_head_sampling_launch_config(
        4,
        151_936,
        1536,
        block_vocab=16,
        block_hidden=64,
    )
    assert (config.block_vocab, config.block_hidden) == (16, 64)


def test_lm_head_sampling_rejects_bad_launch_shapes():
    with pytest.raises(ValueError):
        lm_head_sampling_launch_config(0, 151_936, 1536)
    with pytest.raises(ValueError):
        lm_head_sampling_launch_config(4, 151_936, 1537)
    with pytest.raises(ValueError):
        lm_head_sampling_launch_config(4, 151_936, 1536, block_hidden=96)


def test_lm_head_sampling_gate_is_intentionally_narrow():
    assert should_use_l20_lm_head_sampling(1, 151_936, 1536)
    assert should_use_l20_lm_head_sampling(4, 151_936, 1536)
    assert should_use_l20_lm_head_sampling(4, 151_936, 1536, top_k=1, top_p=1.0)

    assert not should_use_l20_lm_head_sampling(5, 151_936, 1536)
    assert not should_use_l20_lm_head_sampling(4, 300_000, 1536)
    assert not should_use_l20_lm_head_sampling(4, 151_936, 1537)
    assert not should_use_l20_lm_head_sampling(4, 151_936, 1536, top_k=50)
    assert not should_use_l20_lm_head_sampling(4, 151_936, 1536, top_p=0.9)


def test_lm_head_sparse_penalty_gate_reuses_lm_head_policy():
    assert should_use_l20_lm_head_sparse_penalty_sampling(1, 151_936, 1536, 128)
    assert should_use_l20_lm_head_sparse_penalty_sampling(4, 151_936, 1536, 256)

    assert not should_use_l20_lm_head_sparse_penalty_sampling(5, 151_936, 1536, 128)
    assert not should_use_l20_lm_head_sparse_penalty_sampling(1, 151_936, 1536, 0)
    assert not should_use_l20_lm_head_sparse_penalty_sampling(1, 151_936, 1536, 512)
    assert not should_use_l20_lm_head_sparse_penalty_sampling(
        1, 151_936, 1536, 128, top_k=50
    )


def test_lm_head_sampling_source_has_no_full_logits_materialization():
    import l20_stack.ops.triton_lm_head_sampling as module

    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "_lm_head_sampling_partial_kernel" in source
    assert "_lm_head_sampling_reduce_kernel" in source
    assert "USE_GUMBEL" in source
    assert "HAS_SPARSE_PENALTIES" in source
    assert "history_tokens + batch_offsets * MAX_HISTORY + hist_idx" in source
    assert "frequency_penalties + batch_offsets" in source
    assert "tl.log(-tl.log(uniform))" in source
    assert "tl.dot(w, h" in source
    assert "hidden @ weight.T" not in source
    assert "adjusted_logits" not in source


def test_lm_head_sampling_import_is_cpu_safe():
    import l20_stack.ops.triton_lm_head_sampling as module

    assert module.lm_head_sampling_launch_config(1, 8192, 512).blocks_per_row == 256
    assert module.should_use_l20_lm_head_sampling(1, 8192, 512)


def test_lm_head_sparse_penalty_cuda_matches_reference():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")

    from l20_stack.ops.triton_sampling import apply_sparse_token_penalties_reference

    torch.manual_seed(2026)
    hidden = torch.randn((2, 64), device="cuda", dtype=torch.float16)
    weight = torch.randn((128, 64), device="cuda", dtype=torch.float16)
    history_tokens = torch.tensor(
        [[0, 0, 3, 9, 200], [5, 5, 5, 7, -1]],
        device="cuda",
        dtype=torch.int64,
    )
    history_lengths = torch.tensor([4, 4], device="cuda", dtype=torch.int64)
    frequency = torch.tensor([0.2, 0.1], device="cuda", dtype=torch.float32)
    presence = torch.tensor([0.3, 0.2], device="cuda", dtype=torch.float32)
    repetition = torch.tensor([1.5, 1.25], device="cuda", dtype=torch.float32)

    values, tokens = lm_head_sample(
        hidden,
        weight,
        history_tokens=history_tokens,
        history_lengths=history_lengths,
        frequency_penalties=frequency,
        presence_penalties=presence,
        repetition_penalties=repetition,
        use_gumbel=False,
        block_vocab=16,
        block_hidden=64,
    )
    torch.cuda.synchronize()

    logits = hidden.float() @ weight.float().T
    adjusted = apply_sparse_token_penalties_reference(
        logits,
        history_tokens,
        history_lengths,
        frequency_penalties=frequency,
        presence_penalties=presence,
        repetition_penalties=repetition,
    )
    expected_values, expected_tokens = adjusted.max(dim=-1)

    assert torch.equal(tokens.cpu(), expected_tokens.cpu())
    assert torch.allclose(values.cpu(), expected_values.cpu(), atol=6e-2, rtol=1e-3)
