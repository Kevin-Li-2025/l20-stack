import builtins
import importlib
import sys

import pytest


def _flash_sampling():
    return importlib.import_module("l20_stack.epilogue.flash_sampling")


def _request(**overrides):
    module = _flash_sampling()
    values = {
        "batch_size": 1,
        "vocab_size": 151_936,
        "hidden_size": 1536,
    }
    values.update(overrides)
    return module.FlashSamplingRequest(**values)


def test_flash_sampling_import_does_not_require_torch_or_triton(monkeypatch):
    module_name = "l20_stack.epilogue.flash_sampling"
    sys.modules.pop(module_name, None)
    real_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".", 1)[0] in {"torch", "triton"}:
            raise AssertionError(f"unexpected optional dependency import: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    module = importlib.import_module(module_name)
    request = module.FlashSamplingRequest(batch_size=1, vocab_size=151_936, hidden_size=1536)

    assert module.should_use_flash_sampling_epilogue(request)


def test_flash_sampling_gate_accepts_greedy_and_gumbel_decode_shapes():
    module = _flash_sampling()

    greedy = module.plan_flash_sampling_epilogue(_request(top_k=-1, top_p=1.0, num_logprobs=-1))
    gumbel = module.plan_flash_sampling_epilogue(_request(batch_size=4, sampling_mode="gumbel"))

    assert greedy.eligible
    assert greedy.reasons == ()
    assert gumbel.eligible
    assert gumbel.policy.block_vocab == 64
    assert gumbel.policy.block_hidden == 128


@pytest.mark.parametrize(
    ("reason_name", "overrides"),
    [
        ("REASON_NOT_DECODE_ONLY", {"decode_only": False}),
        ("REASON_BATCH_GT_4", {"batch_size": 5}),
        ("REASON_VOCAB_GT_262144", {"vocab_size": 262_145}),
        ("REASON_HIDDEN_NOT_DIVISIBLE_BY_64", {"hidden_size": 1537}),
        ("REASON_SAMPLING_MODE_UNSUPPORTED", {"sampling_mode": "topk_topp"}),
        ("REASON_LOGPROBS_UNSUPPORTED", {"num_logprobs": 1}),
        ("REASON_PENALTIES_UNSUPPORTED", {"has_penalties": True}),
        ("REASON_BAD_WORDS_UNSUPPORTED", {"has_bad_words": True}),
        ("REASON_STRUCTURED_OUTPUT_UNSUPPORTED", {"has_structured_output": True}),
        ("REASON_SPEC_DECODE_UNSUPPORTED", {"speculative_decode": True}),
        ("REASON_TOP_K_TOP_P_UNSUPPORTED", {"top_k": 50}),
        ("REASON_TOP_K_TOP_P_UNSUPPORTED", {"top_p": 0.9}),
    ],
)
def test_flash_sampling_gate_reports_each_fallback_reason(reason_name, overrides):
    module = _flash_sampling()
    decision = module.plan_flash_sampling_epilogue(_request(**overrides))
    reason = getattr(module, reason_name)

    assert not decision.eligible
    assert decision.policy is None
    assert reason in decision.reasons


def test_flash_sampling_policy_keeps_batch_one_measured_lm_head_top1_shape():
    module = _flash_sampling()
    policy = module.flash_sampling_launch_policy(_request(batch_size=1))

    assert policy.block_vocab == 32
    assert policy.block_hidden == 64
    assert policy.blocks_per_row == 4748
    assert policy.reduce_block == 8192
    assert policy.num_warps == 4
    assert policy.num_stages == 3
    assert policy.strategy == "two_stage_lm_head_flash_sampling_epilogue_plan"


def test_flash_sampling_policy_uses_wider_tiles_for_batched_decode():
    module = _flash_sampling()
    policy = module.flash_sampling_launch_policy(_request(batch_size=4))

    assert policy.block_vocab == 64
    assert policy.block_hidden == 128
    assert policy.blocks_per_row == 2374
    assert policy.reduce_block == 4096
    assert policy.num_warps == 8


def test_flash_sampling_launch_policy_rejects_requests_outside_gate():
    module = _flash_sampling()

    with pytest.raises(ValueError, match="top_k_top_p_unsupported"):
        module.flash_sampling_launch_policy(_request(top_k=50))
