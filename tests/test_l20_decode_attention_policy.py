from l20_stack.ops.triton_decode_attention import (
    gqa_decode_attention_fp8_split_kv,
    should_use_l20_gqa_decode_attention,
    should_use_l20_split_kv_attention,
)


def test_single_batch_gate_rejects_long_context():
    assert should_use_l20_gqa_decode_attention(1, 1024)
    assert not should_use_l20_gqa_decode_attention(1, 2048)
    assert not should_use_l20_gqa_decode_attention(1, 4096)


def test_multi_batch_gate_accepts_measured_regime():
    assert not should_use_l20_gqa_decode_attention(2, 4096)
    assert should_use_l20_gqa_decode_attention(4, 128)
    assert should_use_l20_gqa_decode_attention(4, 4096)


def test_split_kv_gate_targets_long_context():
    assert not should_use_l20_split_kv_attention(1024)
    assert should_use_l20_split_kv_attention(2048)
    assert should_use_l20_split_kv_attention(4096)


def test_fp8_decode_attention_entrypoint_exists():
    assert callable(gqa_decode_attention_fp8_split_kv)
