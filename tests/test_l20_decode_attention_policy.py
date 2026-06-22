from l20_stack.ops.triton_decode_attention import (
    should_use_l20_gqa_decode_attention,
)


def test_single_batch_gate_rejects_long_context():
    assert should_use_l20_gqa_decode_attention(1, 512)
    assert not should_use_l20_gqa_decode_attention(1, 2048)
    assert not should_use_l20_gqa_decode_attention(1, 4096)


def test_multi_batch_gate_accepts_measured_regime():
    assert should_use_l20_gqa_decode_attention(8, 128)
    assert should_use_l20_gqa_decode_attention(8, 4096)
