from pathlib import Path


def test_fp8_kv_decode_benchmark_uses_real_torch_fp8():
    source = Path("scripts/benchmark_fp8_kv_decode_attention.py").read_text()

    assert "torch.float8_e4m3fn" in source
    assert "gqa_decode_attention_fp8_split_kv" in source
    assert "fp8_predequantized_attention" in source
    assert "fp8_materialize_dequant_then_attention" in source
    assert "fp8_fused_dequant_attention" in source
    assert "fused_fp8_vs_materialized_fp8" in source
