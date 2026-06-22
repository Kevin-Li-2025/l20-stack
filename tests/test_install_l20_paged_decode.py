from pathlib import Path


def test_installer_has_conservative_service_gate():
    source = Path("integrations/vllm/install_l20_paged_decode.py").read_text()
    assert "l20_batch == 1 and l20_max_seq <= 2304" in source
    assert "l20_batch <= 4 and l20_max_seq <= 640" in source
    assert "paged_decode_split_out" in source
    assert "decode_wrapper.run" in source
