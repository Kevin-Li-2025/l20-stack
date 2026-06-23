from pathlib import Path


def test_tree_attention_installer_copies_op_into_vllm_namespace():
    source = Path("integrations/vllm/install_l20_tree_attention.py").read_text()
    assert 'op_dir / "l20_tree_attention.py"' in source
    assert 'op_dir / "l20_tree_attention_dispatch.py"' in source
    assert 'find_spec("vllm.v1.attention.backends.flashinfer")' in source
    assert "VLLM_SOURCE_TREE" in source
    assert "src/l20_stack/ops/triton_tree_attention.py" in source
    assert "integrations/vllm/l20_tree_attention_dispatch.py" in source
    assert "maybe_run_l20_tree_attention" in source
    assert "maybe_run_l20_causal_verifier_from_prefill" in source
    assert "maybe_l20_causal_verifier_attention" in source
    assert "maybe_l20_tree_attention" in source
    assert "maybe_run_l20_tree_attention_from_prefill" in source
    assert "block_tables=block_table_tensor[prefill_start:]" in source
    assert "max_seq_len=max_seq_len" in source
    assert "attn_metadata.prefill.max_seq_len" in source
    assert "not attn_metadata.causal" in source
    assert "if not l20_tree_ran" in source
    assert "VLLM_L20_TREE_ATTENTION_TRACE" in source
    assert "VLLM_L20_TREE_ATTENTION_TIMING" in source
    assert "_l20_tree_cuda_event_ms" in source
    assert "native_prefill_site" in source
    assert "native_prefill_timing" in source
    assert "prefill_hook_run" in source
    assert "prefill_hook_timing" in source
    assert "prefill_hook_skip" in source
    assert "causal_verifier_run" in source
    assert "causal_verifier_timing" in source
    assert "causal_verifier_skip" in source
    assert "--uninstall" in source


def test_vllm_tree_attention_smoke_uses_vllm_import_path():
    source = Path("scripts/smoke_vllm_l20_tree_attention.py").read_text()
    assert "from vllm.v1.attention.ops.l20_tree_attention import" in source
    assert "from vllm.v1.attention.ops.l20_tree_attention_dispatch import" in source
    assert "maybe_l20_tree_attention" in source
    assert "VLLM_ENABLE_L20_TREE_ATTENTION" in source
    assert "torch_tree_attention_reference" in source
    assert "l20_stack.ops" not in source


def test_vllm_prefill_hook_smoke_calls_backend_hook():
    source = Path("scripts/smoke_vllm_l20_tree_prefill_hook.py").read_text()
    assert "from vllm.v1.attention.backends.flashinfer import" in source
    assert "maybe_run_l20_tree_attention_from_prefill" in source
    assert "maybe_run_l20_causal_verifier_from_prefill" in source
    assert "--causal-verifier" in source
    assert "--dtype" in source
    assert "VLLM_ENABLE_L20_TREE_ATTENTION" in source
    assert "torch_tree_attention_reference" in source
    assert "l20_stack.ops" not in source


def test_tree_attention_dispatch_is_env_gated_and_l20_only():
    source = Path("integrations/vllm/l20_tree_attention_dispatch.py").read_text()
    assert "VLLM_ENABLE_L20_TREE_ATTENTION" in source
    assert "torch.cuda.get_device_capability() != (8, 9)" in source
    assert "torch.cuda.is_current_stream_capturing()" in source
    assert "cached_length >= min_cached_length" in source
    assert "torch.bfloat16" in source
    assert "causal_verifier_attention_paged" in source
    assert "should_dispatch_l20_causal_verifier_attention" in source
    assert "min_cached_length=1024" in source
