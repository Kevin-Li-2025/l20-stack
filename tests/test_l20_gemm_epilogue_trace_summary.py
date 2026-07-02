import importlib.util
from pathlib import Path


def load_summarizer():
    path = Path("scripts/summarize_l20_gemm_epilogue_trace.py")
    spec = importlib.util.spec_from_file_location("summarize_l20_gemm_epilogue_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gemm_epilogue_semantic_trace_summary_splits_prefill_from_decode():
    module = load_summarizer()
    summary = module.summarize_trace(
        Path(
            "benchmarks/results/a100-vllm-gemm-epilogue-semantic-trace/"
            "qwen25-05b-topk-topp-penalty-r8/gemm_epilogue_trace.jsonl"
        )
    )

    assert summary["events"] == 320
    assert summary["event_eligible"] == 310
    assert summary["semantic_candidate_eligible"] == 320
    assert summary["decode_semantic_candidate_eligible"] == 310
    assert summary["targets"] == {
        "fused_topk_topp_sparse_penalty_lm_head_epilogue": 320
    }
    assert summary["history_sources"] == {"input_batch_token_ids_cpu": 320}
    assert summary["event_reasons"] == {
        "not_single_token_decode": 10,
        "scheduled_tokens_mismatch": 10,
    }
    assert round(summary["decode_estimated_logits_mib_fp32"], 2) == 179.67
