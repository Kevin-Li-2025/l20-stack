import json
from pathlib import Path

from scripts.plan_sampler_semantics_targets import build_plan, render_markdown


def test_build_plan_recommends_largest_p0_semantics_boundary():
    summary = {
        "model": "Qwen/Qwen2.5-0.5B-Instruct",
        "cases": [
            {"case": "greedy_no_penalty", "itl_ms": {"median": 6.72}},
            {"case": "greedy_default_repetition", "itl_ms": {"median": 9.22}},
            {"case": "sample_topk_topp", "itl_ms": {"median": 9.54}},
            {"case": "sample_topk_topp_penalty", "itl_ms": {"median": 9.56}},
            {"case": "greedy_token_logprobs", "itl_ms": {"median": 9.33}},
        ],
    }

    plan = build_plan(summary)

    assert plan["decision"] == "target_sampling_semantics_boundary"
    assert plan["recommended_next_target"]["case"] == "sample_topk_topp_penalty"
    assert plan["recommended_next_target"]["plan"]["target"] == "fused_topk_topp+penalty"
    assert plan["recommended_next_target"]["plan"]["priority"] == "p0"


def test_render_markdown_mentions_recommendation():
    plan = {
        "rows": [
            {
                "case": "sample_topk_topp",
                "median_itl_ms": 9.54,
                "observed_itl_delta_vs_greedy_pct": 42.0,
                "plan": {"target": "fused_topk_topp", "priority": "p0"},
            }
        ],
        "recommended_next_target": {
            "plan": {"target": "fused_topk_topp", "priority": "p0"}
        },
    }

    markdown = render_markdown(plan)

    assert "fused_topk_topp" in markdown
    assert "largest observed ITL delta" in markdown


def test_plan_script_accepts_real_artifact_shape(tmp_path):
    source = Path(
        "benchmarks/results/a100-vllm-sampling-semantics-qwen25-05b/"
        "sampling_semantics_summary.json"
    )
    summary = json.loads(source.read_text(encoding="utf-8"))

    plan = build_plan(summary)

    assert plan["recommended_next_target"]["plan"]["priority"] == "p0"
    assert plan["baseline_median_itl_ms"] > 0.0
