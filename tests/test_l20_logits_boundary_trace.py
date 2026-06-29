import importlib.util
import json
from pathlib import Path

import numpy as np


def load_module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Device:
    type = "cuda"


class Tensor:
    def __init__(self, shape, dtype="torch.float16"):
        self.shape = shape
        self.dtype = dtype
        self.device = Device()


class NpOwner:
    def __init__(self, values):
        self.np = np.array(values)


class ParallelConfig:
    tensor_parallel_size = 1


class Sampler:
    def __init__(self):
        self.sampling_states = type(
            "SamplingStates",
            (),
            {
                "temperature": NpOwner([0.8, 1.0]),
                "top_k": NpOwner([50, 50]),
                "top_p": NpOwner([0.9, 1.0]),
                "num_logprobs": np.array([-1, -1], dtype=np.int32),
                "min_p": NpOwner([0.0, 0.0]),
            },
        )()
        self.logprob_token_ids_state = type(
            "LogprobTokenIdsState",
            (),
            {"num_token_ids": NpOwner([0, 0])},
        )()
        self.penalties_state = type(
            "PenaltiesState",
            (),
            {"use_penalty": np.array([False, False])},
        )()
        self.logit_bias_state = type(
            "LogitBiasState",
            (),
            {"use_logit_bias": np.array([False, False])},
        )()
        self.bad_words_state = type(
            "BadWordsState",
            (),
            {"num_bad_words": NpOwner([0, 0])},
        )()


class ModelRunner:
    parallel_config = ParallelConfig()

    def __init__(self):
        self.sampler = Sampler()


class InputBatch:
    num_reqs = 2
    num_tokens = 2
    num_draft_tokens = 0
    has_structured_output_reqs = False
    idx_mapping_np = np.array([0, 1], dtype=np.int32)
    is_prefilling_np = np.array([False, False])


class SchedulerOutput:
    num_scheduled_tokens = {"req0": 1, "req1": 1}
    total_num_scheduled_tokens = 2


class V2InputBatch:
    num_reqs = 2
    num_draft_tokens = 0
    has_structured_output_reqs = False
    temperature_cpu = np.array([0.8, 1.0, 999.0], dtype=np.float32)
    top_k_cpu = np.array([50, 50, 999], dtype=np.int32)
    top_p_cpu = np.array([0.9, 1.0, 999.0], dtype=np.float32)
    frequency_penalties_cpu = np.array([0.0, 0.0, 999.0], dtype=np.float32)
    presence_penalties_cpu = np.array([0.0, 0.0, 999.0], dtype=np.float32)
    repetition_penalties_cpu = np.array([1.0, 1.0, 999.0], dtype=np.float32)
    logits_processing_needs_token_ids = np.array([False, False, True])
    num_logprobs = {}
    logprob_token_ids = {}
    has_allowed_token_ids = set()
    bad_words_token_ids = {}
    generators = {}


def test_l20_logits_boundary_trace_records_eligible_event(tmp_path, monkeypatch):
    module = load_module(
        "integrations/vllm/l20_logits_boundary_trace.py",
        "l20_logits_boundary_trace",
    )
    module._TRACE_COUNT = 0
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    module.maybe_trace_l20_logits_boundary(
        ModelRunner(),
        InputBatch(),
        None,
        Tensor((2, 2048)),
        Tensor((2, 151936)),
    )

    event = json.loads(trace.read_text(encoding="utf-8"))
    assert event["event"] == "l20_logits_boundary_gate"
    assert event["eligible"] is True
    assert event["reasons"] == []
    assert event["metadata"]["num_reqs"] == 2
    assert event["metadata"]["logits_shape"] == [2, 151936]
    assert event["metadata"]["logits_dtype"] == "torch.float16"
    assert event["metadata"]["logits_element_bytes"] == 2
    assert event["metadata"]["logits_bytes"] == 2 * 151936 * 2
    assert event["metadata"]["hidden_dim"] == 2048
    assert event["metadata"]["vocab_size"] == 151936
    shadow = event["metadata"]["shadow_epilogue"]
    assert shadow["mode"] == "shadow_trace_only"
    assert shadow["would_use_epilogue"] is True
    assert shadow["mutates_outputs"] is False
    assert shadow["avoidable_logits_materialization_bytes"] == 2 * 151936 * 2
    assert event["metadata"]["sampling"]["temperature_min"] == 0.8
    assert event["metadata"]["sampling"]["top_k_max"] == 50.0
    assert event["metadata"]["sampling"]["top_p_min"] == 0.9


def test_l20_logits_boundary_trace_supports_v2_input_batch(tmp_path, monkeypatch):
    module = load_module(
        "integrations/vllm/l20_logits_boundary_trace.py",
        "l20_logits_boundary_trace_v2",
    )
    module._TRACE_COUNT = 0
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    module.maybe_trace_l20_logits_boundary(
        type("V2ModelRunner", (), {"parallel_config": ParallelConfig()})(),
        V2InputBatch(),
        None,
        Tensor((2, 2048)),
        Tensor((2, 151936)),
        SchedulerOutput(),
    )

    event = json.loads(trace.read_text(encoding="utf-8"))
    assert event["eligible"] is True
    assert event["reasons"] == []
    assert event["metadata"]["scheduler_max_scheduled_tokens"] == 1
    assert event["metadata"]["sampling"]["temperature_max"] == 1.0
    assert event["metadata"]["sampling"]["top_k_max"] == 50.0
    assert event["metadata"]["sampling"]["top_p_max"] == 1.0


def test_l20_logits_boundary_trace_rejects_v2_prefill_and_processors(
    tmp_path,
    monkeypatch,
):
    module = load_module(
        "integrations/vllm/l20_logits_boundary_trace.py",
        "l20_logits_boundary_trace_v2_reject",
    )
    module._TRACE_COUNT = 0
    batch = V2InputBatch()
    batch.num_logprobs = {"req0": 1}
    batch.frequency_penalties_cpu = np.array([0.1, 0.0], dtype=np.float32)
    trace = tmp_path / "trace.jsonl"
    scheduler_output = type(
        "PrefillSchedulerOutput",
        (),
        {
            "num_scheduled_tokens": {"req0": 128, "req1": 1},
            "total_num_scheduled_tokens": 129,
        },
    )()
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    module.maybe_trace_l20_logits_boundary(
        type("V2ModelRunner", (), {"parallel_config": ParallelConfig()})(),
        batch,
        None,
        Tensor((2, 2048)),
        Tensor((2, 151936)),
        scheduler_output,
    )

    event = json.loads(trace.read_text(encoding="utf-8"))
    assert event["eligible"] is False
    assert "prefill" in event["reasons"]
    assert "not_single_token_decode" in event["reasons"]
    assert "token_logprobs" in event["reasons"]
    assert "penalties" in event["reasons"]


def test_l20_logits_boundary_trace_records_reject_reasons(tmp_path, monkeypatch):
    module = load_module(
        "integrations/vllm/l20_logits_boundary_trace.py",
        "l20_logits_boundary_trace_reject",
    )
    module._TRACE_COUNT = 0
    batch = InputBatch()
    batch.num_tokens = 4
    batch.num_draft_tokens = 1
    batch.is_prefilling_np = np.array([False, True])
    runner = ModelRunner()
    runner.sampler.sampling_states.num_logprobs = np.array([-1, 5], dtype=np.int32)
    runner.sampler.sampling_states.min_p = NpOwner([0.0, 0.25])
    trace = tmp_path / "trace.jsonl"
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    module.maybe_trace_l20_logits_boundary(
        runner,
        batch,
        object(),
        Tensor((2, 2048)),
        Tensor((4, 151936)),
    )

    event = json.loads(trace.read_text(encoding="utf-8"))
    assert event["eligible"] is False
    assert "grammar_or_structured_output" in event["reasons"]
    assert "spec_decode" in event["reasons"]
    assert "prefill" in event["reasons"]
    assert "not_single_token_decode" in event["reasons"]
    assert "logits_rows_not_num_reqs" in event["reasons"]
    assert "token_logprobs" in event["reasons"]
    assert "min_p" in event["reasons"]
    shadow = event["metadata"]["shadow_epilogue"]
    assert shadow["would_use_epilogue"] is False
    assert shadow["avoidable_logits_materialization_bytes"] == 0
    assert "min_p" in shadow["fallback_reasons"]


def test_install_l20_logits_boundary_trace_patches_and_uninstalls(tmp_path):
    installer = load_module(
        "integrations/vllm/install_l20_logits_boundary_trace.py",
        "install_l20_logits_boundary_trace",
    )
    package = tmp_path / "vllm"
    target = package / "v1/worker/gpu/model_runner.py"
    target.parent.mkdir(parents=True)
    v2_target = package / "v1/worker/gpu_model_runner.py"
    v2_target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        """from vllm.v1.worker.gpu.structured_outputs import StructuredOutputsWorker

class GPUModelRunner:
    def sample(self, hidden_states, input_batch, grammar_output):
        sample_hidden_states = hidden_states[input_batch.logits_indices]
        logits = self.model.compute_logits(sample_hidden_states)
        if grammar_output is not None:
            pass
        return self.sampler(logits, input_batch)
""",
        encoding="utf-8",
    )
    v2_target.write_text(
        """from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

class GPUModelRunner:
    def sample_tokens(self, grammar_output):
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            pass
        return self._sample(logits, spec_decode_metadata)
""",
        encoding="utf-8",
    )

    installer.install(package)
    patched = target.read_text(encoding="utf-8")
    v2_patched = v2_target.read_text(encoding="utf-8")
    assert "maybe_trace_l20_logits_boundary" in patched
    assert "maybe_trace_l20_logits_boundary" in v2_patched
    assert "scheduler_output" in v2_patched
    assert (package / "v1/worker/gpu/l20_logits_boundary_trace.py").exists()
    installer.install(package)
    assert target.read_text(encoding="utf-8") == patched
    assert v2_target.read_text(encoding="utf-8") == v2_patched

    installer.uninstall(package)
    restored = target.read_text(encoding="utf-8")
    v2_restored = v2_target.read_text(encoding="utf-8")
    assert "maybe_trace_l20_logits_boundary" not in restored
    assert "maybe_trace_l20_logits_boundary" not in v2_restored
    assert not (package / "v1/worker/gpu/l20_logits_boundary_trace.py").exists()


def test_l20_logits_boundary_trace_summarizer_counts_reasons_and_shapes(tmp_path):
    summarizer = load_module(
        "scripts/summarize_l20_logits_boundary_trace.py",
        "summarize_l20_logits_boundary_trace",
    )
    events = [
        {
            "eligible": True,
            "reasons": [],
            "metadata": {
                "logits_shape": [2, 10],
                "hidden_shape": [2, 4],
                "logits_dtype": "torch.float16",
                "hidden_dtype": "torch.float16",
            },
        },
        {
            "eligible": False,
            "reasons": ["prefill", "token_logprobs"],
            "metadata": {
                "logits_shape": [4, 10],
                "hidden_shape": [4, 4],
                "logits_dtype": "torch.float16",
                "hidden_dtype": "torch.float16",
            },
        },
        {
            "eligible": False,
            "reasons": ["prefill"],
            "metadata": {
                "logits_shape": [4, 10],
                "hidden_shape": [4, 4],
                "logits_dtype": "torch.float16",
                "hidden_dtype": "torch.float16",
            },
        },
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    summary = summarizer.summarize(summarizer.read_events(trace))
    assert summary["total_events"] == 3
    assert summary["eligible_events"] == 1
    assert summary["fallback_events"] == 2
    assert summary["reason_counts"] == {"prefill": 2, "token_logprobs": 1}
    assert summary["logits_shape_counts"] == {"4x10": 2, "2x10": 1}
    assert summary["schema_version"] == 3
    assert summary["eligible_logits_bytes"] == 2 * 10 * 2
    assert summary["total_logits_bytes"] == (2 * 10 + 4 * 10 + 4 * 10) * 2
    assert summary["logits_unknown_bytes_events"] == 0
    shape_budget = {row["shape"]: row for row in summary["shape_budget"]}
    assert shape_budget["2x10"]["eligible_events"] == 1
    assert shape_budget["2x10"]["eligible_logits_bytes"] == 2 * 10 * 2
    assert shape_budget["4x10"]["events"] == 2
    assert shape_budget["4x10"]["fallback_events"] == 2


def test_l20_logits_boundary_trace_summarizer_counts_shadow_epilogue(tmp_path):
    summarizer = load_module(
        "scripts/summarize_l20_logits_boundary_trace.py",
        "summarize_l20_logits_boundary_trace_shadow",
    )
    events = [
        {
            "eligible": True,
            "reasons": [],
            "metadata": {
                "logits_shape": [1, 10],
                "logits_dtype": "torch.float16",
                "shadow_epilogue": {
                    "would_use_epilogue": True,
                    "avoidable_logits_materialization_bytes": 20,
                    "fallback_reasons": [],
                },
            },
        },
        {
            "eligible": False,
            "reasons": ["prefill"],
            "metadata": {
                "logits_shape": [4, 10],
                "logits_dtype": "torch.float16",
                "shadow_epilogue": {
                    "would_use_epilogue": False,
                    "avoidable_logits_materialization_bytes": 0,
                    "fallback_reasons": ["prefill"],
                },
            },
        },
    ]
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    summary = summarizer.summarize(summarizer.read_events(trace))
    assert summary["shadow_events"] == 2
    assert summary["shadow_eligible_events"] == 1
    assert summary["shadow_fallback_events"] == 1
    assert summary["shadow_reason_counts"] == {"prefill": 1}
    assert summary["shadow_avoidable_logits_bytes"] == 20


def test_l20_logits_boundary_campaign_summarizer_reads_serving_reports(tmp_path):
    summarizer = load_module(
        "scripts/summarize_l20_logits_boundary_campaign.py",
        "summarize_l20_logits_boundary_campaign",
    )
    campaign = tmp_path / "campaign"
    campaign.mkdir()
    (campaign / "run-config.json").write_text(
        json.dumps({"model": "Qwen/Qwen3-0.6B"}),
        encoding="utf-8",
    )
    (campaign / "logits-boundary-summary.json").write_text(
        json.dumps(
            {
                "total_events": 10,
                "eligible_events": 7,
                "fallback_events": 3,
                "eligible_fraction": 0.7,
                "reason_counts": {"prefill": 3},
            }
        ),
        encoding="utf-8",
    )
    base_report = {
        "completed": 4,
        "failed": 0,
        "request_throughput": 2.0,
        "output_throughput": 64.0,
        "median_ttft_ms": 20.0,
        "p95_ttft_ms": 30.0,
        "median_itl_ms": 5.0,
        "p95_itl_ms": 6.0,
    }
    (campaign / "c1-i512-r1.json").write_text(
        json.dumps(base_report),
        encoding="utf-8",
    )
    second = dict(base_report)
    second["median_itl_ms"] = 7.0
    second["output_throughput"] = 60.0
    (campaign / "c1-i512-r2.json").write_text(
        json.dumps(second),
        encoding="utf-8",
    )

    summary = summarizer.summarize(campaign)
    assert summary["serving_report_count"] == 2
    assert summary["trace_summary"]["eligible_fraction"] == 0.7
    assert summary["shapes"][0]["max_concurrency"] == 1
    assert summary["shapes"][0]["input_tokens"] == 512
    assert summary["shapes"][0]["metrics"]["median_itl_ms"] == 6.0
    assert summary["shapes"][0]["metrics"]["output_throughput"] == 62.0
