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
    def __init__(self, shape):
        self.shape = shape
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
    assert event["metadata"]["sampling"]["temperature_min"] == 0.8
    assert event["metadata"]["sampling"]["top_k_max"] == 50.0
    assert event["metadata"]["sampling"]["top_p_min"] == 0.9


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


def test_install_l20_logits_boundary_trace_patches_and_uninstalls(tmp_path):
    installer = load_module(
        "integrations/vllm/install_l20_logits_boundary_trace.py",
        "install_l20_logits_boundary_trace",
    )
    package = tmp_path / "vllm"
    target = package / "v1/worker/gpu/model_runner.py"
    target.parent.mkdir(parents=True)
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

    installer.install(package)
    patched = target.read_text(encoding="utf-8")
    assert "maybe_trace_l20_logits_boundary" in patched
    assert (package / "v1/worker/gpu/l20_logits_boundary_trace.py").exists()
    installer.install(package)
    assert target.read_text(encoding="utf-8") == patched

    installer.uninstall(package)
    restored = target.read_text(encoding="utf-8")
    assert "maybe_trace_l20_logits_boundary" not in restored
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
            "metadata": {"logits_shape": [2, 10], "hidden_shape": [2, 4]},
        },
        {
            "eligible": False,
            "reasons": ["prefill", "token_logprobs"],
            "metadata": {"logits_shape": [4, 10], "hidden_shape": [4, 4]},
        },
        {
            "eligible": False,
            "reasons": ["prefill"],
            "metadata": {"logits_shape": [4, 10], "hidden_shape": [4, 4]},
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
