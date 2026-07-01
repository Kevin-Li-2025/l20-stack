import importlib.util
import json
from pathlib import Path

import numpy as np


def load_helper():
    path = Path("integrations/vllm/l20_gemm_epilogue_trace.py")
    spec = importlib.util.spec_from_file_location("l20_gemm_epilogue_trace", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Device:
    type = "cuda"


class Tensor:
    shape = (1, 1024)
    dtype = "torch.float16"
    device = Device()


class InputBatch:
    num_reqs = 1
    num_draft_tokens = 0
    has_structured_output_reqs = False
    temperature_cpu = np.array([0.8], dtype=np.float32)
    top_k_cpu = np.array([20], dtype=np.int32)
    top_p_cpu = np.array([0.95], dtype=np.float32)
    min_p_cpu = np.array([0.0], dtype=np.float32)
    frequency_penalties_cpu = np.array([0.0], dtype=np.float32)
    presence_penalties_cpu = np.array([0.0], dtype=np.float32)
    repetition_penalties_cpu = np.array([1.0], dtype=np.float32)
    logits_processing_needs_token_ids = np.array([False])
    num_logprobs = {}
    logprob_token_ids = {}
    has_allowed_token_ids = set()
    bad_words_token_ids = {}
    generators = {}


class GreedyInputBatch(InputBatch):
    all_greedy = True
    no_top_p = True
    no_top_k = True
    temperature_cpu = np.array([-1.0], dtype=np.float32)
    top_k_cpu = np.array([32_000], dtype=np.int32)
    top_p_cpu = np.array([1.0], dtype=np.float32)


class SchedulerOutput:
    num_scheduled_tokens = {"req0": 1}
    total_num_scheduled_tokens = 1


class ParallelConfig:
    tensor_parallel_size = 1


class LogitsProcessor:
    def __init__(self, output=None):
        self.output = output
        self.calls = []

    def try_sample_from_lm_head(
        self,
        lm_head,
        hidden_states,
        sampling_metadata,
        embedding_bias=None,
    ):
        self.calls.append((lm_head, hidden_states, sampling_metadata, embedding_bias))
        return self.output


class LmHead:
    bias = "bias"


class Model:
    def __init__(self, logits_processor):
        self.logits_processor = logits_processor
        self.lm_head = LmHead()


class Runner:
    parallel_config = ParallelConfig()

    def __init__(self, logits_processor):
        self.model = Model(logits_processor)


def read_event(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_gemm_epilogue_trace_calls_fallback_first_api(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    logits_processor = LogitsProcessor()
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        Runner(logits_processor),
        InputBatch(),
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is None
    assert len(logits_processor.calls) == 1
    event = read_event(trace)
    assert event["event"] == "l20_gemm_epilogue_boundary"
    assert event["eligible"] is True
    assert event["reasons"] == []
    assert event["metadata"]["phase"] == "fallback_first_api_trace"
    assert event["metadata"]["api"]["try_api_found"] is True
    assert event["metadata"]["api"]["api_called"] is True
    assert event["metadata"]["api"]["output_enabled"] is False
    assert event["metadata"]["api"]["fallback_to_compute_logits"] is True
    assert event["metadata"]["mutates_outputs"] is False


def test_gemm_epilogue_trace_rejects_unsupported_semantics(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    batch = InputBatch()
    batch.num_logprobs = {"req0": 1}
    batch.frequency_penalties_cpu = np.array([0.1], dtype=np.float32)
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        Runner(LogitsProcessor()),
        batch,
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is None
    event = read_event(trace)
    assert event["eligible"] is False
    assert "token_logprobs" in event["reasons"]
    assert "penalties" in event["reasons"]
    assert event["metadata"]["api"]["api_called"] is False


def test_gemm_epilogue_enable_can_surface_non_none_output(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    output = object()
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_ENABLE", "1")
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        Runner(LogitsProcessor(output=output)),
        GreedyInputBatch(),
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is output
    event = read_event(trace)
    assert event["metadata"]["api"]["api_returned_output"] is True
    assert event["metadata"]["api"]["output_enabled"] is True
    assert event["metadata"]["api"]["fallback_to_compute_logits"] is False
    assert event["metadata"]["mutates_outputs"] is True


def test_gemm_epilogue_enable_can_surface_greedy_candidate_output(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    output = object()

    def fake_candidate(*args, **kwargs):
        return output, None, {
            "attempted": True,
            "mode": "greedy_argmax",
            "returned_output": True,
            "fallback_to_compute_logits": False,
        }

    monkeypatch.setattr(module, "_try_lm_head_greedy_sampler_output", fake_candidate)
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_ENABLE", "1")
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        Runner(LogitsProcessor(output=None)),
        GreedyInputBatch(),
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is output
    event = read_event(trace)
    assert event["eligible"] is True
    assert event["metadata"]["api"]["api_called"] is True
    assert event["metadata"]["api"]["api_returned_output"] is False
    assert event["metadata"]["api"]["fallback_to_compute_logits"] is False
    assert event["metadata"]["epilogue"]["attempted"] is True
    assert event["metadata"]["epilogue"]["returned_output"] is True
    assert event["metadata"]["mutates_outputs"] is True


def test_gemm_epilogue_enable_rejects_non_greedy_candidate(tmp_path, monkeypatch):
    module = load_helper()
    module._TRACE_COUNT = 0
    trace = tmp_path / "gemm.jsonl"
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_TRACE", str(trace))
    monkeypatch.setenv("VLLM_L20_GEMM_EPILOGUE_ENABLE", "1")
    monkeypatch.setenv("VLLM_L20_LOGITS_BOUNDARY_ALLOW_NON_L20", "1")

    result = module.maybe_try_l20_gemm_epilogue(
        Runner(LogitsProcessor(output=None)),
        InputBatch(),
        None,
        Tensor(),
        SchedulerOutput(),
        None,
    )

    assert result is None
    event = read_event(trace)
    assert event["eligible"] is False
    assert "non_greedy_temperature" in event["reasons"]
    assert "top_k" in event["reasons"]
    assert "top_p" in event["reasons"]
    assert event["metadata"]["api"]["api_called"] is False
    assert event["metadata"]["epilogue"]["attempted"] is False
