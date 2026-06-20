# Research Plan

The pasted project idea is useful as ambition, but the repo should earn claims in stages.

## Non-Negotiable Rules

- A result is not a result until the exact command, config, hardware, and output are recorded.
- Synthetic smoke tests are only for correctness. They do not support performance claims.
- Custom CUDA work starts after baseline profiling shows a specific bottleneck.
- Large model or dataset artifacts stay outside Git.

## Phase 0: Planning Harness

Goal: prevent expensive failed runs.

- Keep experiment configs small and explicit.
- Estimate memory before training.
- Add tests for config parsing and budget calculations.

Exit criteria:

- `unittest` passes locally.
- `l20_stack.cli plan` emits JSON for the default config.

## Phase 1: QLoRA Baseline

Goal: run a real single-L20 fine-tuning job.

- Start with a 7B or 14B model.
- Use a tiny checked-in JSONL fixture for CI-style smoke tests.
- Use external dataset paths or Hugging Face dataset names for real runs.
- Record peak memory, tokens/sec, loss curve, and final eval.

Exit criteria:

- One reproducible real run on L20.
- One benchmark table with no manual edits.

## Phase 2: Serving Baseline

Goal: measure vLLM before modifying inference internals.

- Add request-shape fixtures.
- Measure prefill latency, decode throughput, p50/p95 latency, and memory.
- Compare quantization settings on the same model and prompts.

Exit criteria:

- Baseline report for at least one real model on L20.
- Profiling data identifies the next bottleneck.

## Phase 3: Systems Work

Goal: only build custom kernels where they have a measured reason to exist.

- Start with a narrow benchmark.
- Write correctness tests against a trusted implementation.
- Add profiling before and after every kernel change.

Exit criteria:

- Correctness tests pass.
- Performance improvement survives repeated runs and realistic prompt shapes.
