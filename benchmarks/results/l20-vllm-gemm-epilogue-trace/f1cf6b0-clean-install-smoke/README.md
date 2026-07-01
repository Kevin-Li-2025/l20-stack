# Clean Upstream GEMM Epilogue Install Smoke

This artifact validates the first fallback-first LM-head GEMM epilogue hook on
a clean upstream vLLM checkout.

## Source

- vLLM source: `/workspace/vllm-clean`
- vLLM commit: `f1cf6b0`
- vLLM branch: `main`
- Dirty before install: `False`
- Host GPU: `NVIDIA A100-SXM4-80GB`

The A100 host is only a source-level control environment. This artifact is not
an L20 performance result.

## What Was Installed

Installer:

```text
integrations/vllm/install_l20_gemm_epilogue_trace.py
```

Helper:

```text
integrations/vllm/l20_gemm_epilogue_trace.py
```

Patched upstream files:

```text
vllm/model_executor/layers/logits_processor.py
vllm/v1/worker/gpu/model_runner.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/gpu/l20_gemm_epilogue_trace.py
```

Patch points after install:

| Patch point | Location |
| --- | --- |
| `try_sample_from_lm_head` | `vllm/model_executor/layers/logits_processor.py:75` |
| native runner call | `vllm/v1/worker/gpu/model_runner.py:1055` |
| two-stage execute call | `vllm/v1/worker/gpu_model_runner.py:4362` |
| two-stage sample handoff | `vllm/v1/worker/gpu_model_runner.py:4470` |

## Checks

Remote unit tests:

```text
PYTHONPATH=src /workspace/venvs/l20-stack/bin/python -m pytest \
  tests/test_l20_gemm_epilogue_trace.py \
  tests/test_l20_gemm_epilogue_installer.py -q
```

Result:

```text
5 passed in 0.75s
```

Remote `py_compile` passed for:

```text
/workspace/vllm-clean/vllm/model_executor/layers/logits_processor.py
/workspace/vllm-clean/vllm/v1/worker/gpu/model_runner.py
/workspace/vllm-clean/vllm/v1/worker/gpu_model_runner.py
/workspace/vllm-clean/vllm/v1/worker/gpu/l20_gemm_epilogue_trace.py
```

Uninstall restored the clean upstream checkout:

```text
STATUS_AFTER_UNINSTALL
```

No tracked or untracked files remained in `git status --short`.

## Claim Boundary

This proves:

- the fallback-first API shape installs on clean upstream vLLM;
- the modified upstream files compile;
- uninstall restores a clean source tree.

This does not prove:

- L20 performance;
- vLLM serving ITL improvement;
- CUDA graph compatibility under a live server.

The next result must be a real L20 vLLM serving trace with
`VLLM_L20_GEMM_EPILOGUE_TRACE` enabled.
