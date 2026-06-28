#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/profile_qk_norm_rope_kv_ncu.sh [OUT_PREFIX] [TOKENS]

Examples:
  scripts/profile_qk_norm_rope_kv_ncu.sh \
    benchmarks/results/ncu/qk-norm-rope-kv/tokens-64 64

Environment:
  NCU_BIN             Optional explicit path to ncu.
  NCU_LAUNCH_SKIP    Matching kernel launches to skip. Defaults to 5.
  NCU_LAUNCH_COUNT   Matching kernel launches to profile. Defaults to 1.
  PYTHON_BIN          Python executable. Defaults to python3.
  VLLM_SOURCE         Optional vLLM source checkout to prepend to PYTHONPATH.
  ROUNDS             Benchmark timing rounds. Defaults to 1 for profiling.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

output=${1:-benchmarks/results/ncu/qk-norm-rope-kv/tokens-64}
tokens=${2:-64}
rounds=${ROUNDS:-1}
python_bin=${PYTHON_BIN:-python3}
repo_pythonpath="src"
if [[ -n "${VLLM_SOURCE:-}" ]]; then
  repo_pythonpath="${VLLM_SOURCE}:$repo_pythonpath"
fi
if [[ -n "${PYTHONPATH:-}" ]]; then
  repo_pythonpath="$repo_pythonpath:$PYTHONPATH"
fi

scripts/profile_kernel.sh \
  --output "$output" \
  --kernel-name 'regex:_l20_qk_norm_rope_kv_kernel' \
  --launch-skip "${NCU_LAUNCH_SKIP:-5}" \
  --launch-count "${NCU_LAUNCH_COUNT:-1}" \
  -- env PYTHONPATH="$repo_pythonpath" "$python_bin" scripts/benchmark_qk_norm_rope_kv.py \
    --tokens "$tokens" \
    --rounds "$rounds" \
    --output "${output}-bench.json"
