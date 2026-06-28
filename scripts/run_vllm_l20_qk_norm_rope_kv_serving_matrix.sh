#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_qk_norm_rope_kv_serving_matrix.sh \
  MODEL SERVED_NAME OUTPUT_DIR VLLM_SOURCE_DIR

Environment defaults:
  INPUTS="512 1024"
  CONCURRENCIES="1 4"
  RUNS=2
  NUM_PROMPTS=24
  OUTPUT_TOKENS=64
  REQUEST_RATE=inf
EOF
  exit 2
fi

export INPUTS="${INPUTS:-512 1024}"
export CONCURRENCIES="${CONCURRENCIES:-1 4}"
export RUNS="${RUNS:-2}"
export NUM_PROMPTS="${NUM_PROMPTS:-24}"
export OUTPUT_TOKENS="${OUTPUT_TOKENS:-64}"
export REQUEST_RATE="${REQUEST_RATE:-inf}"

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
exec "$script_dir/run_vllm_l20_qk_norm_rope_kv_serving_smoke.sh" "$@"
