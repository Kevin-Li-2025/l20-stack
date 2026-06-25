#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL SERVED_NAME OUTPUT_DIR" >&2
  exit 2
fi

model=$1
served_name=$2
output_dir=$3

port=${PORT:-8000}
turns=${TURNS:-8}
prefix_chars=${PREFIX_CHARS:-24000}
max_tokens=${OUTPUT_TOKENS:-32}
temperature=${TEMPERATURE:-0}
prefix_caching=${PREFIX_CACHING:-0}
max_model_len=${MAX_MODEL_LEN:-4096}
enforce_eager=${ENFORCE_EAGER:-1}
extra_vllm_args=${VLLM_EXTRA_ARGS:-}
extra_vllm_pythonpath=${VLLM_SOURCE_TREE:-"$HOME/vllm-l20-upstream"}
python_bin=${PYTHON:-python}
mkdir -p "$output_dir"
python_dir=$(dirname "$python_bin")
if [[ -x "$python_dir/vllm" ]]; then
  export PATH="$python_dir:$PATH"
fi

prefix_args=()
case "$prefix_caching" in
  0) prefix_args=(--no-enable-prefix-caching) ;;
  1) prefix_args=(--enable-prefix-caching) ;;
  *) echo "PREFIX_CACHING must be 0 or 1" >&2; exit 2 ;;
esac
eager_args=()
if [[ "$enforce_eager" == "1" ]]; then
  eager_args=(--enforce-eager)
fi
# shellcheck disable=SC2206
extra_args=(${extra_vllm_args})

export PYTHONPATH="$extra_vllm_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
server_log="$output_dir/server.log"
setsid env PYTHONPATH="$PYTHONPATH" VLLM_USE_FLASHINFER_SAMPLER=1 \
  vllm serve "$model" \
    --served-model-name "$served_name" \
    --host 127.0.0.1 \
    --port "$port" \
    --attention-backend FLASHINFER \
    --max-model-len "$max_model_len" \
    "${eager_args[@]}" \
    "${prefix_args[@]}" \
    "${extra_args[@]}" \
    >"$server_log" 2>&1 &
server_pid=$!

cleanup() {
  kill -- "-$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    tail -160 "$server_log" >&2
    exit 1
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:$port/health" >/dev/null

PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" \
  scripts/benchmark_multiturn_kv_pressure.py \
  --base-url "http://127.0.0.1:$port" \
  --model "$served_name" \
  --turns "$turns" \
  --prefix-chars "$prefix_chars" \
  --max-tokens "$max_tokens" \
  --temperature "$temperature" \
  --output "$output_dir/kv-pressure-prefix-cache-${prefix_caching}.json"
