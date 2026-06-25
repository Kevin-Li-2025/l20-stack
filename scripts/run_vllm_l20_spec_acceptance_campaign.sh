#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODEL SERVED_NAME MODE OUTPUT_DIR" >&2
  echo "MODE: off|custom" >&2
  echo "For MODE=custom, pass current-vLLM speculative flags through SPECULATIVE_ARGS." >&2
  exit 2
fi

model=$1
served_name=$2
mode=$3
output_dir=$4

case "$mode" in
  off) speculative_args=() ;;
  custom)
    if [[ -z "${SPECULATIVE_ARGS:-}" ]]; then
      echo "SPECULATIVE_ARGS is required for MODE=custom" >&2
      exit 2
    fi
    # shellcheck disable=SC2206
    speculative_args=(${SPECULATIVE_ARGS})
    ;;
  *) echo "unknown MODE: $mode" >&2; exit 2 ;;
esac

port=${PORT:-8000}
inputs=${INPUTS:-"512 2048"}
concurrencies=${CONCURRENCIES:-"1 4"}
runs=${RUNS:-1}
num_prompts=${NUM_PROMPTS:-32}
output_tokens=${OUTPUT_TOKENS:-64}
temperature=${TEMPERATURE:-0}
extra_vllm_pythonpath=${VLLM_SOURCE_TREE:-"$HOME/vllm-l20-upstream"}
python_bin=${PYTHON:-python}
mkdir -p "$output_dir"
python_dir=$(dirname "$python_bin")
if [[ -x "$python_dir/vllm" ]]; then
  export PATH="$python_dir:$PATH"
fi

export PYTHONPATH="$extra_vllm_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
server_log="$output_dir/server.log"
setsid env PYTHONPATH="$PYTHONPATH" VLLM_USE_FLASHINFER_SAMPLER=1 \
  vllm serve "$model" \
    --served-model-name "$served_name" \
    --host 127.0.0.1 \
    --port "$port" \
    --attention-backend FLASHINFER \
    --no-enable-prefix-caching \
    "${speculative_args[@]}" \
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

for concurrency in $concurrencies; do
  for input_tokens in $inputs; do
    for run in $(seq 1 "$runs"); do
      filename="mode-${mode}-c${concurrency}-i${input_tokens}-r${run}.json"
      vllm bench serve \
        --backend vllm \
        --base-url "http://127.0.0.1:$port" \
        --model "$served_name" \
        --tokenizer "$model" \
        --dataset-name random \
        --random-input-len "$input_tokens" \
        --random-output-len "$output_tokens" \
        --num-prompts "$num_prompts" \
        --num-warmups 3 \
        --max-concurrency "$concurrency" \
        --disable-tqdm \
        --ignore-eos \
        --temperature "$temperature" \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,95,99 \
        --save-result \
        --result-dir "$output_dir" \
        --result-filename "$filename"
      "$python_bin" - "$output_dir/$filename" "$num_prompts" <<'PY'
import json
import sys

path, expected = sys.argv[1], int(sys.argv[2])
report = json.load(open(path, encoding="utf-8"))
if report.get("completed") != expected or report.get("failed") != 0:
    raise SystemExit(
        f"invalid benchmark report {path}: "
        f"completed={report.get('completed')} failed={report.get('failed')}"
    )
PY
    done
  done
done

PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" \
  scripts/summarize_spec_decode_acceptance.py \
  --log "$server_log" \
  --result-dir "$output_dir" \
  --output "$output_dir/spec-acceptance-summary.json"
