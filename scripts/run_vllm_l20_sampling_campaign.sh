#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODEL SERVED_NAME SAMPLER_MODE OUTPUT_DIR" >&2
  echo "SAMPLER_MODE: flashinfer|torch" >&2
  exit 2
fi

model=$1
served_name=$2
sampler_mode=$3
output_dir=$4

case "$sampler_mode" in
  flashinfer) use_flashinfer_sampler=1 ;;
  torch) use_flashinfer_sampler=0 ;;
  *) echo "unknown SAMPLER_MODE: $sampler_mode" >&2; exit 2 ;;
esac

port=${PORT:-8000}
inputs=${INPUTS:-"512"}
concurrencies=${CONCURRENCIES:-"1 16"}
runs=${RUNS:-1}
num_prompts=${NUM_PROMPTS:-32}
output_tokens=${OUTPUT_TOKENS:-32}
temperature=${TEMPERATURE:-0.8}
top_p=${TOP_P:-0.9}
top_k=${TOP_K:-50}
extra_vllm_pythonpath=${VLLM_SOURCE_TREE:-"$HOME/vllm-l20-upstream"}
python_bin=${PYTHON:-python}
mkdir -p "$output_dir"

export PYTHONPATH="$extra_vllm_pythonpath${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER="$use_flashinfer_sampler"

if [[ "$sampler_mode" == "flashinfer" ]]; then
  eval "$(PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" - <<'PY'
import shlex
from l20_stack.flashinfer_env import configure_flashinfer_cuda13_env

env = configure_flashinfer_cuda13_env(required=True)
print(f"export CUDA_HOME={shlex.quote(env.cuda_home)}")
print(f"export CUDACXX={shlex.quote(env.nvcc)}")
print(f"export PATH={shlex.quote(env.cuda_home + '/bin')}:$PATH")
PY
)"
  PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" scripts/prewarm_flashinfer_sampling.py \
    >"$output_dir/flashinfer-prewarm.json"
fi

compilation_config='{"mode":3,"splitting_ops":[],"cudagraph_mode":"FULL","pass_config":{"fuse_rope_kvcache":false}}'
server_log="$output_dir/server.log"
setsid env \
  PYTHONPATH="$PYTHONPATH" \
  VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER" \
  vllm serve "$model" \
    --served-model-name "$served_name" \
    --host 127.0.0.1 \
    --port "$port" \
    --attention-backend FLASHINFER \
    --no-enable-prefix-caching \
    --compilation-config "$compilation_config" \
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
      filename="c${concurrency}-i${input_tokens}-r${run}.json"
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
        --top-p "$top_p" \
        --top-k "$top_k" \
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

PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" scripts/inspect_vllm_sampling_path.py \
  --log "$server_log" \
  --output "$output_dir/sampling-path.json"
