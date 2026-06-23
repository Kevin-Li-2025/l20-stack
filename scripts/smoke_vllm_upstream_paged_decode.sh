#!/usr/bin/env bash
set -euo pipefail

model=${MODEL:-"$HOME/models/Qwen2.5-Coder-1.5B-Instruct"}
source_tree=${VLLM_SOURCE_TREE:-"$HOME/vllm-l20-upstream"}
fragment_dir=${L20_FRAGMENT_DIR:-/tmp/l20-vllm-site}
port=${PORT:-8013}
log=${LOG:-/tmp/l20-upstream-vllm.log}

export PYTHONPATH="$fragment_dir:$source_tree${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER=0

setsid vllm serve "$model" \
  --served-model-name qwen-upstream \
  --host 127.0.0.1 \
  --port "$port" \
  --attention-backend FLASHINFER \
  --enforce-eager \
  --no-enable-prefix-caching \
  >"$log" 2>&1 &
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
    tail -100 "$log"
    exit 1
  fi
  sleep 2
done

curl -fsS "http://127.0.0.1:$port/v1/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen-upstream","prompt":"Write one Python function name:","max_tokens":8,"temperature":0}' \
  > /tmp/l20-upstream-response.json

python - <<'PY'
import json

with open("/tmp/l20-upstream-response.json", encoding="utf-8") as handle:
    response = json.load(handle)
print(
    {
        "finish_reason": response["choices"][0]["finish_reason"],
        "completion_tokens": response["usage"]["completion_tokens"],
    }
)
PY
