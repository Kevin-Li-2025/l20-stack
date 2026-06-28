#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODEL SERVED_NAME SAMPLER_MODE OUTPUT_DIR" >&2
  echo "SAMPLER_MODE: flashinfer|torch|l20" >&2
  exit 2
fi

model=$1
served_name=$2
sampler_mode=$3
output_dir=$4

case "$sampler_mode" in
  flashinfer) use_flashinfer_sampler=1 ;;
  l20) use_flashinfer_sampler=1 ;;
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
max_model_len=${MAX_MODEL_LEN:-2048}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.70}
l20_trace=${L20_TRACE:-0}
extra_vllm_pythonpath=${VLLM_SOURCE_TREE:-"$HOME/vllm-l20-upstream"}
python_bin=${PYTHON:-python}
mkdir -p "$output_dir"

python_dir=$(dirname "$("$python_bin" -c 'import sys; print(sys.executable)')")
if [[ -x "$python_dir/vllm" || -x "$python_dir/ninja" ]]; then
  export PATH="$python_dir:$PATH"
fi
export PYTHONPATH="$extra_vllm_pythonpath:$(pwd)/src${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_USE_FLASHINFER_SAMPLER="$use_flashinfer_sampler"

if [[ "$sampler_mode" != "l20" ]]; then
  "$python_bin" integrations/vllm/install_l20_topk_topp_sampler.py \
    --vllm-source "$extra_vllm_pythonpath" \
    --uninstall >/dev/null || true
fi

if [[ "$sampler_mode" == "flashinfer" || "$sampler_mode" == "l20" ]]; then
  eval "$(PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" - <<'PY'
import shlex
from l20_stack.flashinfer_env import configure_flashinfer_cuda13_env

env = configure_flashinfer_cuda13_env(required=True)
print(f"export CUDA_HOME={shlex.quote(env.cuda_home)}")
print(f"export CUDACXX={shlex.quote(env.nvcc)}")
print(f"export PATH={shlex.quote(env.cuda_home + '/bin')}:$PATH")
PY
)"
  if ! PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" scripts/prewarm_flashinfer_sampling.py \
    >"$output_dir/flashinfer-prewarm.json" 2>"$output_dir/flashinfer-prewarm.stderr"; then
    "$python_bin" - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
prewarm_path = output_dir / "flashinfer-prewarm.json"
stderr_path = output_dir / "flashinfer-prewarm.stderr"
try:
    prewarm = json.loads(prewarm_path.read_text(encoding="utf-8"))
except json.JSONDecodeError:
    prewarm = {
        "schema_version": 1,
        "status": "error",
        "error": prewarm_path.read_text(encoding="utf-8", errors="replace")[-4000:],
    }
result = {
    "schema_version": 1,
    "sampler_mode": "flashinfer",
    "prewarm_failed": True,
    "flashinfer_sampling_available": False,
    "cpu_fallback_suspected": False,
    "prewarm": prewarm,
    "stderr_tail": stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:],
    "notes": [
        "FlashInfer sampling JIT failed before vLLM serving started.",
        "No FlashInfer stochastic serving ITL claim should be made from this run.",
    ],
}
(output_dir / "sampling-path.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
    exit 1
  fi
fi

if [[ "$sampler_mode" == "l20" ]]; then
  "$python_bin" integrations/vllm/install_l20_topk_topp_sampler.py \
    --vllm-source "$extra_vllm_pythonpath" >/dev/null
  export VLLM_L20_TOPK_TOPP_SAMPLER=1
  if [[ "$l20_trace" == "1" ]]; then
    export VLLM_L20_TOPK_TOPP_SAMPLER_TRACE="$output_dir/l20-topk-topp-trace.jsonl"
    rm -f "$VLLM_L20_TOPK_TOPP_SAMPLER_TRACE"
  else
    export VLLM_L20_TOPK_TOPP_SAMPLER_TRACE=""
  fi
  if ! PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" scripts/prewarm_l20_topk_topp_sampling.py \
    --batch 1 \
    --vocab 151936 \
    --top-k "$top_k" \
    --top-p "$top_p" \
    >"$output_dir/l20-prewarm.json" 2>"$output_dir/l20-prewarm.stderr"; then
    "$python_bin" - "$output_dir" <<'PY'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
stderr_path = output_dir / "l20-prewarm.stderr"
result = {
    "schema_version": 1,
    "sampler_mode": "l20",
    "prewarm_failed": True,
    "l20_sampling_available": False,
    "stderr_tail": stderr_path.read_text(encoding="utf-8", errors="replace")[-4000:],
    "notes": [
        "The custom L20 top-k/top-p sampler failed before vLLM serving started.",
        "No L20 stochastic serving ITL claim should be made from this run.",
    ],
}
(output_dir / "sampling-path.json").write_text(
    json.dumps(result, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
    exit 1
  fi
  PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" scripts/prewarm_l20_topk_topp_sampling.py \
    --batch 4 \
    --vocab 151936 \
    --top-k "$top_k" \
    --top-p "$top_p" \
    >"$output_dir/l20-prewarm-b4.json" 2>"$output_dir/l20-prewarm-b4.stderr" || true
fi

compilation_config='{"mode":3,"splitting_ops":[],"cudagraph_mode":"FULL","pass_config":{"fuse_rope_kvcache":false}}'
server_log="$output_dir/server.log"
server_env=(
  "PYTHONPATH=$PYTHONPATH"
  "VLLM_USE_FLASHINFER_SAMPLER=$VLLM_USE_FLASHINFER_SAMPLER"
)
if [[ "$sampler_mode" == "l20" ]]; then
  server_env+=(
    "VLLM_L20_TOPK_TOPP_SAMPLER=${VLLM_L20_TOPK_TOPP_SAMPLER:-1}"
    "VLLM_L20_TOPK_TOPP_SAMPLER_TRACE=${VLLM_L20_TOPK_TOPP_SAMPLER_TRACE:-}"
  )
fi
setsid env \
  "${server_env[@]}" \
  vllm serve "$model" \
    --served-model-name "$served_name" \
    --host 127.0.0.1 \
    --port "$port" \
    --max-model-len "$max_model_len" \
    --gpu-memory-utilization "$gpu_memory_utilization" \
    --attention-backend FLASHINFER \
    --generation-config vllm \
    --no-enable-prefix-caching \
    --compilation-config "$compilation_config" \
    >"$server_log" 2>&1 &
server_pid=$!

cleanup() {
  kill -- "-$server_pid" 2>/dev/null || true
  wait "$server_pid" 2>/dev/null || true
}
trap cleanup EXIT

write_sampling_failure_report() {
  local reason=$1
  PYTHONPATH="$(pwd)/src:$PYTHONPATH" "$python_bin" scripts/inspect_vllm_sampling_path.py \
    --log "$server_log" \
    --output "$output_dir/sampling-path.json" >/dev/null || true
  "$python_bin" - "$output_dir/sampling-path.json" "$reason" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
reason = sys.argv[2]
if path.exists():
    report = json.loads(path.read_text(encoding="utf-8"))
else:
    report = {"schema_version": 1, "matches": {}, "match_counts": {}}
report["server_start_failed"] = True
report["server_start_failure_reason"] = reason
report.setdefault("notes", []).append(
    "The server did not become healthy, so this run has no valid serving ITL result."
)
path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$server_pid" 2>/dev/null; then
    write_sampling_failure_report "server_process_exited_before_health"
    tail -160 "$server_log" >&2
    exit 1
  fi
  sleep 5
done
if ! curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
  write_sampling_failure_report "health_check_timeout"
  tail -160 "$server_log" >&2
  exit 1
fi

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

if [[ "$sampler_mode" == "l20" && -n "${VLLM_L20_TOPK_TOPP_SAMPLER_TRACE:-}" ]]; then
  "$python_bin" - "$output_dir/l20-topk-topp-trace.jsonl" "$output_dir/l20-topk-topp-summary.json" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

trace_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
events = []
if trace_path.exists():
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
reason_counts = Counter()
eligible = 0
shape_counts = Counter()
for event in events:
    if event.get("eligible"):
        eligible += 1
    for reason in event.get("reasons", []):
        reason_counts[reason] += 1
    shape = event.get("metadata", {}).get("logits_shape")
    if shape:
        shape_counts["x".join(str(dim) for dim in shape)] += 1
summary = {
    "schema_version": 1,
    "total_events": len(events),
    "eligible_events": eligible,
    "fallback_events": len(events) - eligible,
    "eligible_fraction": eligible / len(events) if events else 0.0,
    "reason_counts": dict(reason_counts.most_common()),
    "logits_shape_counts": dict(shape_counts.most_common()),
}
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
fi
