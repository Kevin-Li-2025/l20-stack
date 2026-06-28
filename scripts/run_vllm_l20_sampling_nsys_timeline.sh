#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_sampling_nsys_timeline.sh \
  MODEL SERVED_NAME SAMPLER_MODE OUTPUT_DIR VLLM_SOURCE_DIR

Runs one real vLLM stochastic serving profile under Nsight Systems for sampler
path evidence. SAMPLER_MODE must be flashinfer, torch, or l20.

Important environment:
  NSYS_BIN              Optional explicit path to nsys.
  PYTHON                Python executable. Defaults to python.
  PORT                  Server port. Defaults to 8000.
  NSYS_DURATION         Capture duration in seconds. Defaults to 240.
  INPUT_TOKENS          Random prompt length. Defaults to 512.
  OUTPUT_TOKENS         Random output length. Defaults to 32.
  NUM_PROMPTS           Benchmark prompt count. Defaults to 16.
  MAX_CONCURRENCY       Benchmark max concurrency. Defaults to 4.
  REQUEST_RATE          Benchmark request rate. Defaults to inf.
  TEMPERATURE           Sampling temperature. Defaults to 0.8.
  TOP_P                 Sampling top-p. Defaults to 0.9.
  TOP_K                 Sampling top-k. Defaults to 50.
  MAX_MODEL_LEN         vLLM max model length. Defaults to 2048.
  GPU_MEMORY_UTILIZATION Defaults to 0.70.
  L20_NSYS_TMPDIR       Short writable tmpdir. Defaults to $HOME/tmp/l20-nsys.
  L20_TRACE             Set to 1 to record L20 sampler eligibility trace.
  REQUIRE_SAMPLER_KERNEL Set to 0 to allow zero matched sampler kernels.
                         Defaults to 1 for flashinfer/l20 and 0 for torch.
EOF
  exit 2
fi

model=$1
served_name=$2
sampler_mode=$3
output_dir=$4
vllm_source_dir=$5

case "$sampler_mode" in
  flashinfer) use_flashinfer_sampler=1 ;;
  l20) use_flashinfer_sampler=1 ;;
  torch) use_flashinfer_sampler=0 ;;
  *) echo "SAMPLER_MODE must be flashinfer, torch, or l20" >&2; exit 2 ;;
esac

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python}
port=${PORT:-8000}
duration=${NSYS_DURATION:-240}
input_tokens=${INPUT_TOKENS:-512}
output_tokens=${OUTPUT_TOKENS:-32}
num_prompts=${NUM_PROMPTS:-16}
max_concurrency=${MAX_CONCURRENCY:-4}
request_rate=${REQUEST_RATE:-inf}
temperature=${TEMPERATURE:-0.8}
top_p=${TOP_P:-0.9}
top_k=${TOP_K:-50}
max_model_len=${MAX_MODEL_LEN:-2048}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.70}
attention_backend=${ATTENTION_BACKEND:-FLASHINFER}
extra_vllm_args=${VLLM_EXTRA_ARGS:-}
l20_trace=${L20_TRACE:-0}
require_sampler_kernel=${REQUIRE_SAMPLER_KERNEL:-}
if [[ -z "$require_sampler_kernel" ]]; then
  if [[ "$sampler_mode" == "flashinfer" || "$sampler_mode" == "l20" ]]; then
    require_sampler_kernel=1
  else
    require_sampler_kernel=0
  fi
fi

find_nsys() {
  if [[ -n "${NSYS_BIN:-}" ]]; then
    if command -v "$NSYS_BIN" >/dev/null 2>&1; then
      command -v "$NSYS_BIN"
    else
      echo "$NSYS_BIN"
    fi
    return 0
  fi
  if command -v nsys >/dev/null 2>&1; then
    command -v nsys
    return 0
  fi
  local candidate
  for candidate in \
    /usr/local/cuda/bin/nsys \
    /usr/local/cuda-13.0/bin/nsys \
    /opt/nvidia/nsight-systems/*/bin/nsys \
    /opt/nvidia/nsight-compute/*/host/target-linux-x64/nsys; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done
  return 1
}

nsys_bin="$(find_nsys || true)"
if [[ -z "$nsys_bin" || ! -x "$nsys_bin" ]]; then
  echo "nsys is required; set NSYS_BIN or install NVIDIA Nsight Systems" >&2
  exit 2
fi

mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)
stats_dir="$output_dir/stats"
tmp_dir=${L20_NSYS_TMPDIR:-"${HOME:-$output_dir}/tmp/l20-nsys"}
mkdir -p "$stats_dir" "$tmp_dir"
export TMPDIR="$tmp_dir"

python_dir=$(dirname "$("$python_bin" -c 'import sys; print(sys.executable)')")
export PATH="$python_dir:$PATH"
export PYTHONPATH="$vllm_source_dir:$repo_root/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ "$sampler_mode" == "flashinfer" || "$sampler_mode" == "l20" ]]; then
  eval "$("$python_bin" - <<'PY'
import shlex
from l20_stack.flashinfer_env import configure_flashinfer_cuda13_env

env = configure_flashinfer_cuda13_env(required=True)
print(f"export CUDA_HOME={shlex.quote(env.cuda_home)}")
print(f"export CUDACXX={shlex.quote(env.nvcc)}")
print(f"export PATH={shlex.quote(env.cuda_home + '/bin')}:$PATH")
print(f"export LD_LIBRARY_PATH={shlex.quote(env.cuda_home + '/lib64')}:${{LD_LIBRARY_PATH:-}}")
PY
)"
  "$python_bin" "$repo_root/scripts/prewarm_flashinfer_sampling.py" \
    >"$output_dir/flashinfer-prewarm.json" \
    2>"$output_dir/flashinfer-prewarm.stderr"
fi

if [[ "$sampler_mode" == "l20" ]]; then
  "$python_bin" "$repo_root/integrations/vllm/install_l20_topk_topp_sampler.py" \
    --vllm-source "$vllm_source_dir" >/dev/null
  for prewarm_batch in 1 2 3 4; do
    "$python_bin" "$repo_root/scripts/prewarm_l20_topk_topp_sampling.py" \
      --batch "$prewarm_batch" \
      --vocab 151936 \
      --top-k "$top_k" \
      --top-p "$top_p" \
      >"$output_dir/l20-prewarm-b${prewarm_batch}.json" \
      2>"$output_dir/l20-prewarm-b${prewarm_batch}.stderr" || true
  done
fi

compilation_config='{"mode":3,"splitting_ops":[],"cudagraph_mode":"FULL","pass_config":{"fuse_rope_kvcache":false}}'
server_log="$output_dir/server.log"
nsys_log="$output_dir/nsys.log"
profile_prefix="$output_dir/vllm-sampling"
rm -f \
  "$server_log" \
  "$nsys_log" \
  "$output_dir/timeline-failure.json" \
  "$output_dir/timeline-summary.json" \
  "$output_dir/sampling-path.json" \
  "$profile_prefix".nsys-rep \
  "$profile_prefix".sqlite
rm -f "$stats_dir"/*.csv

server_args=(
  "$model"
  --served-model-name "$served_name"
  --host 127.0.0.1
  --port "$port"
  --trust-remote-code
  --dtype half
  --max-model-len "$max_model_len"
  --gpu-memory-utilization "$gpu_memory_utilization"
  --attention-backend "$attention_backend"
  --no-enable-prefix-caching
  --generation-config vllm
  --compilation-config "$compilation_config"
)
if [[ -n "$extra_vllm_args" ]]; then
  # shellcheck disable=SC2206
  extra_args=( $extra_vllm_args )
  server_args+=("${extra_args[@]}")
fi

"$python_bin" - "$output_dir/run-config.json" <<PY
import json, os, sys
path = sys.argv[1]
payload = {
    "schema_version": 1,
    "model": "$model",
    "served_name": "$served_name",
    "sampler_mode": "$sampler_mode",
    "use_flashinfer_sampler": "$use_flashinfer_sampler" == "1",
    "attention_backend": "$attention_backend",
    "input_tokens": int("$input_tokens"),
    "output_tokens": int("$output_tokens"),
    "num_prompts": int("$num_prompts"),
    "max_concurrency": int("$max_concurrency"),
    "request_rate": "$request_rate",
    "temperature": float("$temperature"),
    "top_p": float("$top_p"),
    "top_k": int("$top_k"),
    "max_model_len": int("$max_model_len"),
    "gpu_memory_utilization": float("$gpu_memory_utilization"),
    "nsys_bin": "$nsys_bin",
    "nsys_duration_seconds": int("$duration"),
    "tmpdir": os.environ.get("TMPDIR"),
    "cuda_home": os.environ.get("CUDA_HOME"),
    "cudacxx": os.environ.get("CUDACXX"),
    "require_sampler_kernel": "$require_sampler_kernel" != "0",
    "l20_trace": "$l20_trace" == "1",
}
open(path, "w", encoding="utf-8").write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

l20_trace_path=""
if [[ "$sampler_mode" == "l20" && "$l20_trace" == "1" ]]; then
  l20_trace_path="$output_dir/l20-topk-topp-trace.jsonl"
  rm -f "$l20_trace_path"
fi

cleanup() {
  if [[ -n "${nsys_pid:-}" ]] && kill -0 "$nsys_pid" 2>/dev/null; then
    kill -- "-$nsys_pid" 2>/dev/null || true
    wait "$nsys_pid" 2>/dev/null || true
  fi
  if [[ "$sampler_mode" == "l20" ]]; then
    "$python_bin" "$repo_root/integrations/vllm/install_l20_topk_topp_sampler.py" \
      --vllm-source "$vllm_source_dir" --uninstall >/dev/null || true
  fi
}
trap cleanup EXIT

write_failure_report() {
  local reason=$1
  "$python_bin" - "$output_dir" "$reason" <<'PY'
import json, sys
from pathlib import Path

output_dir = Path(sys.argv[1])
reason = sys.argv[2]
report = {
    "schema_version": 1,
    "server_start_failed": True,
    "server_start_failure_reason": reason,
}
for name in ("server.log", "nsys.log", "flashinfer-prewarm.stderr"):
    path = output_dir / name
    report[f"{name}_tail"] = (
        path.read_text(encoding="utf-8", errors="replace")[-12000:]
        if path.exists()
        else ""
    )
(output_dir / "timeline-failure.json").write_text(
    json.dumps(report, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
}

cd "$vllm_source_dir"
echo "Using Nsight Systems CLI: $nsys_bin" | tee -a "$nsys_log"
setsid "$nsys_bin" profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --cpuctxsw=none \
  --backtrace=none \
  --cuda-memory-usage=false \
  --cuda-graph-trace=graph \
  --force-overwrite=true \
  --export=sqlite \
  --duration "$duration" \
  --kill=sigterm \
  --wait=all \
  --output "$profile_prefix" \
  env \
    PYTHONPATH="$PYTHONPATH" \
    VLLM_USE_FLASHINFER_SAMPLER="$use_flashinfer_sampler" \
    VLLM_L20_TOPK_TOPP_SAMPLER="$([[ "$sampler_mode" == "l20" ]] && echo 1 || echo 0)" \
    VLLM_L20_TOPK_TOPP_SAMPLER_TRACE="$l20_trace_path" \
    "$python_bin" -m vllm.entrypoints.cli.main serve "${server_args[@]}" \
  >"$server_log" 2>>"$nsys_log" &
nsys_pid=$!

for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
    break
  fi
  if ! kill -0 "$nsys_pid" 2>/dev/null; then
    write_failure_report "server_or_profiler_exited_before_health"
    tail -160 "$server_log" >&2 || true
    tail -160 "$nsys_log" >&2 || true
    exit 1
  fi
  sleep 5
done
if ! curl -fsS "http://127.0.0.1:$port/health" >/dev/null; then
  write_failure_report "health_check_timeout"
  tail -160 "$server_log" >&2 || true
  tail -160 "$nsys_log" >&2 || true
  exit 1
fi

"$python_bin" -m vllm.entrypoints.cli.main bench serve \
  --backend openai \
  --model "$served_name" \
  --tokenizer "$model" \
  --host 127.0.0.1 \
  --port "$port" \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len "$input_tokens" \
  --random-output-len "$output_tokens" \
  --num-prompts "$num_prompts" \
  --request-rate "$request_rate" \
  --max-concurrency "$max_concurrency" \
  --ignore-eos \
  --temperature "$temperature" \
  --top-p "$top_p" \
  --top-k "$top_k" \
  --save-result \
  --result-dir "$output_dir" \
  --result-filename "serving.json"

"$python_bin" - "$output_dir/serving.json" "$num_prompts" <<'PY'
import json, sys
path, expected = sys.argv[1], int(sys.argv[2])
report = json.load(open(path, encoding="utf-8"))
if report.get("completed") != expected or report.get("failed") != 0:
    raise SystemExit(
        f"invalid benchmark report {path}: "
        f"completed={report.get('completed')} failed={report.get('failed')}"
    )
PY

"$python_bin" "$repo_root/scripts/inspect_vllm_sampling_path.py" \
  --log "$server_log" \
  --output "$output_dir/sampling-path.json" >/dev/null || true

wait "$nsys_pid"
trap - EXIT
if [[ "$sampler_mode" == "l20" ]]; then
  "$python_bin" "$repo_root/integrations/vllm/install_l20_topk_topp_sampler.py" \
    --vllm-source "$vllm_source_dir" --uninstall >/dev/null || true
fi

profile_rep="$profile_prefix.nsys-rep"
if [[ ! -f "$profile_rep" ]]; then
  write_failure_report "missing_nsys_report"
  exit 1
fi

for report in cuda_gpu_kern_sum cuda_kern_exec_sum cuda_api_sum nvtx_sum cuda_gpu_trace; do
  "$nsys_bin" stats \
    --force-export true \
    --force-overwrite true \
    --report "$report" \
    --format csv \
    --output "$stats_dir/$report" \
    "$profile_rep" \
    >/dev/null
done

cd "$repo_root"
"$python_bin" scripts/summarize_nsys_timeline.py \
  --input-dir "$stats_dir" \
  --output "$output_dir/timeline-summary.json" \
  --match-label sampler \
  --match-kernel sampling \
  --match-kernel Sampling \
  --match-kernel gumbel \
  --match-kernel Gumbel \
  --match-kernel top_p \
  --match-kernel top_k \
  --match-kernel topp \
  --match-kernel topk \
  --match-kernel TopP \
  --match-kernel TopK \
  --match-kernel _topk_topp_reduce_sample_seed_kernel \
  --match-kernel _topk_topp_partial_kernel

"$python_bin" scripts/summarize_nsys_kernel_families.py \
  --input-dir "$stats_dir" \
  --output-json "$output_dir/kernel-family-summary.json" \
  --output-md "$output_dir/kernel-family-summary.md"

if [[ "$sampler_mode" == "l20" && -n "$l20_trace_path" && -f "$l20_trace_path" ]]; then
  "$python_bin" - "$l20_trace_path" "$output_dir/l20-topk-topp-summary.json" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

trace_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
rows = [json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()]
eligible = sum(1 for row in rows if row.get("eligible"))
reasons = Counter(reason for row in rows for reason in row.get("reasons", []))
shapes = Counter(
    "x".join(map(str, row.get("metadata", {}).get("logits_shape", [])))
    for row in rows
)
summary = {
    "schema_version": 1,
    "total_events": len(rows),
    "eligible_events": eligible,
    "fallback_events": len(rows) - eligible,
    "eligible_fraction": eligible / len(rows) if rows else 0.0,
    "reason_counts": dict(reasons.most_common()),
    "logits_shape_counts": {
        key: value for key, value in shapes.most_common() if key
    },
}
out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
PY
fi

"$python_bin" - "$output_dir/timeline-summary.json" "$require_sampler_kernel" <<'PY'
import json
import sys

path, require = sys.argv[1], sys.argv[2] != "0"
summary = json.load(open(path, encoding="utf-8"))
count = int(summary.get("matched_kernel_instance_count") or 0)
if require and count <= 0:
    raise SystemExit(
        "Nsight Systems captured zero matched sampling kernel instances; "
        "refusing to classify this as a GPU-sampler serving timeline. Set "
        "REQUIRE_SAMPLER_KERNEL=0 only when intentionally recording a "
        "negative or unknown integration run."
    )
PY
