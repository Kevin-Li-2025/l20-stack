#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 MODEL SERVED_NAME OUTPUT_DIR" >&2
  exit 2
fi

model=$1
served_name=$2
output_dir=$3

kv_dtypes=${KV_DTYPES:-"auto fp8"}
prefix_modes=${PREFIX_MODES:-"0 1"}
base_port=${PORT:-8100}
python_bin=${PYTHON:-python}
mkdir -p "$output_dir"

run_index=0
run_dirs=()
for kv_dtype in $kv_dtypes; do
  for prefix_mode in $prefix_modes; do
    run_dir="$output_dir/kv-${kv_dtype}-prefix-${prefix_mode}"
    run_dirs+=("$run_dir")
    port=$((base_port + run_index))
    run_index=$((run_index + 1))
    echo "[l20-kv-pressure] kv_cache_dtype=$kv_dtype prefix_caching=$prefix_mode port=$port"
    if ! PORT="$port" \
      KV_CACHE_DTYPE="$kv_dtype" \
      CALCULATE_KV_SCALES="${CALCULATE_KV_SCALES:-1}" \
      PREFIX_CACHING="$prefix_mode" \
      FLASHINFER_SAMPLER="${FLASHINFER_SAMPLER:-0}" \
      "$PWD/scripts/run_vllm_l20_kv_pressure_campaign.sh" "$model" "$served_name" "$run_dir"; then
      echo "[l20-kv-pressure] run failed; keeping structured report in $run_dir" >&2
    fi
  done
done

"$python_bin" "$PWD/scripts/summarize_kv_pressure.py" \
  "${run_dirs[@]}" \
  --output "$output_dir/kv-pressure-summary.json"
