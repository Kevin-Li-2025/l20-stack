#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  scripts/profile_kernel.sh --output OUT_PREFIX --kernel-name REGEX -- COMMAND [ARGS...]

Example:
  scripts/profile_kernel.sh \
    --output benchmarks/results/ncu/rope-kv/tokens-1024 \
    --kernel-name 'regex:_l20_.*rope_kv_kernel' \
    -- env PYTHONPATH=src python scripts/profile_vllm_l20_rope_kv.py --execute-tokens 1024

Outputs:
  OUT_PREFIX.ncu-rep   Nsight Compute report
  OUT_PREFIX.csv       Nsight raw CSV
  OUT_PREFIX.json      Parsed roofline/profile summary
  OUT_PREFIX.md        Markdown dashboard fragment
EOF
}

output=""
kernel_name="regex:.*"
launch_skip="${NCU_LAUNCH_SKIP:-5}"
launch_count="${NCU_LAUNCH_COUNT:-1}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --output)
      output="${2:?missing --output value}"
      shift 2
      ;;
    --kernel-name)
      kernel_name="${2:?missing --kernel-name value}"
      shift 2
      ;;
    --launch-skip)
      launch_skip="${2:?missing --launch-skip value}"
      shift 2
      ;;
    --launch-count)
      launch_count="${2:?missing --launch-count value}"
      shift 2
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

if [[ -z "$output" || $# -eq 0 ]]; then
  usage
  exit 2
fi

find_ncu() {
  if [[ -n "${NCU_BIN:-}" ]]; then
    if command -v "$NCU_BIN" >/dev/null 2>&1; then
      command -v "$NCU_BIN"
    else
      echo "$NCU_BIN"
    fi
    return 0
  fi

  if command -v ncu >/dev/null 2>&1; then
    command -v ncu
    return 0
  fi

  local candidate
  for candidate in \
    /usr/local/cuda/bin/ncu \
    /usr/local/cuda-13.0/bin/ncu \
    /usr/local/cuda-12.9/bin/ncu \
    /usr/local/cuda-12.8/bin/ncu \
    /usr/local/cuda-12.6/bin/ncu \
    /opt/nvidia/nsight-compute/*/ncu \
    /opt/nvidia/nsight-compute/*/target/linux-desktop-glibc_2_11_3-x64/ncu; do
    if [[ -x "$candidate" ]]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

ncu_bin="$(find_ncu || true)"
if [[ -z "$ncu_bin" || ! -x "$ncu_bin" ]]; then
  echo "ncu is required; set NCU_BIN or install NVIDIA Nsight Compute on the L20 host" >&2
  exit 2
fi

mkdir -p "$(dirname "$output")"

metrics=$(
  IFS=,
  echo "${NCU_METRICS:-\
gpu__time_duration.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__warps_active.avg.pct_of_peak_sustained_elapsed,\
dram__throughput.avg.pct_of_peak_sustained_elapsed,\
dram__bytes.sum,\
lts__throughput.avg.pct_of_peak_sustained_elapsed,\
lts__t_sectors_srcunit_tex_op_read.sum,\
lts__t_sectors_srcunit_tex_op_write.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,\
smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,\
smsp__warp_issue_stalled_barrier_per_warp_active.pct,\
smsp__warp_issue_stalled_membar_per_warp_active.pct,\
smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_fmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum}"
)

echo "Using Nsight Compute CLI: $ncu_bin" >&2

"$ncu_bin" \
  --target-processes all \
  --kernel-name "$kernel_name" \
  --launch-skip "$launch_skip" \
  --launch-count "$launch_count" \
  --metrics "$metrics" \
  --section SpeedOfLight \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section WarpStateStats \
  --section LaunchStats \
  --export "$output" \
  "$@"

"$ncu_bin" --import "${output}.ncu-rep" --page raw --csv > "${output}.csv"
summary_python="${PYTHON:-python3}"
"$summary_python" scripts/summarize_ncu_profile.py \
  --csv "${output}.csv" \
  --json-output "${output}.json" \
  --markdown-output "${output}.md"
