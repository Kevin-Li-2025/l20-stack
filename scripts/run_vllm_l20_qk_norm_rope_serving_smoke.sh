#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_qk_norm_rope_serving_smoke.sh \
  MODEL SERVED_NAME OUTPUT_DIR VLLM_SOURCE_DIR

Runs a paired vLLM O2 serving smoke with enable_qk_norm_rope_fusion off/on.
EOF
  exit 2
fi

model=$1
served_name=$2
output_dir=$3
vllm_source_dir=$4

repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
python_bin=${PYTHON:-python}
base_port=${PORT:-8000}
mkdir -p "$output_dir"
output_dir=$(cd "$output_dir" && pwd)

run_variant() {
  local name=$1
  local enabled=$2
  local port=$3
  local enabled_json=false
  if [[ "$enabled" == "1" ]]; then
    enabled_json=true
  fi
  (
    cd "$repo_root"
    export PORT="$port"
    export COMPILATION_CONFIG="{\"mode\":3,\"splitting_ops\":[],\"cudagraph_mode\":\"FULL\",\"pass_config\":{\"enable_qk_norm_rope_fusion\":$enabled_json,\"fuse_rope_kvcache\":false}}"
    scripts/run_vllm_l20_paged_decode_rfc_campaign.sh \
      "$model" \
      "$served_name" \
      o2 \
      0 \
      "$output_dir/$name" \
      "$vllm_source_dir"
  )
}

run_variant qk-off 0 "$base_port"
run_variant qk-on 1 "$((base_port + 1))"

"$python_bin" - "$output_dir" <<'PY'
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path

root = Path(sys.argv[1])
metrics = (
    "output_throughput",
    "mean_ttft_ms",
    "median_ttft_ms",
    "p99_ttft_ms",
    "mean_itl_ms",
    "median_itl_ms",
    "p99_itl_ms",
)
pattern = re.compile(r"c\d+-i\d+-r\d+\.json")


def load_reports(name: str) -> list[dict]:
    paths = sorted(path for path in (root / name).glob("*.json") if pattern.fullmatch(path.name))
    if not paths:
        raise SystemExit(f"no serving reports found under {root / name}")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for path, report in zip(paths, reports):
        if report.get("failed") not in (0, None):
            raise SystemExit(f"serving report has failures: {path}")
    return reports


def summarize_reports(name: str) -> dict:
    reports = load_reports(name)
    return {
        metric: statistics.median(report[metric] for report in reports)
        for metric in metrics
    } | {"report_count": len(reports)}


def log_evidence(name: str) -> dict:
    path = root / name / "server.log"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    return {
        "server_log_exists": path.exists(),
        "qk_config_true": "'enable_qk_norm_rope_fusion': True" in text
        or '"enable_qk_norm_rope_fusion": true' in text,
        "qk_config_false": "'enable_qk_norm_rope_fusion': False" in text
        or '"enable_qk_norm_rope_fusion": false' in text,
        "flashinfer_backend": "AttentionBackendEnum.FLASHINFER" in text,
        "flashinfer_sampling": "Using FlashInfer for top-p & top-k sampling" in text,
        "cudagraph_disabled": "Cudagraph is disabled" in text,
        "full_decode_only": "FULL_DECODE_ONLY" in text,
        "torch_compile_mentions": text.count("torch.compile"),
    }


rows = {"qk-off": summarize_reports("qk-off"), "qk-on": summarize_reports("qk-on")}
changes = {}
for metric in metrics:
    baseline = rows["qk-off"][metric]
    fused = rows["qk-on"][metric]
    changes[metric] = round((fused / baseline - 1.0) * 100.0, 3)

result = {
    "schema_version": 1,
    "summary": "vLLM O2 serving smoke comparing enable_qk_norm_rope_fusion off vs on.",
    "rows": rows,
    "changes_pct": changes,
    "log_evidence": {
        "qk-off": log_evidence("qk-off"),
        "qk-on": log_evidence("qk-on"),
    },
}
serialized = json.dumps(result, indent=2, sort_keys=True)
print(serialized)
(root / "qk-serving-summary.json").write_text(serialized + "\n", encoding="utf-8")
PY
