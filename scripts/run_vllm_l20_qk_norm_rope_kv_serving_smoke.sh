#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  cat >&2 <<'EOF'
usage: scripts/run_vllm_l20_qk_norm_rope_kv_serving_smoke.sh \
  MODEL SERVED_NAME OUTPUT_DIR VLLM_SOURCE_DIR

Runs a paired vLLM O2 serving smoke/matrix with the L20 Q/K norm + RoPE +
KV-cache custom path off/on.  This is different from vLLM's native
enable_qk_norm_rope_fusion: the native pass is forced off for both variants.
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
  (
    cd "$repo_root"
    export PORT="$port"
    export VLLM_L20_QK_ROPE_KV="$enabled"
    export VLLM_L20_QK_ROPE_KV_STRICT="$enabled"
    export VLLM_L20_QK_ROPE_KV_TRACE="$output_dir/$name/qk-kv-trace.txt"
    export VLLM_L20_QK_ROPE_KV_TRACE_LIMIT="${VLLM_L20_QK_ROPE_KV_TRACE_LIMIT:-256}"
    export COMPILATION_CONFIG="${COMPILATION_CONFIG:-{\"mode\":3,\"splitting_ops\":[],\"cudagraph_mode\":\"FULL\",\"pass_config\":{\"enable_qk_norm_rope_fusion\":false,\"fuse_rope_kvcache\":false}}}"
    scripts/run_vllm_l20_paged_decode_rfc_campaign.sh \
      "$model" \
      "$served_name" \
      o2 \
      0 \
      "$output_dir/$name" \
      "$vllm_source_dir"
  )
}

run_variant qk-kv-off 0 "$base_port"
run_variant qk-kv-on 1 "$((base_port + 1))"

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
pattern = re.compile(r"c(?P<concurrency>\d+)-i(?P<input>\d+)-r(?P<run>\d+)\.json")


def load_reports(name: str) -> list[tuple[Path, dict]]:
    paths = sorted(path for path in (root / name).glob("*.json") if pattern.fullmatch(path.name))
    if not paths:
        raise SystemExit(f"no serving reports found under {root / name}")
    reports = [(path, json.loads(path.read_text(encoding="utf-8"))) for path in paths]
    for path, report in reports:
        if report.get("failed") not in (0, None):
            raise SystemExit(f"serving report has failures: {path}")
    return reports


def summarize_reports(name: str) -> dict:
    reports = load_reports(name)
    return {
        metric: statistics.median(report[metric] for _, report in reports)
        for metric in metrics
    } | {"report_count": len(reports)}


def shape_groups(name: str) -> dict[tuple[int, int], list[dict]]:
    groups: dict[tuple[int, int], list[dict]] = {}
    for path, report in load_reports(name):
        match = pattern.fullmatch(path.name)
        assert match is not None
        key = (int(match.group("concurrency")), int(match.group("input")))
        groups.setdefault(key, []).append(report)
    return groups


def median_metrics(reports: list[dict]) -> dict:
    return {
        metric: statistics.median(report[metric] for report in reports)
        for metric in metrics
    } | {"report_count": len(reports)}


def shape_summaries() -> list[dict]:
    off = shape_groups("qk-kv-off")
    on = shape_groups("qk-kv-on")
    if off.keys() != on.keys():
        raise SystemExit(
            f"shape mismatch: qk-kv-off={sorted(off)} qk-kv-on={sorted(on)}"
        )
    summaries = []
    for concurrency, input_tokens in sorted(off):
        baseline = median_metrics(off[(concurrency, input_tokens)])
        fused = median_metrics(on[(concurrency, input_tokens)])
        changes = {
            metric: round((fused[metric] / baseline[metric] - 1.0) * 100.0, 3)
            for metric in metrics
            if baseline[metric] != 0
        }
        summaries.append(
            {
                "max_concurrency": concurrency,
                "input_tokens": input_tokens,
                "qk_kv_off": baseline,
                "qk_kv_on": fused,
                "changes_pct": changes,
            }
        )
    return summaries


def log_evidence(name: str) -> dict:
    path = root / name / "server.log"
    trace = root / name / "qk-kv-trace.txt"
    text = path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""
    trace_text = trace.read_text(encoding="utf-8", errors="replace") if trace.exists() else ""
    return {
        "server_log_exists": path.exists(),
        "trace_exists": trace.exists(),
        "trace_hit_count": trace_text.count("hit\t"),
        "trace_fallback_count": trace_text.count("fallback\t"),
        "native_qk_fusion_false": "'enable_qk_norm_rope_fusion': False" in text
        or '"enable_qk_norm_rope_fusion": false' in text,
        "custom_env_seen": "VLLM_L20_QK_ROPE_KV" in text,
        "strict_env_seen": "VLLM_L20_QK_ROPE_KV_STRICT" in text,
        "flashinfer_backend": "AttentionBackendEnum.FLASHINFER" in text,
        "flashinfer_sampling": "Using FlashInfer for top-p & top-k sampling" in text,
        "full_decode_only": "FULL_DECODE_ONLY" in text,
        "cudagraph_disabled": "Cudagraph is disabled" in text,
        "torch_compile_mentions": text.count("torch.compile"),
    }


rows = {
    "qk-kv-off": summarize_reports("qk-kv-off"),
    "qk-kv-on": summarize_reports("qk-kv-on"),
}
changes = {}
for metric in metrics:
    baseline = rows["qk-kv-off"][metric]
    fused = rows["qk-kv-on"][metric]
    if baseline != 0:
        changes[metric] = round((fused / baseline - 1.0) * 100.0, 3)

result = {
    "schema_version": 1,
    "summary": (
        "vLLM O2 serving matrix comparing the L20 Q/K norm + RoPE + "
        "KV-cache custom path off vs on. vLLM native QK fusion is disabled."
    ),
    "rows": rows,
    "changes_pct": changes,
    "shapes": shape_summaries(),
    "log_evidence": {
        "qk-kv-off": log_evidence("qk-kv-off"),
        "qk-kv-on": log_evidence("qk-kv-on"),
    },
}
serialized = json.dumps(result, indent=2, sort_keys=True)
print(serialized)
(root / "qk-rope-kv-serving-summary.json").write_text(
    serialized + "\n", encoding="utf-8"
)
PY
