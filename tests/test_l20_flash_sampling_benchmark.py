import importlib.util
import json
import subprocess
import sys
from pathlib import Path


SCRIPT = Path("scripts/benchmark_l20_flash_sampling_boundary.py")


def test_flash_sampling_benchmark_script_is_importable():
    spec = importlib.util.spec_from_file_location("benchmark_l20_flash_sampling_boundary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    args = module.parse_args([])
    assert args.batch == 4
    assert args.sampling_mode == "greedy"


def test_flash_sampling_benchmark_dry_run_outputs_gate_and_status(tmp_path):
    output = tmp_path / "dry-run.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--include-candidate",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": "src"},
    )
    payload = json.loads(completed.stdout)

    assert output.exists()
    assert payload["schema_version"] == 1
    assert payload["gate"]["eligible"] is True
    assert payload["gate"]["policy"]["block_vocab"] == 64
    assert payload["paths"]["full_logits_reference"]["status"] == "dry_run"
    assert payload["paths"]["l20_lm_head_sampling_candidate"]["status"] == "dry_run"


def test_flash_sampling_benchmark_records_fallback_gate(tmp_path):
    output = tmp_path / "fallback.json"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--dry-run",
            "--sampling-mode",
            "gumbel",
            "--batch",
            "5",
            "--top-k",
            "50",
            "--output",
            str(output),
        ],
        check=True,
        text=True,
        capture_output=True,
        env={"PYTHONPATH": "src"},
    )
    payload = json.loads(completed.stdout)

    assert payload["gate"]["eligible"] is False
    assert "batch_gt_4" in payload["gate"]["reasons"]
    assert "top_k_top_p_unsupported" in payload["gate"]["reasons"]
    assert payload["gate"]["policy"] is None
