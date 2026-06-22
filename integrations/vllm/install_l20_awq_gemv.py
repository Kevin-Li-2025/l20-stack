#!/usr/bin/env python3
"""Install the experimental L20 AWQ decode GEMV into vLLM."""

from __future__ import annotations

import argparse
import inspect
import shutil
from pathlib import Path

import vllm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uninstall", action="store_true")
    args = parser.parse_args()
    package = Path(inspect.getfile(vllm)).parent
    awq = package / "model_executor" / "layers" / "quantization" / "awq.py"
    kernel = package / "model_executor" / "layers" / "quantization" / "l20_awq_gemv.py"
    backup = awq.with_suffix(".py.l20-backup")
    if args.uninstall:
        if backup.exists():
            shutil.copy2(backup, awq)
        kernel.unlink(missing_ok=True)
        return 0
    if not backup.exists():
        shutil.copy2(awq, backup)
    shutil.copy2(Path(__file__).with_name("l20_awq_gemv.py"), kernel)
    source = awq.read_text(encoding="utf-8")
    import_line = (
        "from vllm.model_executor.layers.quantization.l20_awq_gemv import "
        "l20_awq_gemv, should_use_l20_awq_gemv\n"
    )
    marker = "from vllm.model_executor.layers.quantization.base_config import (\n"
    if import_line not in source:
        source = source.replace(marker, import_line + marker, 1)
    old = """        # num_tokens >= threshold
        FP16_MATMUL_HEURISTIC_CONDITION = x.shape[:-1].numel() >= 256
        # Batch invariant mode requires torch.matmul path
        # for Triton override
        if FP16_MATMUL_HEURISTIC_CONDITION or envs.VLLM_BATCH_INVARIANT:
"""
    new = """        if should_use_l20_awq_gemv(reshaped_x, self.quant_config.group_size):
            out = l20_awq_gemv(
                reshaped_x, qweight, scales, qzeros, self.quant_config.group_size
            )
        # num_tokens >= threshold
        elif x.shape[:-1].numel() >= 256 or envs.VLLM_BATCH_INVARIANT:
"""
    if new not in source:
        if old not in source:
            raise RuntimeError("cannot find AWQ dispatch patch point")
        source = source.replace(old, new, 1)
    awq.write_text(source, encoding="utf-8")
    print(package)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
