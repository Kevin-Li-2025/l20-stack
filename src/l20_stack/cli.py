"""Command line interface for L20 Stack."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from l20_stack.experiment import ExperimentConfig
from l20_stack.memory import estimate_training_memory


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l20-stack")
    subparsers = parser.add_subparsers(dest="command", required=True)

    plan = subparsers.add_parser("plan", help="estimate memory for an experiment config")
    plan.add_argument("--config", required=True, help="path to a JSON experiment config")

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "plan":
        config = ExperimentConfig.from_file(args.config)
        estimate = estimate_training_memory(config.model, config.training)
        print(
            json.dumps(
                {
                    "task": config.task,
                    "dataset": config.dataset,
                    "output_dir": config.output_dir,
                    "estimate": estimate.to_dict(),
                    "note": (
                        "Planning estimate only; validate with real CUDA telemetry "
                        "before making performance claims."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    parser.error("unknown command: " + str(args.command))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
