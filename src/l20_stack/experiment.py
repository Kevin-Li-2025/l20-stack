"""Experiment manifest loading and validation."""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

from l20_stack.memory import ModelSpec, TrainingSpec

T = TypeVar("T")


@dataclass(frozen=True)
class ExperimentConfig:
    task: str
    dataset: str
    output_dir: str
    model: ModelSpec
    training: TrainingSpec

    @classmethod
    def from_file(cls, path: str) -> "ExperimentConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "ExperimentConfig":
        required = {"task", "dataset", "output_dir", "model"}
        missing = sorted(required.difference(payload))
        if missing:
            raise ValueError("missing required config keys: " + ", ".join(missing))

        model = _dataclass_from_mapping(ModelSpec, payload["model"])
        training = _dataclass_from_mapping(TrainingSpec, payload.get("training", {}))
        return cls(
            task=str(payload["task"]),
            dataset=str(payload["dataset"]),
            output_dir=str(payload["output_dir"]),
            model=model,
            training=training,
        )


def _dataclass_from_mapping(cls: Type[T], values: Dict[str, Any]) -> T:
    if not isinstance(values, dict):
        raise ValueError(cls.__name__ + " config must be an object")

    field_names = {field.name for field in fields(cls)}
    unknown = sorted(set(values).difference(field_names))
    if unknown:
        raise ValueError("unknown " + cls.__name__ + " keys: " + ", ".join(unknown))
    return cls(**values)
