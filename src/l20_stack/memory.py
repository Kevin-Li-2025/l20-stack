"""Memory planning utilities for single-GPU LLM experiments.

The estimates here are intentionally conservative planning numbers, not a
replacement for measuring `torch.cuda.max_memory_allocated()` during real runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict

GIB = 1024**3


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params_b: float
    hidden_size: int
    layers: int
    target_modules_per_layer: int = 4


@dataclass(frozen=True)
class TrainingSpec:
    seq_len: int = 4096
    micro_batch_size: int = 1
    grad_accum_steps: int = 16
    lora_rank: int = 64
    quant_bits: int = 4
    activation_checkpointing: bool = True
    activation_bytes: int = 2
    activation_multiplier: float = 3.0
    optimizer_bytes_per_param: int = 10
    device_memory_gib: float = 48.0
    reserved_memory_gib: float = 4.0
    safety_margin_gib: float = 2.0


@dataclass(frozen=True)
class MemoryEstimate:
    model_name: str
    quantized_weight_gib: float
    lora_trainable_params_m: float
    adapter_optimizer_gib: float
    activation_gib: float
    reserved_gib: float
    total_gib: float
    available_after_margin_gib: float
    fits_device: bool

    def to_dict(self) -> Dict[str, object]:
        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, float):
                payload[key] = round(value, 3)
        return payload


def quantization_overhead(bits: int) -> float:
    """Return a rough storage overhead multiplier for common quantized formats."""

    if bits <= 0:
        raise ValueError("quant_bits must be positive")
    if bits <= 4:
        return 1.18
    if bits <= 8:
        return 1.08
    return 1.0


def estimate_lora_trainable_params(model: ModelSpec, training: TrainingSpec) -> int:
    """Estimate LoRA adapter parameters for square projection matrices."""

    if training.lora_rank <= 0:
        raise ValueError("lora_rank must be positive")
    if model.hidden_size <= 0 or model.layers <= 0:
        raise ValueError("model hidden_size and layers must be positive")

    params_per_projection = 2 * model.hidden_size * training.lora_rank
    return model.layers * model.target_modules_per_layer * params_per_projection


def estimate_training_memory(model: ModelSpec, training: TrainingSpec) -> MemoryEstimate:
    """Estimate GPU memory needed for a LoRA/QLoRA-style training plan."""

    if model.params_b <= 0:
        raise ValueError("params_b must be positive")
    if training.seq_len <= 0 or training.micro_batch_size <= 0:
        raise ValueError("seq_len and micro_batch_size must be positive")

    weight_bytes = model.params_b * 1_000_000_000 * (training.quant_bits / 8)
    quantized_weight_gib = weight_bytes * quantization_overhead(training.quant_bits) / GIB

    lora_params = estimate_lora_trainable_params(model, training)
    adapter_optimizer_gib = (
        lora_params * (training.activation_bytes + training.optimizer_bytes_per_param) / GIB
    )

    checkpoint_factor = 0.35 if training.activation_checkpointing else 1.0
    activation_bytes = (
        training.micro_batch_size
        * training.seq_len
        * model.hidden_size
        * model.layers
        * training.activation_bytes
        * training.activation_multiplier
        * checkpoint_factor
    )
    activation_gib = activation_bytes / GIB

    total_gib = (
        quantized_weight_gib
        + adapter_optimizer_gib
        + activation_gib
        + training.reserved_memory_gib
    )
    available_after_margin = training.device_memory_gib - training.safety_margin_gib

    return MemoryEstimate(
        model_name=model.name,
        quantized_weight_gib=quantized_weight_gib,
        lora_trainable_params_m=lora_params / 1_000_000,
        adapter_optimizer_gib=adapter_optimizer_gib,
        activation_gib=activation_gib,
        reserved_gib=training.reserved_memory_gib,
        total_gib=total_gib,
        available_after_margin_gib=available_after_margin,
        fits_device=total_gib <= available_after_margin,
    )
