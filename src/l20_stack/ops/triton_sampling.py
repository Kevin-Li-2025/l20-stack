"""L20-oriented GPU-side decode sampling primitives."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Optional

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None


@dataclass(frozen=True)
class SamplingLaunchConfig:
    block_vocab: int
    blocks_per_row: int
    num_warps: int
    num_stages: int
    strategy: str

    def to_dict(self):
        return asdict(self)


def next_power_of_2(value: int) -> int:
    if value <= 0:
        raise ValueError("value must be positive")
    return 1 << (value - 1).bit_length()


def greedy_sampling_launch_config(
    vocab_size: int,
    *,
    block_vocab_override: Optional[int] = None,
) -> SamplingLaunchConfig:
    """Return the L20 launch policy for greedy sampling.

    The target is decode serving, where transferring `[batch, vocab]` logits to
    CPU just to choose one token is usually worse than a small GPU reduction. A
    single CTA is too serial for Qwen-sized vocabularies, so large vocabularies
    use a two-stage block reduction.
    """

    if vocab_size > 262_144:
        raise ValueError("vocab_size above 262144 requires a multi-stage sampling path")
    if vocab_size > 65_536:
        block_vocab = block_vocab_override or 1024
        if block_vocab not in {512, 1024, 2048, 4096, 8192}:
            raise ValueError("block_vocab_override must be one of 512, 1024, 2048, 4096, 8192")
        blocks_per_row = (vocab_size + block_vocab - 1) // block_vocab
        num_warps = 8 if block_vocab >= 4096 else 4
        strategy = "two_stage_block_argmax"
    else:
        if block_vocab_override is not None:
            raise ValueError("block_vocab_override is only supported for large vocabularies")
        block_vocab = next_power_of_2(vocab_size)
        blocks_per_row = 1
        num_warps = 4 if block_vocab >= 32_768 else 2
        strategy = "single_cta_argmax"
    return SamplingLaunchConfig(
        block_vocab=block_vocab,
        blocks_per_row=blocks_per_row,
        num_warps=num_warps,
        num_stages=1,
        strategy=strategy,
    )


def should_use_l20_gpu_greedy_sampling(batch: int, vocab_size: int, top_k: int = 1) -> bool:
    """Conservative L20 gate for the first GPU-side sampler path."""

    if top_k != 1:
        return False
    if batch <= 0 or vocab_size <= 0:
        return False
    return batch <= 64 and vocab_size <= 262_144


if triton is not None:  # pragma: no cover - requires CUDA

    @triton.jit
    def _greedy_sample_kernel(
        logits,
        output,
        BATCH: tl.constexpr,
        VOCAB: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        TEMPERATURE: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, BLOCK_VOCAB)
        mask = offsets < VOCAB
        values = tl.load(logits + row * VOCAB + offsets, mask=mask, other=-float("inf"))
        if TEMPERATURE != 1.0:
            values = values / TEMPERATURE
        max_value = tl.max(values, axis=0)
        is_max = values == max_value
        # Tie-break with the smallest token id to match torch.argmax semantics.
        token_values = tl.where(is_max, offsets, BLOCK_VOCAB)
        token = tl.min(token_values, axis=0)
        tl.store(output + row, token)

    @triton.jit
    def _greedy_sample_partial_kernel(
        logits,
        partial_values,
        partial_tokens,
        VOCAB: tl.constexpr,
        BLOCK_VOCAB: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        TEMPERATURE: tl.constexpr,
    ):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offsets = block * BLOCK_VOCAB + tl.arange(0, BLOCK_VOCAB)
        mask = offsets < VOCAB
        values = tl.load(logits + row * VOCAB + offsets, mask=mask, other=-float("inf"))
        if TEMPERATURE != 1.0:
            values = values / TEMPERATURE
        max_value = tl.max(values, axis=0)
        is_max = values == max_value
        token_values = tl.where(is_max, offsets, VOCAB)
        token = tl.min(token_values, axis=0)
        out_offset = row * BLOCKS_PER_ROW + block
        tl.store(partial_values + out_offset, max_value)
        tl.store(partial_tokens + out_offset, token)

    @triton.jit
    def _greedy_sample_reduce_kernel(
        partial_values,
        partial_tokens,
        output,
        VOCAB: tl.constexpr,
        BLOCKS_PER_ROW: tl.constexpr,
        REDUCE_BLOCK: tl.constexpr,
    ):
        row = tl.program_id(0)
        offsets = tl.arange(0, REDUCE_BLOCK)
        mask = offsets < BLOCKS_PER_ROW
        values = tl.load(
            partial_values + row * BLOCKS_PER_ROW + offsets,
            mask=mask,
            other=-float("inf"),
        )
        tokens = tl.load(
            partial_tokens + row * BLOCKS_PER_ROW + offsets,
            mask=mask,
            other=VOCAB,
        )
        max_value = tl.max(values, axis=0)
        is_max = values == max_value
        token_values = tl.where(is_max, tokens, VOCAB)
        token = tl.min(token_values, axis=0)
        tl.store(output + row, token)


def greedy_sample(logits, temperature: float = 1.0):
    """Sample greedily on GPU without materializing logits on CPU.

    This implements the deterministic `top_k=1` serving case. Temperature is
    accepted to preserve the sampler contract; for greedy argmax, positive
    scalar temperature does not change the selected token.
    """

    if torch is None or triton is None:
        raise RuntimeError("greedy_sample requires PyTorch and Triton")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if not logits.is_cuda:
        raise ValueError("logits must be a CUDA tensor")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, vocab = logits.shape
    if not should_use_l20_gpu_greedy_sampling(int(batch), int(vocab), top_k=1):
        raise ValueError("shape is outside the L20 greedy sampling gate")
    config = greedy_sampling_launch_config(int(vocab))
    output = torch.empty((batch,), device=logits.device, dtype=torch.int64)
    if config.strategy == "two_stage_block_argmax":
        partial_values = torch.empty(
            (batch, config.blocks_per_row),
            device=logits.device,
            dtype=torch.float32,
        )
        partial_tokens = torch.empty(
            (batch, config.blocks_per_row),
            device=logits.device,
            dtype=torch.int64,
        )
        greedy_sample_out(
            logits,
            output,
            partial_values=partial_values,
            partial_tokens=partial_tokens,
            temperature=temperature,
        )
        return output
    greedy_sample_out(logits, output, temperature=temperature)
    return output


def greedy_sample_out(
    logits,
    output,
    *,
    partial_values=None,
    partial_tokens=None,
    temperature: float = 1.0,
    block_vocab_override: Optional[int] = None,
):
    """Write greedy samples into a caller-owned output tensor.

    vLLM-style serving loops can keep `output`, `partial_values`, and
    `partial_tokens` live across decode steps, avoiding allocator noise in the
    hot path. Large-vocab shapes require both partial workspaces.
    """

    if torch is None or triton is None:
        raise RuntimeError("greedy_sample_out requires PyTorch and Triton")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if output.shape != (logits.shape[0],) or output.dtype != torch.int64:
        raise ValueError("output must have shape [batch] and dtype int64")
    if not logits.is_cuda or not output.is_cuda:
        raise ValueError("logits and output must be CUDA tensors")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    batch, vocab = logits.shape
    if not should_use_l20_gpu_greedy_sampling(int(batch), int(vocab), top_k=1):
        raise ValueError("shape is outside the L20 greedy sampling gate")
    config = greedy_sampling_launch_config(
        int(vocab),
        block_vocab_override=block_vocab_override,
    )
    if config.strategy == "two_stage_block_argmax":
        expected_workspace = (batch, config.blocks_per_row)
        if partial_values is None or partial_values.shape != expected_workspace:
            raise ValueError("partial_values workspace has the wrong shape")
        if partial_tokens is None or partial_tokens.shape != expected_workspace:
            raise ValueError("partial_tokens workspace has the wrong shape")
        if partial_values.dtype != torch.float32 or partial_tokens.dtype != torch.int64:
            raise ValueError("partial workspaces must be float32 and int64")
        if not partial_values.is_cuda or not partial_tokens.is_cuda:
            raise ValueError("partial workspaces must be CUDA tensors")
        _greedy_sample_partial_kernel[(batch, config.blocks_per_row)](
            logits,
            partial_values,
            partial_tokens,
            VOCAB=int(vocab),
            BLOCK_VOCAB=config.block_vocab,
            BLOCKS_PER_ROW=config.blocks_per_row,
            TEMPERATURE=float(temperature),
            num_warps=config.num_warps,
            num_stages=config.num_stages,
        )
        _greedy_sample_reduce_kernel[(batch,)](
            partial_values,
            partial_tokens,
            output,
            VOCAB=int(vocab),
            BLOCKS_PER_ROW=config.blocks_per_row,
            REDUCE_BLOCK=next_power_of_2(config.blocks_per_row),
            num_warps=1,
            num_stages=1,
        )
        return None
    _greedy_sample_kernel[(batch,)](
        logits,
        output,
        BATCH=int(batch),
        VOCAB=int(vocab),
        BLOCK_VOCAB=config.block_vocab,
        TEMPERATURE=float(temperature),
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return None


def greedy_sample_reference(logits, temperature: float = 1.0):
    if torch is None:
        raise RuntimeError("greedy_sample_reference requires PyTorch")
    if logits.ndim != 2:
        raise ValueError("expected logits with shape [batch, vocab]")
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    return torch.argmax(logits / temperature, dim=-1).to(torch.int64)
