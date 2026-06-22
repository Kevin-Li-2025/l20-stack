"""Dispatcher-backed entry point for the L20 SM89 paged-decode operator."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import torch


def load_library(path: Optional[Union[str, Path]] = None) -> None:
    library = (
        Path(path)
        if path is not None
        else Path(__file__).with_name("l20_paged_decode_ops.so")
    )
    torch.ops.load_library(str(library))


def paged_decode_split_out(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_lens: torch.Tensor,
    partial_output: torch.Tensor,
    partial_max: torch.Tensor,
    partial_sum: torch.Tensor,
    output: torch.Tensor,
    max_seq_len: int,
    split_size: int,
) -> torch.Tensor:
    return torch.ops.l20_stack.paged_decode_split_out(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partial_output,
        partial_max,
        partial_sum,
        output,
        max_seq_len,
        split_size,
    )


def _register_fake() -> None:
    @torch.library.register_fake("l20_stack::paged_decode_split_out")
    def _paged_decode_split_out_fake(
        query,
        key_cache,
        value_cache,
        block_table,
        seq_lens,
        partial_output,
        partial_max,
        partial_sum,
        output,
        max_seq_len,
        split_size,
    ):
        return output


load_library()
_register_fake()
