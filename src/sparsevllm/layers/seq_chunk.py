from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from sparsevllm.utils.context import get_context


def current_mlp_seq_chunk_size() -> int:
    """Return the active prefill MLP seq chunk size, or 0 when disabled."""
    ctx = get_context()
    if not getattr(ctx, "is_prefill", False):
        return 0
    config = getattr(getattr(ctx, "cache_manager", None), "config", None)
    return int(getattr(config, "mlp_seq_chunk_size", 0) or 0)


def apply_seq_chunked(
    x: torch.Tensor,
    chunk_size: int,
    fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Apply a token-wise function in chunks along the sequence dimension."""
    chunk_size = int(chunk_size or 0)
    if chunk_size <= 0:
        return fn(x)

    seq_dim = -2 if x.dim() >= 3 else 0
    if x.shape[seq_dim] <= chunk_size:
        return fn(x)

    return torch.cat([fn(chunk) for chunk in x.split(chunk_size, dim=seq_dim)], dim=seq_dim)


class SeqChunkedModule(nn.Module):
    """Wrap token-wise MLP modules to reduce prefill activation memory."""

    def __init__(self, module: nn.Module, chunk_size: int):
        super().__init__()
        self.module = module
        self.chunk_size = int(chunk_size or 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return apply_seq_chunked(x, current_mlp_seq_chunk_size(), self.module)
