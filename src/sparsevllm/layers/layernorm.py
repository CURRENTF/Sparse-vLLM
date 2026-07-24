from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from typing import Callable

import torch
from torch import nn


RMSNormFn = Callable[[torch.Tensor, torch.Tensor, float], torch.Tensor]
FusedAddRMSNormFn = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor, float],
    None,
]


@dataclass(frozen=True)
class _RMSNormOps:
    provider_name: str
    rmsnorm: RMSNormFn
    fused_add_rmsnorm: FusedAddRMSNormFn


def _load_flashinfer_ops(*, zero_centered_weight: bool) -> _RMSNormOps:
    module = import_module("flashinfer.norm")
    if zero_centered_weight:
        rmsnorm = getattr(module, "gemma_rmsnorm")
        fused_add_rmsnorm = getattr(module, "gemma_fused_add_rmsnorm")
    else:
        rmsnorm = getattr(module, "rmsnorm")
        fused_add_rmsnorm = getattr(module, "fused_add_rmsnorm")

    def run_rmsnorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return rmsnorm(x, weight, eps=eps)

    def run_fused_add_rmsnorm(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> None:
        fused_add_rmsnorm(x, residual, weight, eps=eps)

    return _RMSNormOps(
        provider_name="flashinfer",
        rmsnorm=run_rmsnorm,
        fused_add_rmsnorm=run_fused_add_rmsnorm,
    )


def _load_triton_ops(*, zero_centered_weight: bool) -> _RMSNormOps:
    from sparsevllm.triton_kernel.rmsnorm import (
        fused_add_rmsnorm_forward,
        rmsnorm_forward,
    )

    def run_rmsnorm(
        x: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        return rmsnorm_forward(
            x,
            weight,
            eps,
            zero_centered_weight=zero_centered_weight,
        )

    def run_fused_add_rmsnorm(
        x: torch.Tensor,
        residual: torch.Tensor,
        weight: torch.Tensor,
        eps: float,
    ) -> None:
        fused_add_rmsnorm_forward(
            x,
            residual,
            weight,
            eps,
            zero_centered_weight=zero_centered_weight,
        )

    return _RMSNormOps(
        provider_name="triton",
        rmsnorm=run_rmsnorm,
        fused_add_rmsnorm=run_fused_add_rmsnorm,
    )


@lru_cache(maxsize=2)
def _resolve_rmsnorm_ops(*, zero_centered_weight: bool) -> _RMSNormOps:
    if find_spec("flashinfer") is not None:
        return _load_flashinfer_ops(
            zero_centered_weight=zero_centered_weight,
        )
    return _load_triton_ops(zero_centered_weight=zero_centered_weight)


class RMSNorm(nn.Module):
    """CUDA RMSNorm with FlashInfer preferred over a local Triton baseline."""

    zero_centered_weight = False

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self._ops = _resolve_rmsnorm_ops(
            zero_centered_weight=self.zero_centered_weight,
        )

    @property
    def provider_name(self) -> str:
        return self._ops.provider_name

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self._ops.rmsnorm(x, self.weight, self.eps)
        self._ops.fused_add_rmsnorm(x, residual, self.weight, self.eps)
        return x, residual


class GemmaRMSNorm(RMSNorm):
    """RMSNorm with the Hugging Face ``1 + weight`` convention."""

    zero_centered_weight = True

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(hidden_size, eps=eps)
        nn.init.zeros_(self.weight)
