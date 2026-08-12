from __future__ import annotations

import torch
from torch import nn


def _torch_rmsnorm(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    output = x.float()
    output *= torch.pow(output.square().mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        output *= weight.float()
    return output.to(x.dtype)


class Gemma4RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6, *, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = float(eps)
        if with_scale:
            self.weight = nn.Parameter(torch.ones(hidden_size))
        else:
            self.register_parameter("weight", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not x.is_cuda:
            return _torch_rmsnorm(x, self.weight, self.eps)
        from sparsevllm.kernels.triton.gemma4_rmsnorm import gemma4_rmsnorm

        return gemma4_rmsnorm(x, self.weight, self.eps)


__all__ = ["Gemma4RMSNorm"]
