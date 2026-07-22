import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _rms_forward_impl(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x_float = x.float()
        var = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(var + self.eps)
        return (x_norm * self.weight.float()).to(orig_dtype)

    @torch.compile(dynamic=True)
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self._rms_forward_impl(x)

    def _add_rms_forward_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x_float = x.float() + residual.float()
        residual = x_float.to(orig_dtype)
        var = x_float.pow(2).mean(dim=-1, keepdim=True)
        x_norm = x_float * torch.rsqrt(var + self.eps)
        return (x_norm * self.weight.float()).to(orig_dtype), residual

    @torch.compile(dynamic=True)
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._add_rms_forward_impl(x, residual)

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        return self.add_rms_forward(x, residual)
