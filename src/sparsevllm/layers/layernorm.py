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


class FlashInferRMSNorm(RMSNorm):
    """RMSNorm backed by FlashInfer for CUDA inference."""

    @staticmethod
    def _load_flashinfer_ops():
        try:
            from flashinfer.norm import fused_add_rmsnorm, rmsnorm
        except ImportError as exc:
            raise ImportError(
                "FlashInferRMSNorm requires the minimax_m2 optional dependencies. "
                "Install flashinfer-python and the JIT cache matching torch.version.cuda."
            ) from exc
        return rmsnorm, fused_add_rmsnorm

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if not x.is_cuda:
            if residual is None:
                return self._rms_forward_impl(x)
            return self._add_rms_forward_impl(x, residual)
        if x.dtype not in {torch.float16, torch.bfloat16}:
            raise TypeError(
                "FlashInferRMSNorm requires FP16 or BF16 CUDA input, "
                f"got {x.dtype}."
            )
        if x.shape[-1] != self.weight.numel():
            raise ValueError(
                "FlashInferRMSNorm input and weight hidden sizes differ: "
                f"{x.shape[-1]} and {self.weight.numel()}."
            )

        rmsnorm, fused_add_rmsnorm = self._load_flashinfer_ops()
        if residual is None:
            return rmsnorm(x, self.weight, eps=self.eps)
        if residual.shape != x.shape or residual.dtype != x.dtype:
            raise ValueError(
                "FlashInfer fused add-RMSNorm requires input and residual with "
                f"matching shape and dtype, got {tuple(x.shape)}/{x.dtype} and "
                f"{tuple(residual.shape)}/{residual.dtype}."
            )
        fused_add_rmsnorm(x, residual, self.weight, eps=self.eps)
        return x, residual
