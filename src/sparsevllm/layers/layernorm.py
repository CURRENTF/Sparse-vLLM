import torch
from torch import nn


class RMSNorm(nn.Module):
    """FlashInfer RMSNorm for CUDA inference."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @staticmethod
    def _load_flashinfer_ops():
        try:
            from flashinfer.norm import fused_add_rmsnorm, rmsnorm
        except ImportError as exc:
            raise ImportError(
                "RMSNorm requires flashinfer-python and the JIT cache matching "
                "torch.version.cuda."
            ) from exc
        return rmsnorm, fused_add_rmsnorm

    def _validate_input(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> None:
        if not x.is_cuda:
            raise RuntimeError("FlashInfer RMSNorm requires a CUDA input tensor.")
        if x.dtype not in {torch.float16, torch.bfloat16}:
            raise TypeError(
                "FlashInfer RMSNorm requires FP16 or BF16 input, "
                f"got {x.dtype}."
            )
        if x.shape[-1] != self.weight.numel():
            raise ValueError(
                "RMSNorm input and weight hidden sizes differ: "
                f"{x.shape[-1]} and {self.weight.numel()}."
            )
        if residual is not None and (
            residual.shape != x.shape or residual.dtype != x.dtype
        ):
            raise ValueError(
                "Fused add-RMSNorm requires input and residual with matching "
                f"shape and dtype, got {tuple(x.shape)}/{x.dtype} and "
                f"{tuple(residual.shape)}/{residual.dtype}."
            )

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        self._validate_input(x, residual)
        rmsnorm, fused_add_rmsnorm = self._load_flashinfer_ops()
        if residual is None:
            return rmsnorm(x, self.weight, eps=self.eps)
        fused_add_rmsnorm(x, residual, self.weight, eps=self.eps)
        return x, residual


class GemmaRMSNorm(RMSNorm):
    """FlashInfer RMSNorm with the Hugging Face ``1 + weight`` convention."""

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__(hidden_size, eps=eps)
        nn.init.zeros_(self.weight)

    @staticmethod
    def _load_flashinfer_ops():
        try:
            from flashinfer.norm import gemma_fused_add_rmsnorm, gemma_rmsnorm
        except ImportError as exc:
            raise ImportError(
                "GemmaRMSNorm requires flashinfer-python and the JIT cache "
                "matching torch.version.cuda."
            ) from exc
        return gemma_rmsnorm, gemma_fused_add_rmsnorm
