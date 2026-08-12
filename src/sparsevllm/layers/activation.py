import torch
from torch import nn

from sparsevllm.operators.activation import (
    GeluTanhAndMulProvider,
    SiluAndMulProvider,
    TorchGeluTanhAndMulProvider,
    TorchSiluAndMulProvider,
)


class SiluAndMul(nn.Module):

    def __init__(self, provider: SiluAndMulProvider | None = None):
        super().__init__()
        self.provider = provider if provider is not None else TorchSiluAndMulProvider()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.provider(x)


class GeluTanhAndMul(nn.Module):
    def __init__(self, provider: GeluTanhAndMulProvider | None = None):
        super().__init__()
        self.provider = (
            provider if provider is not None else TorchGeluTanhAndMulProvider()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.provider(x)
