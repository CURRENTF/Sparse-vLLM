import torch
from torch import nn
import torch.nn.functional as F


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In-place activation to reduce peak memory:
        # x is typically (num_tokens, 2 * intermediate_size) in prefill; allocating an extra
        # (num_tokens, intermediate_size) output can be multiple GiB at long-context, large-batch.
        x, y = x.chunk(2, -1)
        F.silu(x, inplace=True)
        x.mul_(y)
        return x
