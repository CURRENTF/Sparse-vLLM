from __future__ import annotations

from typing import Protocol

import torch
import torch.distributed as dist

from sparsevllm.utils.log import logger


class AllReduceProvider(Protocol):
    name: str

    def run(self, tensor: torch.Tensor) -> torch.Tensor: ...


class TorchDistributedAllReduceProvider:
    name = "torch_distributed"

    def __init__(self, group: dist.ProcessGroup | None) -> None:
        self.group = group

    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        dist.all_reduce(tensor, group=self.group)
        return tensor


class HopperTp2FlashInferAllReduceProvider:
    name = "hopper_tp2_flashinfer"
    hidden_size = 2048
    max_rows = 256

    def __init__(self, group: dist.ProcessGroup) -> None:
        from flashinfer import comm

        self.comm = comm
        self.fallback = TorchDistributedAllReduceProvider(group)
        self.workspace = comm.create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=2,
            rank=dist.get_rank(group),
            max_token_num=self.max_rows,
            hidden_dim=self.hidden_size,
            dtype=torch.bfloat16,
            group=group,
        )

    def _supports(self, tensor: torch.Tensor) -> bool:
        return (
            tensor.is_cuda
            and tensor.dtype == torch.bfloat16
            and tensor.is_contiguous()
            and tensor.ndim >= 2
            and tensor.shape[-1] == self.hidden_size
            and tensor.numel() <= self.max_rows * self.hidden_size
        )

    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        if not self._supports(tensor):
            return self.fallback.run(tensor)
        output = self.comm.allreduce_fusion(
            input=tensor.view(-1, self.hidden_size),
            workspace=self.workspace,
            pattern=self.comm.AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=True,
            trigger_completion_at_end=tensor.numel() > 16 * self.hidden_size,
        )
        return output.view_as(tensor)


def resolve_all_reduce_provider(
    group: dist.ProcessGroup | None,
    world_size: int,
) -> AllReduceProvider:
    if (
        world_size == 2
        and group is not None
        and dist.get_backend(group) == dist.Backend.NCCL
        and torch.cuda.get_device_capability() == (9, 0)
    ):
        provider = HopperTp2FlashInferAllReduceProvider(group)
        logger.info("AllReduce provider: %s", provider.name)
        return provider
    return TorchDistributedAllReduceProvider(group)
