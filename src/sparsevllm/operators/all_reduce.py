from __future__ import annotations

import re
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from importlib.util import find_spec

import torch
import torch.distributed as dist

import sparsevllm.platforms as platforms
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
    runtime_version_at_least,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class AllReduceOpSpec:
    world_size: int
    max_tokens: int
    hidden_size: int
    dtype: torch.dtype
    cuda_graph: bool

    def __post_init__(self) -> None:
        if self.world_size <= 0 or self.max_tokens <= 0 or self.hidden_size <= 0:
            raise ValueError("All-reduce dimensions must be positive.")


class AllReduceProvider:
    name = ""
    priority = 0

    def prepare(
        self,
        spec: AllReduceOpSpec,
        *,
        group: dist.ProcessGroup,
        rank: int,
    ) -> None:
        del spec, group, rank

    def can_run(self, spec: AllReduceOpSpec, tensor: torch.Tensor) -> bool:
        del spec, tensor
        return True

    def run(
        self,
        spec: AllReduceOpSpec,
        tensor: torch.Tensor,
        *,
        group: dist.ProcessGroup,
    ) -> torch.Tensor:
        raise NotImplementedError


ALL_REDUCE_REGISTRY: OpRegistry[AllReduceOpSpec, AllReduceProvider] = OpRegistry(
    "all-reduce"
)


@ALL_REDUCE_REGISTRY.register
class FlashInferTrtllmAllReduceProvider(AllReduceProvider):
    name = "flashinfer_trtllm_sm90"
    priority = 100

    @classmethod
    def supports(cls, spec: AllReduceOpSpec, caps: DeviceCaps) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.no(
                f"requires CUDA SM90, got {caps.platform.name} {caps.compute_capability}"
            )
        if caps.device_name != "NVIDIA H100 80GB HBM3":
            return SupportResult.no(
                "requires profiled NVIDIA H100 80GB HBM3 hardware, "
                f"got {caps.device_name}"
            )
        if not runtime_version_at_least(caps.runtime_version, (12, 8)):
            return SupportResult.no(
                "requires CUDA runtime >= 12.8, "
                f"got {caps.runtime_version or 'unknown'}"
            )
        if not caps.supports_graph_capture or not spec.cuda_graph:
            return SupportResult.no("requires CUDA Graph execution")
        if spec.world_size != 4:
            return SupportResult.no(
                f"requires profiled world_size=4, got {spec.world_size}"
            )
        if spec.max_tokens > 32:
            return SupportResult.no(
                f"requires max_tokens <= 32, got {spec.max_tokens}"
            )
        if spec.hidden_size != 3072 or spec.dtype != torch.bfloat16:
            return SupportResult.no(
                "requires profiled BF16 hidden_size=3072, got "
                f"{spec.dtype} hidden_size={spec.hidden_size}"
            )
        if find_spec("flashinfer") is None:
            return SupportResult.no("flashinfer is not installed")
        try:
            installed = version("flashinfer-python")
        except PackageNotFoundError:
            return SupportResult.no("flashinfer-python package metadata is unavailable")
        numeric = tuple(int(part) for part in re.findall(r"\d+", installed)[:3])
        if numeric < (0, 6, 15):
            return SupportResult.no(
                f"requires flashinfer-python >= 0.6.15, got {installed}"
            )
        return SupportResult.yes()

    def __init__(self) -> None:
        self.workspace = None

    def prepare(self, spec, *, group, rank) -> None:
        from flashinfer.comm import create_allreduce_fusion_workspace

        self.workspace = create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=spec.world_size,
            rank=rank,
            max_token_num=spec.max_tokens,
            hidden_dim=spec.hidden_size,
            dtype=spec.dtype,
            group=group,
        )

    def can_run(self, spec, tensor) -> bool:
        return (
            tensor.is_cuda
            and tensor.dtype == spec.dtype
            and tensor.ndim == 2
            and 0 < int(tensor.shape[0]) <= spec.max_tokens
            and int(tensor.shape[1]) == spec.hidden_size
            and tensor.is_contiguous()
        )

    def run(self, spec, tensor, *, group) -> torch.Tensor:
        del group
        if self.workspace is None:
            raise RuntimeError("FlashInfer all-reduce provider was not prepared.")
        if not self.can_run(spec, tensor):
            raise ValueError(
                "FlashInfer all-reduce received an unsupported tensor: "
                f"shape={tuple(tensor.shape)} dtype={tensor.dtype} device={tensor.device}."
            )
        from flashinfer.comm import allreduce_fusion
        from flashinfer.comm.trtllm_ar import AllReduceFusionPattern

        return allreduce_fusion(
            input=tensor,
            workspace=self.workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
        )


@ALL_REDUCE_REGISTRY.register
class TorchDistributedAllReduceProvider(AllReduceProvider):
    name = "torch_distributed"
    priority = 10

    @classmethod
    def supports(cls, spec: AllReduceOpSpec, caps: DeviceCaps) -> SupportResult:
        del spec, caps
        return SupportResult.yes()

    def run(self, spec, tensor, *, group) -> torch.Tensor:
        del spec
        dist.all_reduce(tensor, group=group)
        return tensor


def resolve_all_reduce_provider(spec: AllReduceOpSpec) -> AllReduceProvider:
    platform = platforms.current_platform
    device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(ALL_REDUCE_REGISTRY).resolve(spec, caps).provider
