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
        group: dist.ProcessGroup | None,
        rank: int,
        device_index: int | None = None,
    ) -> None:
        del spec, group, rank, device_index

    def close(self) -> None:
        pass

    def run(
        self,
        spec: AllReduceOpSpec,
        tensor: torch.Tensor,
        *,
        group: dist.ProcessGroup | None,
    ) -> torch.Tensor:
        """Return the reduced tensor, which may or may not alias the input."""
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
        self._output_buffer: torch.Tensor | None = None

    def prepare(self, spec, *, group, rank, device_index=None) -> None:
        from flashinfer.comm import create_allreduce_fusion_workspace

        if self.workspace is not None or self._output_buffer is not None:
            raise RuntimeError("FlashInfer all-reduce provider is already prepared.")
        if group is None:
            raise RuntimeError("FlashInfer all-reduce requires a distributed process group.")
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "FlashInfer all-reduce must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        workspace = create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=spec.world_size,
            rank=rank,
            max_token_num=spec.max_tokens,
            hidden_dim=spec.hidden_size,
            dtype=spec.dtype,
            group=group,
        )
        try:
            output_buffer = torch.empty(
                (spec.max_tokens, spec.hidden_size),
                dtype=spec.dtype,
                device=torch.device("cuda", int(device_index)),
            )
        except Exception:
            workspace.destroy()
            raise
        self.workspace = workspace
        self._output_buffer = output_buffer

    def close(self) -> None:
        workspace = self.workspace
        self.workspace = None
        self._output_buffer = None
        if workspace is not None:
            workspace.destroy()

    def run(self, spec, tensor, *, group) -> torch.Tensor:
        del group
        if self.workspace is None or self._output_buffer is None:
            raise RuntimeError("FlashInfer all-reduce provider was not prepared.")
        if not tensor.is_cuda:
            raise ValueError(
                f"FlashInfer all-reduce requires a CUDA tensor, got {tensor.device}."
            )
        from flashinfer.comm import allreduce_fusion
        from flashinfer.comm.trtllm_ar import AllReduceFusionPattern

        output = self._output_buffer[: int(tensor.shape[0])]
        allreduce_fusion(
            input=tensor,
            workspace=self.workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            output=output,
        )
        return output


@ALL_REDUCE_REGISTRY.register
class TorchDistributedAllReduceProvider(AllReduceProvider):
    name = "torch_distributed"
    priority = 10

    @classmethod
    def supports(cls, spec: AllReduceOpSpec, caps: DeviceCaps) -> SupportResult:
        del spec, caps
        return SupportResult.yes()

    def run(self, spec, tensor, *, group) -> torch.Tensor:
        if spec.world_size > 1:
            if group is None:
                raise RuntimeError(
                    "Torch distributed all-reduce requires a process group when world_size > 1."
                )
            dist.all_reduce(tensor, group=group)
        return tensor


def _validate_tensor_contract(spec: AllReduceOpSpec, tensor: torch.Tensor) -> None:
    if tensor.dtype != spec.dtype:
        raise TypeError(
            f"All-reduce expected dtype={spec.dtype}, got {tensor.dtype}."
        )
    if tensor.ndim != 2:
        raise ValueError(
            f"All-reduce expects [tokens, hidden], got shape={tuple(tensor.shape)}."
        )
    token_count, hidden_size = (int(value) for value in tensor.shape)
    if not 0 < token_count <= spec.max_tokens:
        raise ValueError(
            "All-reduce token count is outside the prepared range: "
            f"tokens={token_count} max_tokens={spec.max_tokens}."
        )
    if hidden_size != spec.hidden_size:
        raise ValueError(
            "All-reduce hidden size does not match the prepared operator: "
            f"hidden={hidden_size} expected={spec.hidden_size}."
        )
    if not tensor.is_contiguous():
        raise ValueError(
            f"All-reduce requires a contiguous tensor, got stride={tensor.stride()}."
        )


class PreparedAllReduceOp:
    """A pre-bound all-reduce with no execution-time fallback."""

    def __init__(
        self,
        spec: AllReduceOpSpec,
        provider: AllReduceProvider,
        *,
        group: dist.ProcessGroup | None,
    ) -> None:
        self.spec = spec
        self.provider = provider
        self.group = group
        self._closed = False

    @property
    def name(self) -> str:
        return self.provider.name

    def run(self, tensor: torch.Tensor) -> torch.Tensor:
        if self._closed:
            raise RuntimeError("All-reduce operator is closed.")
        _validate_tensor_contract(self.spec, tensor)
        output = self.provider.run(self.spec, tensor, group=self.group)
        if not isinstance(output, torch.Tensor):
            raise TypeError(
                f"All-reduce provider {self.provider.name} returned "
                f"{type(output).__name__}, expected torch.Tensor."
            )
        _validate_tensor_contract(self.spec, output)
        if output.shape != tensor.shape or output.device != tensor.device:
            raise ValueError(
                f"All-reduce provider {self.provider.name} returned an incompatible tensor: "
                f"input_shape={tuple(tensor.shape)} output_shape={tuple(output.shape)} "
                f"input_device={tensor.device} output_device={output.device}."
            )
        return output

    def close(self) -> None:
        if self._closed:
            return
        self.provider.close()
        self._closed = True


def resolve_all_reduce_provider(
    spec: AllReduceOpSpec,
    *,
    device_index: int | None = None,
) -> AllReduceProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(ALL_REDUCE_REGISTRY).resolve(spec, caps).provider


def prepare_all_reduce_op(
    spec: AllReduceOpSpec,
    *,
    group: dist.ProcessGroup | None,
    rank: int,
    device_index: int | None = None,
    provider: AllReduceProvider | None = None,
) -> PreparedAllReduceOp:
    rank = int(rank)
    if not 0 <= rank < spec.world_size:
        raise ValueError(
            f"All-reduce rank must be in [0, {spec.world_size}), got {rank}."
        )
    if spec.world_size > 1 and group is None:
        raise RuntimeError(
            "All-reduce requires a process group when world_size > 1."
        )
    if provider is None:
        provider = resolve_all_reduce_provider(spec, device_index=device_index)
    provider.prepare(
        spec,
        group=group,
        rank=rank,
        device_index=device_index,
    )
    return PreparedAllReduceOp(spec, provider, group=group)
