from __future__ import annotations

import ctypes
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
    ranks: tuple[int, ...]
    max_rows: int
    hidden_size: int
    dtype: torch.dtype
    cuda_graph: bool
    backend: str

    def __post_init__(self) -> None:
        if self.world_size <= 0 or self.max_rows <= 0 or self.hidden_size <= 0:
            raise ValueError("All-reduce dimensions must be positive.")
        if (
            len(self.ranks) != self.world_size
            or len(set(self.ranks)) != self.world_size
        ):
            raise ValueError(
                f"All-reduce ranks must contain world_size unique ranks, got {self.ranks}."
            )


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


@dataclass(frozen=True)
class _FlashInferTrtllmProfile:
    max_rows: int
    launch_with_pdl: bool = False
    completion_row_threshold: int | None = None
    provider_output_buffer: bool = True


_FLASHINFER_TRTLLM_PROFILES = {
    ("NVIDIA H100 80GB HBM3", 2, 2048): _FlashInferTrtllmProfile(
        max_rows=256,
        launch_with_pdl=True,
        completion_row_threshold=16,
        provider_output_buffer=False,
    ),
    ("NVIDIA H100 80GB HBM3", 4, 3072): _FlashInferTrtllmProfile(max_rows=32),
}


def _flashinfer_dependency_reason() -> str | None:
    if find_spec("flashinfer") is None:
        return "flashinfer is not installed"
    try:
        installed = version("flashinfer-python")
    except PackageNotFoundError:
        return "flashinfer-python package metadata is unavailable"
    numeric = tuple(int(part) for part in re.findall(r"\d+", installed)[:3])
    return (
        f"requires flashinfer-python >= 0.6.15, got {installed}"
        if numeric < (0, 6, 15)
        else None
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
        if spec.backend != "nccl":
            return SupportResult.no(f"requires NCCL, got {spec.backend}")
        profile = _FLASHINFER_TRTLLM_PROFILES.get(
            (caps.device_name, spec.world_size, spec.hidden_size)
        )
        if profile is None or spec.dtype != torch.bfloat16:
            return SupportResult.no(
                "requires a profiled BF16 topology/shape, got "
                f"world_size={spec.world_size} hidden_size={spec.hidden_size} "
                f"dtype={spec.dtype}"
            )
        if spec.max_rows > profile.max_rows:
            return SupportResult.no(
                f"requires max_rows <= {profile.max_rows}, got {spec.max_rows}"
            )
        reason = _flashinfer_dependency_reason()
        return SupportResult.no(reason) if reason else SupportResult.yes()

    def __init__(self) -> None:
        self.workspace = None
        self._output_buffer: torch.Tensor | None = None
        self._profile: _FlashInferTrtllmProfile | None = None

    def prepare(self, spec, *, group, rank, device_index=None) -> None:
        from flashinfer.comm import create_allreduce_fusion_workspace

        if self.workspace is not None or self._output_buffer is not None:
            raise RuntimeError("FlashInfer all-reduce provider is already prepared.")
        if group is None:
            raise RuntimeError("FlashInfer all-reduce requires a distributed process group.")
        if dist.get_backend(group) != dist.Backend.NCCL:
            raise RuntimeError("FlashInfer all-reduce requires an NCCL process group.")
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device:
            raise RuntimeError(
                "FlashInfer all-reduce must be prepared on the selected CUDA device: "
                f"selected={device_index} current={current_device}."
            )
        caps = platforms.current_platform.get_device_caps(current_device)
        profile = _FLASHINFER_TRTLLM_PROFILES[
            (caps.device_name, spec.world_size, spec.hidden_size)
        ]
        workspace = create_allreduce_fusion_workspace(
            backend="trtllm",
            world_size=spec.world_size,
            rank=rank,
            max_token_num=profile.max_rows,
            hidden_dim=spec.hidden_size,
            dtype=spec.dtype,
            group=group,
        )
        try:
            output_buffer = (
                torch.empty(
                    (spec.max_rows, spec.hidden_size),
                    dtype=spec.dtype,
                    device=torch.device("cuda", int(device_index)),
                )
                if profile.provider_output_buffer
                else None
            )
        except Exception:
            workspace.destroy()
            raise
        self.workspace = workspace
        self._output_buffer = output_buffer
        self._profile = profile

    def close(self) -> None:
        workspace = self.workspace
        self.workspace = None
        self._output_buffer = None
        self._profile = None
        if workspace is not None:
            workspace.destroy()

    def run(self, spec, tensor, *, group) -> torch.Tensor:
        del group
        if self.workspace is None or self._profile is None:
            raise RuntimeError("FlashInfer all-reduce provider was not prepared.")
        if not tensor.is_cuda:
            raise ValueError(
                f"FlashInfer all-reduce requires a CUDA tensor, got {tensor.device}."
            )
        from flashinfer.comm import allreduce_fusion
        from flashinfer.comm.trtllm_ar import AllReduceFusionPattern

        flattened = tensor.view(-1, spec.hidden_size)
        output = (
            None
            if self._output_buffer is None
            else self._output_buffer[: int(flattened.shape[0])]
        )
        result = allreduce_fusion(
            input=flattened,
            workspace=self.workspace,
            pattern=AllReduceFusionPattern.kAllReduce,
            launch_with_pdl=self._profile.launch_with_pdl,
            trigger_completion_at_end=(
                self._profile.completion_row_threshold is None
                or int(flattened.shape[0]) > self._profile.completion_row_threshold
            ),
            output=output,
        )
        return result.view_as(tensor)


@ALL_REDUCE_REGISTRY.register
class FlashInferVllmAllReduceProvider(AllReduceProvider):
    name = "flashinfer_vllm_sm90"
    priority = 100
    max_rows = 256
    num_ctas = 32

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
        if spec.cuda_graph:
            return SupportResult.no("requires eager execution")
        if spec.backend != "nccl":
            return SupportResult.no(f"requires NCCL, got {spec.backend}")
        if (
            spec.world_size != 2
            or spec.max_rows > cls.max_rows
            or spec.hidden_size != 2048
            or spec.dtype != torch.bfloat16
        ):
            return SupportResult.no(
                "requires profiled TP2 BF16 [..., 2048] with max_rows <= 256, "
                f"got world_size={spec.world_size} max_rows={spec.max_rows} "
                f"hidden_size={spec.hidden_size} dtype={spec.dtype}"
            )
        reason = _flashinfer_dependency_reason()
        if reason:
            return SupportResult.no(reason)
        try:
            from flashinfer import comm
        except (ImportError, OSError, RuntimeError) as exc:
            return SupportResult.no(
                f"FlashInfer communication APIs are unavailable: {exc}"
            )
        required = (
            "CudaRTLibrary",
            "create_shared_buffer",
            "vllm_all_reduce",
            "vllm_dispose",
            "vllm_init_custom_ar",
            "vllm_meta_size",
            "vllm_register_buffer",
        )
        missing = [name for name in required if not hasattr(comm, name)]
        return (
            SupportResult.no(
                "FlashInfer communication APIs are missing: " + ", ".join(missing)
            )
            if missing
            else SupportResult.yes()
        )

    def __init__(self) -> None:
        self._group: dist.ProcessGroup | None = None
        self._rank = -1
        self._max_size_bytes = 0
        self._rank_data: torch.Tensor | None = None
        self._meta_ptrs: list[int] = []
        self._buffer_ptrs: list[int] = []
        self._handle = None
        self._cudart = None

    def prepare(self, spec, *, group, rank, device_index=None) -> None:
        from flashinfer.comm import (
            CudaRTLibrary,
            create_shared_buffer,
            vllm_init_custom_ar,
            vllm_meta_size,
            vllm_register_buffer,
        )

        if self._handle is not None:
            raise RuntimeError("FlashInfer all-reduce provider is already prepared.")
        if group is None:
            raise RuntimeError(
                "FlashInfer all-reduce requires a distributed process group."
            )
        current_device = torch.cuda.current_device()
        if device_index is None:
            device_index = current_device
        if int(device_index) != current_device or current_device != spec.ranks[rank]:
            raise RuntimeError(
                "FlashInfer vLLM all-reduce requires rank-to-device mapping: "
                f"ranks={spec.ranks} rank={rank} selected={device_index} "
                f"current={current_device}."
            )
        if any(
            peer != current_device
            and not torch.cuda.can_device_access_peer(current_device, peer)
            for peer in spec.ranks
        ):
            raise RuntimeError("FlashInfer vLLM all-reduce requires CUDA peer access.")
        max_size_bytes = spec.max_rows * spec.hidden_size * spec.dtype.itemsize
        meta_ptrs = create_shared_buffer(vllm_meta_size() + max_size_bytes, group)
        buffer_ptrs = create_shared_buffer(max_size_bytes, group)
        rank_data = torch.empty(8 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        handle = vllm_init_custom_ar(meta_ptrs, rank_data, rank, False)
        vllm_register_buffer(handle, buffer_ptrs)
        self._group = group
        self._rank = rank
        self._max_size_bytes = max_size_bytes
        self._rank_data = rank_data
        self._meta_ptrs = meta_ptrs
        self._buffer_ptrs = buffer_ptrs
        self._handle = handle
        self._cudart = CudaRTLibrary()

    def run(self, spec, tensor, *, group) -> torch.Tensor:
        del spec, group
        if self._handle is None:
            raise RuntimeError("FlashInfer all-reduce provider was not prepared.")
        from flashinfer.comm import vllm_all_reduce

        output = torch.empty_like(tensor)
        vllm_all_reduce(
            self._handle,
            tensor,
            output,
            self._buffer_ptrs[self._rank],
            self._max_size_bytes,
            self.num_ctas,
        )
        return output

    @staticmethod
    def _close_shared_buffer(pointers, group, rank, cudart) -> None:
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        close = cudart.lib.cudaIpcCloseMemHandle
        close.restype = ctypes.c_int
        close.argtypes = [ctypes.c_void_p]
        for peer_rank, pointer in enumerate(pointers):
            if peer_rank != rank:
                result = int(close(ctypes.c_void_p(pointer)))
                if result != 0:
                    raise RuntimeError(
                        f"cudaIpcCloseMemHandle failed: {cudart.cudaGetErrorString(result)}"
                    )
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])
        cudart.cudaFree(ctypes.c_void_p(pointers[rank]))
        dist.barrier(group=group, device_ids=[torch.cuda.current_device()])

    def close(self) -> None:
        if self._handle is None:
            return
        from flashinfer.comm import vllm_dispose

        vllm_dispose(self._handle)
        self._handle = None
        self._close_shared_buffer(
            self._buffer_ptrs, self._group, self._rank, self._cudart
        )
        self._close_shared_buffer(
            self._meta_ptrs, self._group, self._rank, self._cudart
        )
        self._rank_data = None


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
    if tensor.ndim < 2:
        raise ValueError(
            f"All-reduce expects [..., hidden], got shape={tuple(tensor.shape)}."
        )
    row_count = tensor.numel() // int(tensor.shape[-1])
    hidden_size = int(tensor.shape[-1])
    if not 0 < row_count <= spec.max_rows:
        raise ValueError(
            "All-reduce row count is outside the prepared range: "
            f"rows={row_count} max_rows={spec.max_rows}."
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
