from __future__ import annotations

import ctypes
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator

import torch
import torch.distributed as dist

from sparsevllm.utils.log import logger

if TYPE_CHECKING:
    from sparsevllm.distributed.parallel_context import ParallelGroup


class FlashInferCustomAllReduce:
    """Small-tensor TP all-reduce using FlashInfer's vLLM kernel."""

    _REQUIRED_COMM_APIS = (
        "CudaRTLibrary",
        "create_shared_buffer",
        "vllm_all_reduce",
        "vllm_dispose",
        "vllm_get_graph_buffer_ipc_meta",
        "vllm_init_custom_ar",
        "vllm_meta_size",
        "vllm_register_buffer",
        "vllm_register_graph_buffers",
    )

    @classmethod
    def unsupported_reason(cls, group: ParallelGroup) -> str | None:
        """Return why this group cannot use the provider, or ``None``."""

        if group.size != 2 or group.process_group is None:
            return "requires a two-rank process group"
        if dist.get_backend(group.process_group) != dist.Backend.NCCL:
            return "requires an NCCL process group"
        if not torch.cuda.is_available():
            return "requires CUDA"

        device_count = int(torch.cuda.device_count())
        if any(rank < 0 or rank >= device_count for rank in group.ranks):
            return (
                "requires all group ranks to map to visible local CUDA devices, "
                f"got ranks={group.ranks}, device_count={device_count}"
            )
        current_device = int(torch.cuda.current_device())
        capability = tuple(
            int(value) for value in torch.cuda.get_device_capability(current_device)
        )
        if capability != (9, 0):
            return f"requires CUDA SM90, got {capability}"
        expected_device = int(group.ranks[group.rank])
        if current_device != expected_device:
            return (
                "requires world-rank-to-device mapping for CUDA IPC, "
                f"got current_device={current_device}, expected={expected_device}"
            )
        for peer_rank, peer_device in enumerate(group.ranks):
            if peer_rank == group.rank:
                continue
            if not torch.cuda.can_device_access_peer(current_device, peer_device):
                return (
                    "requires CUDA peer access between all ranks, "
                    f"missing {current_device}->{peer_device} for ranks={group.ranks}"
                )

        try:
            from flashinfer import comm
        except (ImportError, OSError, RuntimeError) as exc:
            return f"FlashInfer communication APIs are unavailable: {exc}"
        missing = [name for name in cls._REQUIRED_COMM_APIS if not hasattr(comm, name)]
        if missing:
            return "FlashInfer communication APIs are missing: " + ", ".join(missing)
        return None

    def __init__(
        self,
        group: ParallelGroup,
        *,
        max_size_bytes: int = 8 * 1024 * 1024,
        num_ctas: int = 32,
    ) -> None:
        if max_size_bytes <= 0 or num_ctas <= 0:
            raise ValueError(
                "FlashInfer custom all-reduce sizes must be positive, got "
                f"max_size_bytes={max_size_bytes}, num_ctas={num_ctas}."
            )
        unsupported_reason = self.unsupported_reason(group)
        if unsupported_reason is not None:
            raise RuntimeError(
                "FlashInfer custom all-reduce is unsupported: "
                f"{unsupported_reason}."
            )

        from flashinfer.comm import (
            CudaRTLibrary,
            create_shared_buffer,
            vllm_all_reduce,
            vllm_dispose,
            vllm_get_graph_buffer_ipc_meta,
            vllm_init_custom_ar,
            vllm_meta_size,
            vllm_register_buffer,
            vllm_register_graph_buffers,
        )

        self.group = group
        self.max_size_bytes = int(max_size_bytes)
        self.num_ctas = int(num_ctas)
        self._cudart = CudaRTLibrary()
        self._cuda_ipc_close = self._cudart.lib.cudaIpcCloseMemHandle
        self._cuda_ipc_close.restype = ctypes.c_int
        self._cuda_ipc_close.argtypes = [ctypes.c_void_p]
        self._all_reduce = vllm_all_reduce
        self._dispose = vllm_dispose
        self._get_graph_buffer_ipc_meta = vllm_get_graph_buffer_ipc_meta
        self._register_graph_buffers = vllm_register_graph_buffers
        self._capturing = False
        self._closed = False
        self._supported_sizes: dict[int, int] = {}
        self._fallback_sizes: dict[int, int] = {}
        self._captured_sizes: dict[int, int] = {}

        self._meta_ptrs = create_shared_buffer(
            vllm_meta_size() + self.max_size_bytes,
            group.process_group,
        )
        self._buffer_ptrs = create_shared_buffer(
            self.max_size_bytes,
            group.process_group,
        )
        self._rank_data = torch.empty(
            8 * 1024 * 1024,
            dtype=torch.uint8,
            device="cuda",
        )
        # TP2 is valid without assuming NVLink. Peer access was checked above.
        self._handle = vllm_init_custom_ar(
            self._meta_ptrs,
            self._rank_data,
            group.rank,
            False,
        )
        vllm_register_buffer(self._handle, self._buffer_ptrs)

    def supports(self, tensor: torch.Tensor) -> bool:
        size_bytes = int(tensor.numel() * tensor.element_size())
        return bool(
            tensor.is_cuda
            and tensor.dtype in {torch.bfloat16, torch.float16}
            and tensor.is_contiguous()
            and size_bytes > 0
            and size_bytes % 16 == 0
            and size_bytes <= self.max_size_bytes
        )

    def all_reduce(self, tensor: torch.Tensor) -> torch.Tensor | None:
        size_bytes = int(tensor.numel() * tensor.element_size())
        if not self.supports(tensor):
            self._fallback_sizes[size_bytes] = self._fallback_sizes.get(size_bytes, 0) + 1
            return None
        self._supported_sizes[size_bytes] = self._supported_sizes.get(size_bytes, 0) + 1
        output = torch.empty_like(tensor)
        registered = self._capturing and torch.cuda.is_current_stream_capturing()
        if registered:
            self._captured_sizes[size_bytes] = self._captured_sizes.get(size_bytes, 0) + 1
        self._all_reduce(
            self._handle,
            tensor,
            output,
            0 if registered else self._buffer_ptrs[self.group.rank],
            0 if registered else self.max_size_bytes,
            self.num_ctas,
        )
        return output

    def _register_captured_buffers(self) -> None:
        handles, offsets = self._get_graph_buffer_ipc_meta(self._handle)
        metadata = [None] * self.group.size
        dist.all_gather_object(
            metadata,
            [handles, offsets],
            group=self.group.process_group,
        )
        self._register_graph_buffers(
            self._handle,
            [item[0] for item in metadata],
            [item[1] for item in metadata],
        )
        logger.info(
            "FlashInfer custom TP all-reduce capture evidence: rank={}, "
            "captured_sizes={}, supported_sizes={}, fallback_sizes={}, "
            "graph_buffers={}",
            self.group.rank,
            dict(sorted(self._captured_sizes.items())),
            dict(sorted(self._supported_sizes.items())),
            dict(sorted(self._fallback_sizes.items())),
            len(offsets),
        )

    @contextmanager
    def capture(self) -> Iterator[None]:
        if self._capturing:
            raise RuntimeError("FlashInfer custom all-reduce capture is already active.")
        self._capturing = True
        completed = False
        try:
            yield
            completed = True
        finally:
            self._capturing = False
            if completed:
                self._register_captured_buffers()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._dispose(self._handle)
        self._close_shared_buffer(self._buffer_ptrs)
        self._close_shared_buffer(self._meta_ptrs)

    def _close_shared_buffer(self, pointers: list[int]) -> None:
        device_ids = [torch.cuda.current_device()]
        dist.barrier(group=self.group.process_group, device_ids=device_ids)
        for peer_rank, pointer in enumerate(pointers):
            if peer_rank == self.group.rank:
                continue
            result = int(self._cuda_ipc_close(ctypes.c_void_p(pointer)))
            if result != 0:
                raise RuntimeError(
                    "cudaIpcCloseMemHandle failed: "
                    f"{self._cudart.cudaGetErrorString(result)}"
                )
        dist.barrier(group=self.group.process_group, device_ids=device_ids)
        self._cudart.cudaFree(ctypes.c_void_p(pointers[self.group.rank]))
        dist.barrier(group=self.group.process_group, device_ids=device_ids)


__all__ = ["FlashInferCustomAllReduce"]
