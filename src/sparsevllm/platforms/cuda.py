from __future__ import annotations

from functools import lru_cache

import torch

from sparsevllm.platforms.interface import (
    AllocatorStats,
    DeviceCaps,
    Platform,
    PlatformEnum,
)


class CudaPlatform(Platform):
    name = "cuda"
    device_type = "cuda"
    enum = PlatformEnum.CUDA

    def check_available(self) -> bool:
        return bool(torch.cuda.is_available() and torch.version.hip is None)

    def validate_environment(self) -> None:
        if not self.check_available():
            raise RuntimeError("CUDA platform was selected, but torch.cuda is unavailable or is backed by ROCm.")

    def supports_inference(self) -> bool:
        return True

    def get_device(self, local_rank: int = 0) -> torch.device:
        return torch.device(self.device_type, int(local_rank))

    def set_device(self, device: torch.device | int | str) -> None:
        torch.cuda.set_device(device)

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        return torch.cuda.mem_get_info(int(device_id))

    def get_allocator_stats(self, device: torch.device | None = None) -> AllocatorStats:
        stats = torch.cuda.memory_stats(device)
        return AllocatorStats(
            peak_allocated_bytes=int(stats.get("allocated_bytes.all.peak", 0)),
            current_allocated_bytes=int(stats.get("allocated_bytes.all.current", 0)),
        )

    def reset_peak_memory_stats(self, device: torch.device | None = None) -> None:
        torch.cuda.reset_peak_memory_stats(device)

    def empty_cache(self) -> None:
        torch.cuda.empty_cache()

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_stream_capturing(self) -> bool:
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())

    def get_distributed_backend(self) -> str:
        return "nccl"

    def barrier_device_ids(self, rank: int) -> list[int] | None:
        return [int(rank)]

    @lru_cache(maxsize=None)
    def get_device_caps(self, device_index: int = 0) -> DeviceCaps:
        device_index = int(device_index)
        major, minor = torch.cuda.get_device_capability(device_index)
        return DeviceCaps(
            platform=self.enum,
            device_type=self.device_type,
            device_index=device_index,
            device_name=str(torch.cuda.get_device_name(device_index)),
            compute_capability=(int(major), int(minor)),
            runtime_version=torch.version.cuda,
            supports_graph_capture=True,
            supports_torch_compile=True,
            supports_triton=True,
            supports_pin_memory=True,
            supports_bfloat16=(int(major), int(minor)) >= (8, 0),
            # Ada (SM89), Hopper and Blackwell provide native FP8 tensor cores.
            supports_native_fp8=(int(major), int(minor)) >= (8, 9),
        )

    def get_default_attention_backend(self) -> str:
        return "cuda_triton"

    def get_decode_graph_runner_cls(self):
        from sparsevllm.engine.decode_cuda_graph import DecodeCudaGraphRunner

        return DecodeCudaGraphRunner
