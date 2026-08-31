from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

from sparsevllm.platforms.interface import Platform


@dataclass(frozen=True)
class DeviceMemorySnapshot:
    free_bytes: int
    total_bytes: int
    allocated_bytes: int
    peak_allocated_bytes: int

    @classmethod
    def capture(
        cls,
        platform: Platform,
        device: torch.device,
    ) -> "DeviceMemorySnapshot":
        free_bytes, total_bytes = platform.get_available_memory(device.index or 0)
        allocator = platform.get_allocator_stats(device)
        return cls(
            free_bytes=int(free_bytes),
            total_bytes=int(total_bytes),
            allocated_bytes=int(allocator.current_allocated_bytes),
            peak_allocated_bytes=int(allocator.peak_allocated_bytes),
        )


@dataclass(frozen=True)
class MemoryProfileMeasurement:
    consumed_bytes: int
    transient_peak_bytes: int

    @classmethod
    def from_snapshots(
        cls,
        before: DeviceMemorySnapshot,
        after: DeviceMemorySnapshot,
    ) -> "MemoryProfileMeasurement":
        if before.total_bytes != after.total_bytes:
            raise RuntimeError(
                "GPU total memory changed during startup profiling: "
                f"before={before.total_bytes} after={after.total_bytes}."
            )
        return cls(
            consumed_bytes=max(0, before.free_bytes - after.free_bytes),
            transient_peak_bytes=max(
                0,
                after.peak_allocated_bytes - after.allocated_bytes,
            ),
        )


def release_unused_device_memory(platform: Platform) -> None:
    platform.synchronize()
    gc.collect()
    platform.empty_cache()
    platform.synchronize()


__all__ = [
    "DeviceMemorySnapshot",
    "MemoryProfileMeasurement",
    "release_unused_device_memory",
]
