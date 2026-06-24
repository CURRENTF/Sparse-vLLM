from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any

import torch


class PlatformEnum(Enum):
    CUDA = auto()
    ROCM = auto()
    NPU = auto()
    CPU = auto()
    UNSPECIFIED = auto()


@dataclass(frozen=True)
class AllocatorStats:
    peak_allocated_bytes: int = 0
    current_allocated_bytes: int = 0


class Platform:
    name: str = "unknown"
    device_type: str = "cpu"
    enum: PlatformEnum = PlatformEnum.UNSPECIFIED
    supported_quantization: tuple[str, ...] = ()

    def check_available(self) -> bool:
        return False

    def validate_environment(self) -> None:
        if not self.check_available():
            raise RuntimeError(f"Platform {self.name!r} is not available.")

    def supports_inference(self) -> bool:
        return False

    def validate_inference(self) -> None:
        self.validate_environment()
        if not self.supports_inference():
            raise RuntimeError(
                f"Platform {self.name!r} is detected, but Sparse-vLLM inference is not supported "
                "on this platform in the current build."
            )

    def init_backend(self) -> None:
        return None

    def get_device(self, local_rank: int = 0) -> torch.device:
        raise NotImplementedError(f"Platform {self.name!r} does not implement get_device().")

    def set_device(self, device: torch.device | int | str) -> None:
        raise NotImplementedError(f"Platform {self.name!r} does not implement set_device().")

    def get_available_memory(self, device_id: int = 0) -> tuple[int, int]:
        raise NotImplementedError(f"Platform {self.name!r} does not implement get_available_memory().")

    def get_allocator_stats(self, device: torch.device | None = None) -> AllocatorStats:
        return AllocatorStats()

    def empty_cache(self) -> None:
        return None

    def synchronize(self) -> None:
        return None

    def is_stream_capturing(self) -> bool:
        return False

    def get_distributed_backend(self) -> str:
        return "gloo"

    def barrier_device_ids(self, rank: int) -> list[int] | None:
        return None

    def get_communicator_cls(self) -> type | None:
        return None

    def supports_graph_capture(self) -> bool:
        return False

    def supports_torch_compile(self) -> bool:
        return False

    def supports_triton(self) -> bool:
        return False

    def supports_pin_memory(self) -> bool:
        return False

    def supports_fp8(self) -> bool:
        return False

    def supports_bfloat16(self) -> bool:
        return False

    def get_default_attention_backend(self) -> str:
        return "native"

    def get_decode_graph_runner_cls(self):
        return None

    def get_dispatch_key(self) -> str:
        return self.name

    def apply_config_defaults(self, config: Any) -> None:
        return None

    def validate_config(self, config: Any) -> None:
        if getattr(config, "decode_graph", False) or getattr(config, "decode_cuda_graph", False):
            if not self.supports_graph_capture():
                raise RuntimeError(f"Platform {self.name!r} does not support decode graph capture.")

    @contextmanager
    def inference_mode(self):
        with torch.inference_mode():
            yield

    def seed_everything(self, seed: int) -> None:
        torch.manual_seed(int(seed))

    def is_cuda(self) -> bool:
        return self.enum == PlatformEnum.CUDA

    def is_rocm(self) -> bool:
        return self.enum == PlatformEnum.ROCM

    def is_npu(self) -> bool:
        return self.enum == PlatformEnum.NPU

    def is_cpu(self) -> bool:
        return self.enum == PlatformEnum.CPU

    def is_cuda_alike(self) -> bool:
        return self.enum in {PlatformEnum.CUDA, PlatformEnum.ROCM}
