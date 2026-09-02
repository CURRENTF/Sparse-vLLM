from __future__ import annotations

import re
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from typing import Any

import torch


class PlatformEnum(Enum):
    CUDA = auto()
    ROCM = auto()
    NPU = auto()
    CPU = auto()
    UNSPECIFIED = auto()


_ACCELERATOR_FAMILY_PATTERN = re.compile(
    r"(?<![A-Z0-9])(GB\d{3}|B\d{3}|H\d{2,3}|A\d{2,3}|L\d{2,3}S?)(?![A-Z0-9])",
    re.IGNORECASE,
)
_RTX_PRO_FAMILY_PATTERN = re.compile(
    r"(?<![A-Z0-9])RTX\s+PRO\s+(\d{4})(?![A-Z0-9])",
    re.IGNORECASE,
)


def normalize_accelerator_identity(device_name: str) -> tuple[str, str | None]:
    """Return a stable product family and an optional physical variant."""

    normalized_name = " ".join(str(device_name).strip().upper().split())
    rtx_pro_match = _RTX_PRO_FAMILY_PATTERN.search(normalized_name)
    if rtx_pro_match is not None:
        return f"rtx_pro_{rtx_pro_match.group(1)}", None
    match = _ACCELERATOR_FAMILY_PATTERN.search(normalized_name)
    family = match.group(1).lower() if match is not None else "unknown"
    if family != "h100":
        return family, None
    if "PCIE" in normalized_name:
        return family, "pcie"
    if "SXM" in normalized_name or "80GB HBM3" in normalized_name:
        return family, "sxm"
    return family, None


@dataclass(frozen=True)
class AllocatorStats:
    peak_allocated_bytes: int = 0
    current_allocated_bytes: int = 0


@dataclass(frozen=True)
class DeviceCaps:
    platform: PlatformEnum
    device_type: str
    device_index: int
    device_name: str
    compute_capability: tuple[int, int] | None = None
    runtime_version: str | None = None
    supports_graph_capture: bool = False
    supports_torch_compile: bool = False
    supports_triton: bool = False
    supports_pin_memory: bool = False
    supports_bfloat16: bool = False
    supports_native_fp8: bool = False
    multi_processor_count: int | None = None
    accelerator_family: str | None = None
    accelerator_variant: str | None = None

    def __post_init__(self) -> None:
        family, variant = normalize_accelerator_identity(self.device_name)
        if self.accelerator_family is None:
            object.__setattr__(self, "accelerator_family", family)
        if self.accelerator_variant is None and self.accelerator_family == family:
            object.__setattr__(self, "accelerator_variant", variant)


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

    def reset_peak_memory_stats(self, device: torch.device | None = None) -> None:
        del device
        return None

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

    @lru_cache(maxsize=None)
    def get_device_caps(self, device_index: int = 0) -> DeviceCaps:
        return DeviceCaps(
            platform=self.enum,
            device_type=self.device_type,
            device_index=int(device_index),
            device_name=self.name,
            supports_graph_capture=False,
            supports_torch_compile=False,
            supports_triton=False,
            supports_pin_memory=False,
            supports_bfloat16=False,
            supports_native_fp8=False,
        )

    def supports_graph_capture(self) -> bool:
        return self.get_device_caps().supports_graph_capture

    def supports_torch_compile(self) -> bool:
        return self.get_device_caps().supports_torch_compile

    def supports_triton(self) -> bool:
        return self.get_device_caps().supports_triton

    def supports_pin_memory(self) -> bool:
        return self.get_device_caps().supports_pin_memory

    def supports_fp8(self) -> bool:
        return self.get_device_caps().supports_native_fp8

    def supports_bfloat16(self) -> bool:
        return self.get_device_caps().supports_bfloat16

    def get_default_attention_backend(self) -> str:
        return "native"

    def get_decode_graph_runner_cls(self):
        return None

    def get_dispatch_key(self) -> str:
        return self.name

    def apply_config_defaults(self, config: Any) -> None:
        return None

    def validate_config(self, config: Any) -> None:
        if getattr(config, "decode_graph", False):
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
