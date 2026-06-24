from __future__ import annotations

import torch

from sparsevllm.platforms.interface import Platform, PlatformEnum


class RocmPlatform(Platform):
    name = "rocm"
    device_type = "cuda"
    enum = PlatformEnum.ROCM

    def check_available(self) -> bool:
        return bool(torch.cuda.is_available() and torch.version.hip is not None)

    def validate_environment(self) -> None:
        if not self.check_available():
            raise RuntimeError("ROCm platform was selected, but PyTorch is not running with HIP support.")
        raise RuntimeError(
            "ROCm was detected, but Sparse-vLLM ROCm inference is not supported yet. "
            "Do not run this build on ROCm until a ROCm platform/op backend is implemented."
        )

    def get_dispatch_key(self) -> str:
        return "rocm"
