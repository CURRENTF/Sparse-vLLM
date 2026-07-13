from __future__ import annotations

from dataclasses import dataclass
from types import ModuleType

import torch


def _load_transformers_fp8_kernel() -> None:
    """Load the exact Transformers fine-grained FP8 kernel with explicit trust."""
    try:
        from kernels import get_kernel
        from transformers.integrations import hub_kernels
    except ImportError as exc:
        raise RuntimeError(
            "qwen3_5 FP8 requires the optional Qwen3.5 dependencies; "
            "install them with `pip install -e '.[qwen35]'`."
        ) from exc

    kernel_name = "finegrained-fp8"
    if isinstance(hub_kernels._KERNEL_MODULE_MAPPING.get(kernel_name), ModuleType):
        return
    kernel_config = hub_kernels._HUB_KERNEL_MAPPING.get(kernel_name)
    if not isinstance(kernel_config, dict):
        raise RuntimeError("Installed Transformers does not define the finegrained-fp8 Hub Kernel.")
    hub_kernels._KERNEL_MODULE_MAPPING[kernel_name] = get_kernel(
        kernel_config["repo_id"],
        revision=kernel_config.get("revision"),
        version=kernel_config.get("version"),
        trust_remote_code=True,
    )


def _has_native_fp8_dtype() -> bool:
    return hasattr(torch, "float8_e4m3fn")


def require_fp8_backend(backend: str = "auto") -> None:
    backend = str(backend or "auto").strip().lower()
    if backend not in {"auto", "transformers"}:
        raise ValueError(f"Unsupported FP8 backend={backend!r}. Supported backends: 'auto', 'transformers'.")
    if not _has_native_fp8_dtype():
        raise RuntimeError(
            "qwen3_5 FP8 requires torch.float8_e4m3fn. "
            "Install a PyTorch build with native FP8 support."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("qwen3_5 FP8 requires a CUDA device with native FP8 matmul support.")
    major, minor = torch.cuda.get_device_capability()
    if (int(major), int(minor)) < (8, 9):
        raise RuntimeError(
            "qwen3_5 FP8 requires a native FP8 CUDA backend on Hopper/Ada-or-newer GPUs; "
            f"detected compute capability {major}.{minor}."
        )


@dataclass(frozen=True)
class Fp8BlockScaledLinearBackend:
    """Local rank FP8 matmul backend.

    Parallel Linear subclasses still own sharding and communication. This object
    only validates and executes the local dense projection.
    """

    block_size: tuple[int, int] = (128, 128)
    backend: str = "auto"

    def __post_init__(self) -> None:
        if tuple(self.block_size) != (128, 128):
            raise ValueError(f"FP8 backend supports block_size=(128, 128), got {self.block_size}.")
        require_fp8_backend(self.backend)

    def validate_weight_and_scale(
        self,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor | None,
    ) -> None:
        if weight.dtype != torch.float8_e4m3fn:
            raise RuntimeError(f"FP8 Linear weight must be torch.float8_e4m3fn, got {weight.dtype}.")
        if weight_scale_inv is None:
            raise RuntimeError("FP8 Linear requires weight_scale_inv.")
        if weight_scale_inv.dim() != 2:
            raise RuntimeError(f"weight_scale_inv must be rank-2, got shape={tuple(weight_scale_inv.shape)}.")
        expected = (
            (int(weight.shape[0]) + self.block_size[0] - 1) // self.block_size[0],
            (int(weight.shape[1]) + self.block_size[1] - 1) // self.block_size[1],
        )
        if tuple(weight_scale_inv.shape) != expected:
            raise RuntimeError(
                "weight_scale_inv shape mismatch: "
                f"expected={expected}, got={tuple(weight_scale_inv.shape)} for weight={tuple(weight.shape)}."
            )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.validate_weight_and_scale(weight, weight_scale_inv)
        if x.device.type != "cuda" or weight.device.type != "cuda":
            raise RuntimeError("FP8 Linear requires CUDA tensors; CPU fallback is not supported.")

        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        if x_2d.shape[-1] != weight.shape[-1]:
            raise RuntimeError(
                f"FP8 Linear input feature mismatch: input={tuple(x.shape)} weight={tuple(weight.shape)}."
            )

        out_dtype = x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        try:
            from transformers.integrations.finegrained_fp8 import finegrained_fp8_linear
        except ImportError as exc:
            raise RuntimeError(
                "qwen3_5 FP8 requires the optional Qwen3.5 dependencies; "
                "install them with `pip install -e '.[qwen35]'`."
            ) from exc
        _load_transformers_fp8_kernel()
        output = finegrained_fp8_linear(
            x_2d,
            weight,
            weight_scale_inv,
            block_size=self.block_size,
            bias=bias,
            output_dtype=out_dtype,
        )
        return output.reshape(*original_shape, weight.shape[0])
