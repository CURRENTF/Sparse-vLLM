from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import torch
import torch.nn.functional as F


FINEGRAINED_FP8_KERNEL_REPO = "kernels-community/finegrained-fp8"
FINEGRAINED_FP8_KERNEL_VERSION = 2
# Kernel-registry revision resolved from version 2. The corresponding source
# repository revision is 061130fedf845f320c56de4425f7404f6512c87e.
FINEGRAINED_FP8_KERNEL_REVISION = "b73afcaafe864016f23a2c44ced47d2a8da103f3"


def _has_native_fp8_dtype() -> bool:
    return hasattr(torch, "float8_e4m3fn")


def require_fp8_backend(
    backend: str = "auto",
    *,
    model_name: str = "qwen3_5",
) -> None:
    backend = str(backend or "auto").strip().lower()
    if backend not in {"auto", "transformers", "reference"}:
        raise ValueError(
            f"Unsupported FP8 backend={backend!r}. Supported backends: "
            "'auto', 'transformers', 'reference'."
        )
    if not _has_native_fp8_dtype():
        raise RuntimeError(
            f"{model_name} FP8 requires torch.float8_e4m3fn. "
            "Install a PyTorch build with native FP8 dtype support."
        )
    if backend == "reference":
        return
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"{model_name} FP8 requires a CUDA device with native FP8 matmul support."
        )
    major, minor = torch.cuda.get_device_capability()
    if (int(major), int(minor)) < (8, 9):
        raise RuntimeError(
            f"{model_name} FP8 requires Hopper/Ada-or-newer native FP8 CUDA support; "
            f"detected compute capability {major}.{minor}."
        )


@lru_cache(1)
def load_finegrained_fp8_kernel() -> ModuleType:
    """Load the frozen FP8 kernel from the local kernel cache only."""

    try:
        from kernels import get_local_kernel, load_kernel
    except ImportError as exc:
        raise RuntimeError(
            "FP8 execution requires kernels==0.15.2; install the qwen35 optional "
            "dependencies in the active uv environment."
        ) from exc

    local_path = os.getenv("SPARSEVLLM_FINEGRAINED_FP8_KERNEL_PATH")
    if local_path:
        kernel = get_local_kernel(Path(local_path))
    else:
        try:
            kernel = load_kernel(
                FINEGRAINED_FP8_KERNEL_REPO,
                lockfile=None,
                revision=FINEGRAINED_FP8_KERNEL_REVISION,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise RuntimeError(
                "The frozen finegrained-fp8 kernel is not present in the local cache. "
                "Prefetch kernels-community/finegrained-fp8 revision "
                f"{FINEGRAINED_FP8_KERNEL_REVISION} before starting Sparse-vLLM."
            ) from exc

    missing = [
        name
        for name in ("matmul", "matmul_batched")
        if not callable(getattr(kernel, name, None))
    ]
    if missing:
        raise RuntimeError(
            "Frozen finegrained-fp8 kernel is missing required entry points: "
            f"{missing}."
        )
    return kernel


def _validate_fp8_weight_and_scale(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor | None,
    block_size: tuple[int, int],
) -> None:
    if tuple(block_size) != (128, 128):
        raise ValueError(
            f"FP8 backend supports block_size=(128, 128), got {block_size}."
        )
    if weight.ndim != 2:
        raise RuntimeError(
            f"FP8 Linear weight must be rank-2, got shape={tuple(weight.shape)}."
        )
    if weight.dtype != torch.float8_e4m3fn:
        raise RuntimeError(
            f"FP8 Linear weight must be torch.float8_e4m3fn, got {weight.dtype}."
        )
    if weight_scale_inv is None:
        raise RuntimeError("FP8 Linear requires weight_scale_inv.")
    if weight_scale_inv.dtype != torch.float32:
        raise RuntimeError(
            f"weight_scale_inv must be FP32, got dtype={weight_scale_inv.dtype}."
        )
    if weight_scale_inv.dim() != 2:
        raise RuntimeError(
            "weight_scale_inv must be rank-2, "
            f"got shape={tuple(weight_scale_inv.shape)}."
        )
    if weight_scale_inv.device != weight.device:
        raise RuntimeError(
            "FP8 weight and weight_scale_inv must be on the same device, got "
            f"weight={weight.device}, scale={weight_scale_inv.device}."
        )
    expected = (
        (int(weight.shape[0]) + block_size[0] - 1) // block_size[0],
        (int(weight.shape[1]) + block_size[1] - 1) // block_size[1],
    )
    if tuple(weight_scale_inv.shape) != expected:
        raise RuntimeError(
            "weight_scale_inv shape mismatch: "
            f"expected={expected}, got={tuple(weight_scale_inv.shape)} "
            f"for weight={tuple(weight.shape)}."
        )


def fp8_blockwise_dequantize(
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    *,
    block_size: tuple[int, int] = (128, 128),
    output_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Explicit block-wise FP8 dequantization for correctness oracles."""

    block_size = tuple(int(value) for value in block_size)
    _validate_fp8_weight_and_scale(weight, weight_scale_inv, block_size)
    if output_dtype not in {torch.float32, torch.bfloat16, torch.float16}:
        raise TypeError(
            "FP8 reference dequantization output must be FP32, BF16, or FP16, "
            f"got {output_dtype}."
        )
    block_rows, block_cols = block_size
    scales = weight_scale_inv.repeat_interleave(block_rows, dim=0)
    scales = scales.repeat_interleave(block_cols, dim=1)
    scales = scales[: weight.shape[0], : weight.shape[1]]
    return weight.to(output_dtype) * scales.to(output_dtype)


def fp8_blockwise_linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale_inv: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    block_size: tuple[int, int] = (128, 128),
) -> torch.Tensor:
    """Slow explicit-dequant FP8 Linear used only by the reference backend."""

    if x.device != weight.device:
        raise RuntimeError(
            f"FP8 reference input and weight must share a device, got {x.device} and "
            f"{weight.device}."
        )
    if x.shape[-1] != weight.shape[-1]:
        raise RuntimeError(
            f"FP8 Linear input feature mismatch: input={tuple(x.shape)} "
            f"weight={tuple(weight.shape)}."
        )
    output_dtype = (
        x.dtype
        if x.dtype in {torch.float32, torch.bfloat16, torch.float16}
        else torch.bfloat16
    )
    dequantized = fp8_blockwise_dequantize(
        weight,
        weight_scale_inv,
        block_size=block_size,
        output_dtype=output_dtype,
    )
    return F.linear(x.to(output_dtype), dequantized, bias)


@dataclass(frozen=True)
class Fp8BlockScaledLinearBackend:
    """Local-rank block-scaled FP8 Linear backend."""

    block_size: tuple[int, int] = (128, 128)
    backend: str = "auto"
    model_name: str = "qwen3_5"

    def __post_init__(self) -> None:
        normalized_backend = str(self.backend or "auto").strip().lower()
        object.__setattr__(self, "backend", normalized_backend)
        object.__setattr__(
            self,
            "block_size",
            tuple(int(value) for value in self.block_size),
        )
        require_fp8_backend(normalized_backend, model_name=self.model_name)

    def validate_weight_and_scale(
        self,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor | None,
    ) -> None:
        _validate_fp8_weight_and_scale(
            weight,
            weight_scale_inv,
            self.block_size,
        )

    def __call__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        weight_scale_inv: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.validate_weight_and_scale(weight, weight_scale_inv)
        if self.backend == "reference":
            return fp8_blockwise_linear_reference(
                x,
                weight,
                weight_scale_inv,
                bias,
                block_size=self.block_size,
            )
        if x.device.type != "cuda" or weight.device.type != "cuda":
            raise RuntimeError("Native FP8 Linear requires CUDA tensors.")

        original_shape = x.shape[:-1]
        x_2d = x.reshape(-1, x.shape[-1]).contiguous()
        if x_2d.shape[-1] != weight.shape[-1]:
            raise RuntimeError(
                f"FP8 Linear input feature mismatch: input={tuple(x.shape)} "
                f"weight={tuple(weight.shape)}."
            )

        output_dtype = (
            x.dtype if x.dtype in (torch.float16, torch.bfloat16) else torch.bfloat16
        )
        kernel = load_finegrained_fp8_kernel()
        output = kernel.matmul(
            x_2d,
            weight,
            weight_scale_inv,
            list(self.block_size),
            output_dtype,
        )
        if bias is not None:
            output.add_(bias)
        return output.reshape(*original_shape, weight.shape[0])
