from __future__ import annotations

import importlib
import inspect
from functools import lru_cache

import torch

from sparsevllm.kernels.external.flashinfer.support import (
    flashinfer_kernel_support,
)
from sparsevllm.kernels.external.support import ExternalKernelContractError


_FEATURE = "fused top-k page-table transform"
_REQUIRED_PARAMETERS = frozenset(
    {
        "input",
        "src_page_table",
        "lengths",
        "k",
        "deterministic",
        "tie_break",
        "dsa_graph_safe",
    }
)


@lru_cache(maxsize=1)
def _top_k_page_table_transform():
    _, reason = flashinfer_kernel_support(_FEATURE)
    try:
        module = importlib.import_module("flashinfer.topk")
        callable_ = getattr(module, "top_k_page_table_transform")
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            _FEATURE,
            f"failed to load public API: {type(error).__name__}: {error}",
        ) from error
    if not callable(callable_):
        raise ExternalKernelContractError(
            "flashinfer-python",
            _FEATURE,
            "flashinfer.topk.top_k_page_table_transform is not callable",
        )
    try:
        actual = frozenset(inspect.signature(callable_).parameters)
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            _FEATURE,
            f"failed to inspect public API: {type(error).__name__}: {error}",
        ) from error
    missing = sorted(_REQUIRED_PARAMETERS - actual)
    if missing:
        raise ExternalKernelContractError(
            "flashinfer-python",
            _FEATURE,
            f"public API is missing required parameters {missing}",
        )
    return callable_, reason


def flashinfer_top_k_page_table_transform_support() -> tuple[bool, str]:
    _, reason = _top_k_page_table_transform()
    return True, reason


def flashinfer_top_k_page_table_transform(
    scores: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
    *,
    cuda_graph: bool,
) -> torch.Tensor:
    if scores.ndim != 2 or not scores.is_contiguous():
        raise ValueError("FlashInfer fused top-k requires contiguous rank-2 scores.")
    if scores.dtype not in {torch.float32, torch.float16, torch.bfloat16}:
        raise TypeError(
            "FlashInfer fused top-k requires FP32, FP16, or BF16 scores, got "
            f"{scores.dtype}."
        )
    if (
        page_table.shape != scores.shape
        or page_table.dtype != torch.int32
        or not page_table.is_contiguous()
    ):
        raise TypeError(
            "FlashInfer fused top-k requires a contiguous int32 page table "
            f"matching scores, got {tuple(page_table.shape)}/{page_table.dtype}."
        )
    if lengths.shape != (int(scores.shape[0]),) or lengths.dtype != torch.int32:
        raise TypeError(
            "FlashInfer fused top-k requires one int32 length per score row."
        )
    if not lengths.is_contiguous():
        raise ValueError("FlashInfer fused top-k lengths must be contiguous.")
    if (
        not scores.is_cuda
        or page_table.device != scores.device
        or lengths.device != scores.device
    ):
        raise ValueError("FlashInfer fused top-k inputs must share one CUDA device.")
    k = int(k)
    if not 0 < k <= int(scores.shape[1]):
        raise ValueError(
            f"FlashInfer fused top-k requires 0 < k <= {scores.shape[1]}, got {k}."
        )

    callable_, _ = _top_k_page_table_transform()
    output = callable_(
        scores,
        page_table,
        lengths,
        k,
        deterministic=True,
        tie_break=1,
        dsa_graph_safe=bool(cuda_graph),
    )
    if (
        output.shape != (int(scores.shape[0]), k)
        or output.dtype != torch.int32
        or output.device != scores.device
        or not output.is_contiguous()
    ):
        raise RuntimeError(
            "FlashInfer fused top-k returned an invalid page-table result: "
            f"shape={tuple(output.shape)} dtype={output.dtype}."
        )
    return output


__all__ = [
    "flashinfer_top_k_page_table_transform",
    "flashinfer_top_k_page_table_transform_support",
]
