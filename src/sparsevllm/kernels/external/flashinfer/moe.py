from __future__ import annotations

import importlib
from functools import lru_cache

import torch

from sparsevllm.kernels.external.flashinfer.support import (
    flashinfer_kernel_support,
)
from sparsevllm.kernels.external.support import ExternalKernelContractError


@lru_cache(maxsize=1)
def _cutlass_fp8_moe_op():
    feature = "SM90 CUTLASS FP8 MoE"
    _, reason = flashinfer_kernel_support(feature)
    try:
        function = getattr(
            importlib.import_module("flashinfer.fused_moe"),
            "cutlass_fused_moe",
        )
        activation_type = getattr(
            importlib.import_module("flashinfer.tllm_enums").ActivationType,
            "Swiglu",
        )
    except Exception as error:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"failed to load: {type(error).__name__}: {error}",
        ) from error
    if not callable(function):
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            "flashinfer.fused_moe.cutlass_fused_moe is not callable",
        )
    return function, activation_type, reason


def flashinfer_cutlass_fp8_moe_support() -> tuple[bool, str]:
    _, _, reason = _cutlass_fp8_moe_op()
    return True, reason


def flashinfer_cutlass_fused_moe(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w13_scale_inv: torch.Tensor,
    w2_scale_inv: torch.Tensor,
    *,
    ep_size: int,
    ep_rank: int,
    output: torch.Tensor,
) -> None:
    function, activation_type, _ = _cutlass_fp8_moe_op()
    function(
        hidden_states,
        topk_ids.to(dtype=torch.int32),
        topk_weights.to(dtype=torch.float32),
        w13_weight,
        w2_weight,
        hidden_states.dtype,
        quant_scales=[w13_scale_inv, w2_scale_inv],
        ep_size=int(ep_size),
        ep_rank=int(ep_rank),
        output=output,
        use_deepseek_fp8_block_scale=True,
        use_fused_finalize=False,
        enable_pdl=None,
        activation_type=activation_type,
    )


__all__ = [
    "flashinfer_cutlass_fp8_moe_support",
    "flashinfer_cutlass_fused_moe",
]
