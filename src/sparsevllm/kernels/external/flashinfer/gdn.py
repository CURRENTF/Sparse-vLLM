from __future__ import annotations

import importlib
import inspect
import re
from functools import lru_cache

import torch

from sparsevllm.kernels.external.flashinfer.support import (
    flashinfer_kernel_health,
    flashinfer_kernel_support,
)
from sparsevllm.kernels.external.support import ExternalKernelContractError


_GDN_PREFILL_ARGUMENTS = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "scale",
    "initial_state",
    "output_final_state",
    "cu_seqlens",
    "use_qk_l2norm_in_kernel",
    "output",
    "output_state",
    "state_checkpoints",
    "checkpoint_cu_starts",
    "checkpoint_every_n_tokens",
    "use_cp",
    "state_indices",
)


@lru_cache(maxsize=1)
def _gdn_prefill_op():
    feature = "SM90 GDN prefill"
    _, reason = flashinfer_kernel_support(feature)
    try:
        function = getattr(
            importlib.import_module("flashinfer.gdn_prefill"),
            "chunk_gated_delta_rule",
        )
        actual_arguments = tuple(inspect.signature(function).parameters)
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
            "flashinfer.gdn_prefill.chunk_gated_delta_rule is not callable",
        )
    if actual_arguments != _GDN_PREFILL_ARGUMENTS:
        raise ExternalKernelContractError(
            "flashinfer-python",
            feature,
            f"unsupported schema: {actual_arguments}",
        )
    return function, reason


def flashinfer_sm90_gdn_prefill_support() -> tuple[bool, str]:
    health = flashinfer_kernel_health()
    if not health.ready:
        # A broken or absent package family is an environment error. Keep that
        # distinct from a healthy package version that predates this feature.
        flashinfer_kernel_support("SM90 GDN prefill")
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(health.version))
    parsed = tuple(map(int, match.groups())) if match else None
    if parsed is None or parsed < (0, 6, 17):
        return False, (
            "requires flashinfer-python >= 0.6.17 for the validated GDN "
            f"contract, got {health.version}"
        )
    _, reason = _gdn_prefill_op()
    return True, reason


def flashinfer_chunk_gated_delta_rule_sm90(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g_exp: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    function, _ = _gdn_prefill_op()
    result = function(
        q=q,
        k=k,
        v=v,
        g=g_exp,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
    )
    if not isinstance(result, tuple) or len(result) != 2:
        raise RuntimeError(
            "FlashInfer GDN prefill violated its output contract: expected "
            "(output, final_state)."
        )
    return result


__all__ = [
    "flashinfer_chunk_gated_delta_rule_sm90",
    "flashinfer_sm90_gdn_prefill_support",
]
