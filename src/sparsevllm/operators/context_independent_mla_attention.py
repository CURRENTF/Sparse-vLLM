"""Experimental context-independent GLM MLA provider variants."""

from __future__ import annotations

import torch

from sparsevllm.kernels.triton.mla import (
    MlaDecodeLaunchConfig,
    select_glm_mla_decode_config,
)
from sparsevllm.operators.mla_attention import (
    MlaAttentionProvider,
    MlaSglFa3Provider,
    MlaTileLangScoreProvider,
    MlaTritonProvider,
)


def select_context_independent_mla_config(
    *,
    batch_size: int,
    local_q_heads: int,
) -> MlaDecodeLaunchConfig:
    """Select from static model and batch dimensions only.

    TP2 small/medium schedules already depend only on batch size.  Larger
    batches always take the measured long-context schedule so replay never
    changes topology when a request crosses a context bucket.
    """
    return select_glm_mla_decode_config(
        batch_size=batch_size,
        max_context_len=8193,
        local_q_heads=local_q_heads,
    )


class _ContextIndependentMlaMixin:
    cuda_graph_context_independent = True

    def _launch_config_for(
        self,
        *,
        batch_size: int,
        max_context_len: int | None,
        active_slot_width: int,
    ) -> MlaDecodeLaunchConfig:
        del max_context_len, active_slot_width
        fixed = getattr(self, "_fixed_launch_config", None)
        if fixed is not None:
            return fixed
        return select_context_independent_mla_config(
            batch_size=batch_size,
            local_q_heads=self.spec.local_q_heads,
        )


class ContextIndependentMlaTritonProvider(
    _ContextIndependentMlaMixin,
    MlaTritonProvider,
):
    name = "triton_sm90_context_independent"


class ContextIndependentMlaSglFa3Provider(
    _ContextIndependentMlaMixin,
    MlaSglFa3Provider,
):
    name = "sgl_fa3_sm90_context_independent"


class ContextIndependentMlaTileLangScoreProvider(
    _ContextIndependentMlaMixin,
    MlaTileLangScoreProvider,
):
    name = "tilelang_score_sgl_fa3_h100_context_independent"


def bind_context_independent_mla_attention(model: torch.nn.Module) -> str | None:
    """Replace the selected stable MLA provider with its isolated variant."""
    model_body = getattr(model, "model", None)
    mla_attention = getattr(model_body, "mla_attention", None)
    if mla_attention is None:
        return None
    provider = getattr(mla_attention, "provider", None)
    provider_type: type[MlaAttentionProvider]
    if isinstance(provider, MlaTileLangScoreProvider):
        provider_type = ContextIndependentMlaTileLangScoreProvider
    elif isinstance(provider, MlaSglFa3Provider):
        provider_type = ContextIndependentMlaSglFa3Provider
    elif isinstance(provider, MlaTritonProvider):
        provider_type = ContextIndependentMlaTritonProvider
    else:
        return None
    replacement = provider_type(
        op_spec=provider.spec,
        device=provider.device,
        max_batch_size=provider.max_batch_size,
    )
    mla_attention.provider = replacement
    return replacement.name


__all__ = [
    "ContextIndependentMlaSglFa3Provider",
    "ContextIndependentMlaTileLangScoreProvider",
    "ContextIndependentMlaTritonProvider",
    "bind_context_independent_mla_attention",
    "select_context_independent_mla_config",
]
