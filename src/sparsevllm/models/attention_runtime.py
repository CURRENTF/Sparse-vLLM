from __future__ import annotations

import torch
from torch import nn

from sparsevllm.layers.attention import Attention
from sparsevllm.method_registry import (
    normalize_sparse_method,
    sparse_prefill_attention_contract,
)
from sparsevllm.operators.moe import model_activation_dtype
from sparsevllm.operators.prefill_attention import (
    PreparedPrefillAttentionOp,
    PrefillAttentionOpSpec,
    prepare_prefill_attention_op,
)


def build_mha_prefill_attention_op(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    device: torch.device,
) -> PreparedPrefillAttentionOp:
    tp_size = int(attention_tp_size)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    if query_heads % tp_size or kv_heads % tp_size:
        raise ValueError(
            "MHA query and KV heads must be divisible by attention TP size: "
            f"query_heads={query_heads} kv_heads={kv_heads} tp_size={tp_size}."
        )
    head_dim = int(config.head_dim)
    normalized_method = normalize_sparse_method(sparse_method)
    contract = sparse_prefill_attention_contract(normalized_method)
    return prepare_prefill_attention_op(
        PrefillAttentionOpSpec(
            num_query_heads=query_heads // tp_size,
            num_kv_heads=kv_heads // tp_size,
            head_dim=head_dim,
            activation_dtype=model_activation_dtype(config),
            softmax_scale=head_dim**-0.5,
            causal=True,
            page_size=1,
            score_output=contract.main_score_kind,
            layer_varying_page_table=bool(normalized_method),
        ),
        device_index=int(device.index or 0),
    )


def bind_mha_prefill_attention_op(
    model: nn.Module,
    prefill_attention_op: PreparedPrefillAttentionOp,
) -> int:
    attention_layers = [
        module for module in model.modules() if isinstance(module, Attention)
    ]
    if not attention_layers:
        raise ValueError("Cannot bind MHA prefill operator: model has no Attention layers.")
    for attention in attention_layers:
        attention.prefill_op = prefill_attention_op
    return len(attention_layers)


__all__ = [
    "bind_mha_prefill_attention_op",
    "build_mha_prefill_attention_op",
]
