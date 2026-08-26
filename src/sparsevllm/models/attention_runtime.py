from __future__ import annotations

import torch
from torch import nn

from sparsevllm.method_registry import (
    normalize_sparse_method,
    sparse_decode_attention_requires_scores,
    sparse_prefill_attention_contract,
)
from sparsevllm.operators.decode_attention import (
    DecodeAttentionOpSpec,
    PreparedDecodeAttentionOp,
    prepare_decode_attention_op,
)
from sparsevllm.operators.full_attention import (
    FullAttentionOpSpec,
    FullAttentionProvider,
    prepare_full_attention_provider,
)
from sparsevllm.operators.moe import model_activation_dtype
from sparsevllm.operators.prefill_attention import (
    PreparedPrefillAttentionOp,
    PrefillAttentionOpSpec,
    prepare_prefill_attention_op,
)


def resolve_mha_head_dim(config) -> int:
    explicit = getattr(config, "head_dim", None)
    if explicit is not None:
        head_dim = int(explicit)
        if head_dim <= 0:
            raise ValueError(f"MHA head_dim must be positive, got {head_dim}.")
        return head_dim

    hidden_size = int(config.hidden_size)
    num_attention_heads = int(config.num_attention_heads)
    if num_attention_heads <= 0 or hidden_size % num_attention_heads:
        raise ValueError(
            "Cannot infer MHA head_dim from hidden_size / num_attention_heads: "
            f"hidden_size={hidden_size} num_attention_heads={num_attention_heads}."
        )
    return hidden_size // num_attention_heads


def _resolve_mha_local_shape(
    config,
    *,
    attention_tp_size: int,
) -> tuple[int, int, int, torch.dtype]:
    tp_size = int(attention_tp_size)
    query_heads = int(config.num_attention_heads)
    kv_heads = int(config.num_key_value_heads)
    if query_heads % tp_size or kv_heads % tp_size:
        raise ValueError(
            "MHA query and KV heads must be divisible by attention TP size: "
            f"query_heads={query_heads} kv_heads={kv_heads} tp_size={tp_size}."
        )
    return (
        query_heads // tp_size,
        kv_heads // tp_size,
        resolve_mha_head_dim(config),
        model_activation_dtype(config),
    )


def build_mha_prefill_attention_spec(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    runtime_config=None,
) -> PrefillAttentionOpSpec:
    query_heads, kv_heads, head_dim, activation_dtype = _resolve_mha_local_shape(
        config,
        attention_tp_size=attention_tp_size,
    )
    normalized_method = normalize_sparse_method(sparse_method)
    score_config = config if runtime_config is None else runtime_config
    contract = sparse_prefill_attention_contract(
        normalized_method,
        sparse_prefill_score_mode=getattr(
            score_config, "sparse_prefill_score_mode", "probability"
        ),
        h2o_prefill_score_window=getattr(
            score_config, "h2o_prefill_score_window", 0
        ),
    )
    return PrefillAttentionOpSpec(
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        activation_dtype=activation_dtype,
        softmax_scale=head_dim**-0.5,
        causal=True,
        page_size=1,
        score_output=contract.main_score_kind,
        layer_varying_page_table=bool(normalized_method),
        return_softmax_lse=(
            normalized_method == "h2o"
            and getattr(score_config, "sparse_prefill_score_mode", "probability")
            == "probability"
        ),
        allow_softmax_lse_fallback=(
            normalized_method == "h2o"
            and getattr(score_config, "sparse_prefill_score_mode", "probability")
            == "probability"
        ),
    )


def build_mha_prefill_attention_op(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    device: torch.device,
    runtime_config=None,
) -> PreparedPrefillAttentionOp:
    return prepare_prefill_attention_op(
        build_mha_prefill_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            runtime_config=runtime_config,
        ),
        device_index=int(device.index or 0),
    )


def build_mha_decode_attention_spec(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    max_batch_size: int,
    cuda_graph: bool,
    runtime_config=None,
) -> DecodeAttentionOpSpec:
    query_heads, kv_heads, head_dim, activation_dtype = _resolve_mha_local_shape(
        config,
        attention_tp_size=attention_tp_size,
    )
    normalized_method = normalize_sparse_method(sparse_method)
    requires_decode_scores = sparse_decode_attention_requires_scores(
        normalized_method
    )
    return DecodeAttentionOpSpec(
        num_query_heads=query_heads,
        num_kv_heads=kv_heads,
        head_dim=head_dim,
        activation_dtype=activation_dtype,
        softmax_scale=head_dim**-0.5,
        max_batch_size=int(max_batch_size),
        causal=True,
        page_size=1,
        # Score demand can change between decode steps for sparse methods.
        # Bind the score-capable implementation up front instead of
        # switching providers in the runtime path.
        may_require_attention_scores=requires_decode_scores,
        layer_varying_page_table=bool(normalized_method),
        cuda_graph=bool(cuda_graph),
        h2o_layerwise_probability_scores=(
            normalized_method == "h2o" and requires_decode_scores
        ),
        batch_only_cuda_graph=(
            bool(cuda_graph)
            and str(
                getattr(
                    runtime_config,
                    "decode_graph_shape_policy",
                    "batch_only",
                )
            )
            == "batch_only"
        ),
        context_capacity=int(getattr(runtime_config, "max_model_len", 0) or 0)
        or None,
        may_use_full_layer_kivi_int4=(
            normalized_method == "deltakv"
            and int(
                getattr(runtime_config, "full_layer_kv_quant_bits", 0) or 0
            )
            == 4
            and bool(
                getattr(runtime_config, "enable_full_layer_kivi_quant", True)
            )
        ),
        full_layer_kivi_decode_block_seq=int(
            getattr(runtime_config, "full_layer_kivi_decode_block_seq", 256)
            or 256
        ),
        full_layer_kivi_decode_block_n=int(
            getattr(runtime_config, "full_layer_kivi_decode_block_n", 16) or 16
        ),
        full_layer_kivi_decode_num_warps=int(
            getattr(runtime_config, "full_layer_kivi_decode_num_warps", 2) or 2
        ),
        full_layer_kivi_decode_num_stages=int(
            getattr(runtime_config, "full_layer_kivi_decode_num_stages", 3) or 3
        ),
    )


def build_mha_decode_attention_op(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    device: torch.device,
    max_batch_size: int,
    cuda_graph: bool,
) -> PreparedDecodeAttentionOp:
    return prepare_decode_attention_op(
        build_mha_decode_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            max_batch_size=max_batch_size,
            cuda_graph=cuda_graph,
        ),
        device_index=int(device.index or 0),
    )


def build_mha_full_attention_provider(
    config,
    *,
    sparse_method: str | None,
    attention_tp_size: int,
    device: torch.device,
    max_batch_size: int,
    cuda_graph: bool,
    runtime_config=None,
) -> FullAttentionProvider:
    spec = FullAttentionOpSpec(
        prefill=build_mha_prefill_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            runtime_config=runtime_config,
        ),
        decode=build_mha_decode_attention_spec(
            config,
            sparse_method=sparse_method,
            attention_tp_size=attention_tp_size,
            max_batch_size=max_batch_size,
            cuda_graph=cuda_graph,
            runtime_config=runtime_config,
        ),
    )
    return prepare_full_attention_provider(
        spec,
        device_index=int(device.index or 0),
    )


def bind_mha_full_attention_provider(
    model: nn.Module,
    provider: FullAttentionProvider,
) -> int:
    return provider.bind(model)


__all__ = [
    "bind_mha_full_attention_provider",
    "build_mha_decode_attention_op",
    "build_mha_decode_attention_spec",
    "build_mha_full_attention_provider",
    "build_mha_prefill_attention_op",
    "build_mha_prefill_attention_spec",
    "resolve_mha_head_dim",
]
