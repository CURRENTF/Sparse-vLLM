from __future__ import annotations

from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextMLP as HFGemma4MLP,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextModel as HFGemma4Model,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextRotaryEmbedding as HFGemma4RotaryEmbedding,
)
from transformers.models.gemma4.modeling_gemma4 import (
    Gemma4TextRouter as HFGemma4Router,
)

from sparsevllm.configs.sparse import normalize_sparse_methods
from sparsevllm.distributed import ParallelContext, ParallelGroup, ParallelMode
from sparsevllm.method_registry import MODEL_RUNTIME_COMPATIBILITY
from sparsevllm.models.gemma4 import (
    Gemma4Attention,
    Gemma4MLP,
    Gemma4Model,
    Gemma4RotaryEmbedding,
    Gemma4Router,
)
from sparsevllm.models.layout import RuntimeLayout
from sparsevllm.operators.activation import TorchGeluTanhAndMulProvider


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gemma4_router_kernels_match_torch():
    from sparsevllm.kernels.triton.gemma4_router import (
        gemma4_router_input,
        gemma4_router_topk,
    )

    torch.manual_seed(19)
    hidden = torch.randn(7, 2816, dtype=torch.bfloat16, device="cuda")
    scale = torch.randn(2816, dtype=torch.bfloat16, device="cuda")
    actual_input = gemma4_router_input(hidden, scale, 2816**-0.5, 1e-6)
    variance = hidden.float().square().mean(-1, keepdim=True)
    expected_input = (hidden.float() * torch.rsqrt(variance + 1e-6)).to(torch.bfloat16)
    expected_input = (expected_input * scale * 2816**-0.5).to(torch.bfloat16)
    torch.testing.assert_close(actual_input, expected_input, rtol=0, atol=0)

    logits = torch.randn(7, 128, dtype=torch.bfloat16, device="cuda")
    expert_scale = torch.randn(128, dtype=torch.bfloat16, device="cuda")
    actual_weights, actual_ids = gemma4_router_topk(logits, expert_scale, 8)
    probabilities = logits.float().softmax(-1)
    expected_weights, expected_ids = probabilities.topk(8, dim=-1)
    expected_weights.div_(expected_weights.sum(-1, keepdim=True)).mul_(
        expert_scale[expected_ids]
    )
    assert torch.equal(actual_ids, expected_ids)
    torch.testing.assert_close(actual_weights, expected_weights, rtol=1e-5, atol=1e-6)


def _parallel_context() -> ParallelContext:
    group = ParallelGroup(process_group=None, ranks=(0,), rank=0, size=1)
    return ParallelContext(world=group, tensor=group, expert=group, data=group)


def _patch_parallel_context():
    stack = ExitStack()
    context = _parallel_context()
    for target in (
        "sparsevllm.models.gemma4.get_parallel_context",
        "sparsevllm.layers.linear.get_parallel_context",
        "sparsevllm.layers.embed_head.get_parallel_context",
    ):
        stack.enter_context(patch(target, return_value=context))
    return stack


def _config(**overrides) -> Gemma4TextConfig:
    values = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "global_head_dim": 8,
        "num_global_key_value_heads": 1,
        "max_position_embeddings": 32,
        "layer_types": ["sliding_attention", "full_attention"],
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {
                "rope_type": "proportional",
                "rope_theta": 1000000.0,
                "partial_rotary_factor": 0.25,
            },
        },
        "sliding_window": 4,
        "hidden_size_per_layer_input": 0,
        "final_logit_softcapping": 30.0,
    }
    values.update(overrides)
    return Gemma4TextConfig(**values)


def test_gemma4_rope_matches_transformers_for_both_layer_types():
    config = _config()
    positions = torch.arange(9)
    for layer_type, head_dim in (("sliding_attention", 4), ("full_attention", 8)):
        actual = Gemma4RotaryEmbedding(config, layer_type, head_dim)
        reference = HFGemma4RotaryEmbedding(config, layer_type=layer_type)
        cos, sin = reference(torch.zeros(1), positions.unsqueeze(0), layer_type)
        torch.testing.assert_close(
            actual.cos_sin_cache[positions, 0, : head_dim // 2],
            cos[0, :, : head_dim // 2],
        )
        torch.testing.assert_close(
            actual.cos_sin_cache[positions, 0, head_dim // 2 :],
            sin[0, :, : head_dim // 2],
        )


def test_gemma4_dense_mlp_matches_transformers():
    config = _config()
    with _patch_parallel_context():
        actual = Gemma4MLP(config, 0, TorchGeluTanhAndMulProvider())
    reference = HFGemma4MLP(config, 0)
    torch.manual_seed(3)
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.1)
    actual.gate_up_proj.weight_loader(
        actual.gate_up_proj.weight, reference.gate_proj.weight, 0
    )
    actual.gate_up_proj.weight_loader(
        actual.gate_up_proj.weight, reference.up_proj.weight, 1
    )
    actual.down_proj.weight_loader(actual.down_proj.weight, reference.down_proj.weight)
    hidden_states = torch.randn(7, config.hidden_size)
    torch.testing.assert_close(
        actual(hidden_states), reference(hidden_states), atol=1e-6, rtol=1e-5
    )


def test_gemma4_router_matches_transformers():
    config = _config(
        enable_moe_block=True, num_experts=4, top_k_experts=2, moe_intermediate_size=4
    )
    with _patch_parallel_context():
        actual = Gemma4Router(config)
    reference = HFGemma4Router(config)
    torch.manual_seed(5)
    reference.proj.weight.data.normal_(0, 0.2)
    reference.scale.data.normal_(1, 0.1)
    reference.per_expert_scale.data.normal_(1, 0.1)
    actual.load_state_dict(reference.state_dict())
    hidden_states = torch.randn(11, config.hidden_size)
    _, expected_weights, expected_ids = reference(hidden_states)
    weights, ids = actual(hidden_states)
    torch.testing.assert_close(weights, expected_weights)
    assert torch.equal(ids, expected_ids)


def test_gemma4_k_eq_v_loader_duplicates_normalized_projection_slot():
    config = _config(attention_k_eq_v=True)
    with _patch_parallel_context():
        attention = Gemma4Attention(
            config,
            1,
            Gemma4RotaryEmbedding(config, "full_attention", config.global_head_dim),
        )
    loaded_key = torch.randn(
        config.num_global_key_value_heads * config.global_head_dim, config.hidden_size
    )
    attention.qkv_proj.weight_loader(attention.qkv_proj.weight, loaded_key, "k")
    q_end = attention.q_size
    key = attention.qkv_proj.weight[q_end : q_end + attention.kv_size]
    value = attention.qkv_proj.weight[q_end + attention.kv_size :]
    torch.testing.assert_close(key, loaded_key)
    torch.testing.assert_close(value, loaded_key)


def test_gemma4_ple_matches_transformers():
    config = _config(
        hidden_size_per_layer_input=2,
        vocab_size_per_layer_input=32,
    )
    with _patch_parallel_context():
        actual = Gemma4Model(config, TorchGeluTanhAndMulProvider())
    reference = HFGemma4Model(config)
    torch.manual_seed(11)
    for parameter in reference.parameters():
        parameter.data.normal_(0, 0.1)
    actual.embed_tokens.weight.data.copy_(reference.embed_tokens.weight)
    actual.embed_tokens_per_layer.weight.data.copy_(
        reference.embed_tokens_per_layer.weight
    )
    actual.per_layer_model_projection.weight.data.copy_(
        reference.per_layer_model_projection.weight
    )
    actual.per_layer_projection_norm.load_state_dict(
        reference.per_layer_projection_norm.state_dict()
    )
    actual.per_layer_projection_norm._ops = SimpleNamespace(
        rmsnorm=lambda x, weight, eps: (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + eps)
            * weight.float()
        ).to(x.dtype)
    )
    input_ids = torch.tensor([1, 7, 3, 9])
    actual_hidden = actual.embed_tokens(input_ids) * actual.embedding_scale
    reference_hidden = reference.embed_tokens(input_ids)
    expected = reference.project_per_layer_inputs(
        reference_hidden,
        reference.get_per_layer_inputs(input_ids, reference_hidden),
    )
    torch.testing.assert_close(
        actual.get_per_layer_inputs(input_ids, actual_hidden),
        expected,
    )


def test_gemma4_shared_kv_layout_aliases_last_source_by_type():
    config = _config(
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    layout = RuntimeLayout.from_config(config)
    assert layout.num_kv_layers == 2
    assert layout.kv_idx_to_layer_idx == (0, 1)
    assert layout.layer_idx_to_kv_idx == (0, 1, 0, 1)
    assert layout.kv_num_heads == (2, 2)
    assert layout.kv_head_dims == (4, 8)


def test_gemma4_shared_kv_attention_only_allocates_query_projection():
    config = _config(
        num_hidden_layers=4,
        num_kv_shared_layers=2,
        layer_types=[
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "full_attention",
        ],
    )
    with _patch_parallel_context():
        attention = Gemma4Attention(
            config,
            2,
            Gemma4RotaryEmbedding(config, "sliding_attention", config.head_dim),
        )
    assert attention.is_kv_shared_layer
    assert tuple(attention.qkv_proj.weight.shape) == (
        config.num_attention_heads * config.head_dim,
        config.hidden_size,
    )
    assert not hasattr(attention, "k_norm")
    assert not hasattr(attention, "v_norm")


def test_gemma4_sparse_registry_keeps_dedicated_validated_methods():
    compatibility = MODEL_RUNTIME_COMPATIBILITY[("gemma4", ParallelMode.STANDARD)]
    assert compatibility.sparse_methods == {"", "streamingllm", "omnikv"}
    assert compatibility.decode_cuda_graph_methods == compatibility.sparse_methods


def test_gemma4_shared_kv_rejects_per_layer_streaming_eviction():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="gemma4_text",
            num_kv_shared_layers=18,
        ),
        vllm_sparse_method="streamingllm",
    )
    with pytest.raises(NotImplementedError, match="KV-sharing"):
        normalize_sparse_methods(config)
