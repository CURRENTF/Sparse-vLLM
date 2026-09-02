from __future__ import annotations

import inspect
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn.functional as F
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
from sparsevllm.distributed import ParallelContext, ParallelGroup
from sparsevllm.engine.cache_manager.base import ExplicitKVPayload
from sparsevllm.models.gemma4 import (
    Gemma4Attention,
    Gemma4ForCausalLM,
    Gemma4MLP,
    Gemma4Model,
    Gemma4RotaryEmbedding,
    Gemma4Router,
)
from sparsevllm.models.layout import RuntimeLayout
from sparsevllm.operators.gemma4 import (
    Gemma4OpSpec,
    TorchGemma4OperatorProvider,
    TritonGemma4OperatorProvider,
)
from sparsevllm.operators.gemma4_attention import (
    Gemma4FlashInferPrefill,
    _FlashInferState,
)
from sparsevllm.operators.gemma4_moe import (
    GEMMA4_MOE_REGISTRY,
    TorchGemma4MoeProvider,
)
from sparsevllm.operators.gemma4_router import (
    H20Gemma4RouterProvider,
    TorchGemma4RouterProvider,
    TritonGemma4RouterProvider,
)
from sparsevllm.utils.config import config_layer_get
from sparsevllm.utils.context import reset_context, set_context


def test_gemma4_flashinfer_prefill_caches_alternating_attention_contracts():
    prefill = Gemma4FlashInferPrefill()
    wrappers = [Mock(), Mock()]
    prefill._available_states = [
        _FlashInferState(wrapper=wrapper, workspace=torch.empty(0))
        for wrapper in wrappers
    ]
    meta = SimpleNamespace(
        active_slots=torch.arange(4, dtype=torch.int32).view(1, -1),
        req_indices=torch.zeros(1, dtype=torch.int32),
        context_lens=torch.tensor([4], dtype=torch.int32),
        attn_score=None,
    )
    cases = (
        (
            torch.empty(2, 4, 256),
            SimpleNamespace(
                payload=ExplicitKVPayload(
                    torch.empty(4, 2, 256),
                    torch.empty(4, 2, 256),
                ),
                meta=meta,
            ),
            4,
        ),
        (
            torch.empty(2, 4, 512),
            SimpleNamespace(
                payload=ExplicitKVPayload(
                    torch.empty(4, 1, 512),
                    torch.empty(4, 1, 512),
                ),
                meta=meta,
            ),
            None,
        ),
    )
    reset_context()
    set_context(True, cu_seqlens_q=torch.tensor([0, 2], dtype=torch.int32))
    try:
        for index in range(4):
            query, view, window = cases[index % 2]
            prefill.run(
                query,
                view,
                q_start=torch.zeros(1, dtype=torch.int32),
                chunk_lens=torch.tensor([2], dtype=torch.int32),
                max_context_len=4,
                sliding_window=window,
            )
    finally:
        prefill.close()
        reset_context()

    assert sum(wrapper.plan.call_count for wrapper in wrappers) == 2
    assert all(wrapper.plan.call_count == 1 for wrapper in wrappers)


def test_gemma4_runtime_spec_counts_alternating_attention_contracts():
    config = _config()
    engine_config = SimpleNamespace(decode_graph=False)
    captured = {}

    def resolve(spec, *, device_index):
        captured["spec"] = spec
        captured["device_index"] = device_index
        return Mock(name="provider")

    with patch(
        "sparsevllm.models.gemma4.resolve_gemma4_provider",
        side_effect=resolve,
    ):
        Gemma4ForCausalLM.build_runtime_kwargs(
            config,
            engine_config=engine_config,
            parallel_context=_parallel_context(),
            device=torch.device("cuda", 0),
        )

    spec = captured["spec"]
    assert isinstance(spec, Gemma4OpSpec)
    assert spec.head_dims == (4, 8)
    assert len(spec.attention_contracts) == 2
    assert {contract[2:] for contract in spec.attention_contracts} == {
        (4, 3),
        (8, -1),
    }
    assert captured["device_index"] == 0


def test_gemma4_close_runtime_operators_releases_provider():
    operator_provider = Mock(name="operator_provider")
    router_provider = Mock(name="router_provider")
    model = SimpleNamespace(
        operator_provider=operator_provider,
        router_provider=router_provider,
    )

    Gemma4ForCausalLM.close_runtime_operators(model)

    operator_provider.close.assert_called_once_with()
    router_provider.close.assert_called_once_with()


def test_gemma4_runtime_closes_operator_if_router_prepare_fails():
    config = _config(enable_moe_block=True, num_experts=4, top_k_experts=2)
    operator_provider = Mock(name="operator_provider")
    with (
        patch(
            "sparsevllm.models.gemma4.resolve_gemma4_provider",
            return_value=operator_provider,
        ),
        patch(
            "sparsevllm.models.gemma4.resolve_gemma4_router_provider",
            side_effect=RuntimeError("router prepare failed"),
        ),
        pytest.raises(RuntimeError, match="router prepare failed"),
    ):
        Gemma4ForCausalLM.build_runtime_kwargs(
            config,
            engine_config=SimpleNamespace(decode_graph=False),
            parallel_context=_parallel_context(),
            device=torch.device("cuda", 0),
        )

    operator_provider.close.assert_called_once_with()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gemma4_router_kernels_match_torch():
    from sparsevllm.kernels.triton.gemma4_fused_router import (
        gemma4_fused_router_topk,
    )
    from sparsevllm.kernels.triton.gemma4_router import (
        gemma4_router_input,
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
    actual_weights, actual_ids = gemma4_fused_router_topk(logits, expert_scale, 8)
    probabilities = logits.float().softmax(-1)
    expected_weights, expected_ids = probabilities.topk(8, dim=-1)
    expected_weights.div_(expected_weights.sum(-1, keepdim=True)).mul_(
        expert_scale[expected_ids]
    )
    actual_routes = torch.zeros_like(probabilities).scatter_(
        1, actual_ids.long(), actual_weights
    )
    expected_routes = torch.zeros_like(probabilities).scatter_(
        1, expected_ids, expected_weights
    )
    torch.testing.assert_close(actual_routes, expected_routes, rtol=1e-5, atol=1e-6)
    assert actual_ids.dtype == torch.int32

    for _ in range(3):
        gemma4_fused_router_topk(logits, expert_scale, 8)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids = gemma4_fused_router_topk(logits, expert_scale, 8)
    logits.copy_(torch.randn_like(logits))
    graph.replay()
    replay_weights, replay_ids = graph_weights.clone(), graph_ids.clone()
    graph.replay()
    assert torch.equal(replay_ids, graph_ids)
    assert torch.equal(replay_weights, graph_weights)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("rows", [1, 256, 257])
def test_gemma4_provider_gelu_tanh_and_mul_matches_torch(dtype, rows):
    torch.manual_seed(20260813)
    x = torch.randn(rows, 1408, dtype=dtype, device="cuda")
    gate, up = x.chunk(2, -1)
    expected = F.gelu(gate, approximate="tanh") * up
    actual_input = x.clone()
    actual = TritonGemma4OperatorProvider().gelu_tanh_and_mul(actual_input)
    torch.testing.assert_close(actual, expected, rtol=3e-3, atol=3e-3)
    assert actual.data_ptr() == actual_input.data_ptr()


def test_gemma4_moe_torch_oracle_is_not_a_production_fallback():
    assert TorchGemma4MoeProvider not in GEMMA4_MOE_REGISTRY.providers


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("num_experts", [1024, 1025])
def test_gemma4_h20_router_expert_boundary_matches_torch(num_experts):
    provider_type = (
        H20Gemma4RouterProvider
        if num_experts == 1024
        else TritonGemma4RouterProvider
    )
    torch.manual_seed(41)
    logits = torch.randn(3, num_experts, dtype=torch.bfloat16, device="cuda")
    scales = torch.randn(num_experts, dtype=torch.bfloat16, device="cuda")
    weights, ids = provider_type().topk(logits, scales, 8)
    probabilities = logits.float().softmax(-1)
    expected_weights, expected_ids = probabilities.topk(8, dim=-1)
    expected_weights.div_(expected_weights.sum(-1, keepdim=True)).mul_(
        scales[expected_ids]
    )
    actual = torch.zeros_like(probabilities).scatter_(1, ids.long(), weights)
    expected = torch.zeros_like(probabilities).scatter_(
        1, expected_ids, expected_weights
    )
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


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


def test_gemma4_skips_multimodal_weights_when_disabled():
    model = SimpleNamespace(multimodal_encoder=None)

    assert Gemma4ForCausalLM.map_weight_name(
        model, "model.vision_tower.encoder.layers.0.weight"
    ) is None


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


def test_gemma4_serialized_layer_config_uses_global_defaults():
    config = {
        "head_dim": 256,
        "num_key_value_heads": 8,
        "per_layer_config": {"05": {"head_dim": 512}},
    }
    assert config_layer_get(config, 0, "head_dim") == 256
    assert config_layer_get(config, 5, "head_dim") == 512
    assert config_layer_get(config, 5, "num_key_value_heads") == 8


def test_gemma4_rope_matches_transformers_for_both_layer_types():
    config = _config()
    positions = torch.arange(9)
    for layer_idx, (layer_type, head_dim) in enumerate(
        (("sliding_attention", 4), ("full_attention", 8))
    ):
        actual = Gemma4RotaryEmbedding(config, layer_type, head_dim)
        if "layer_type" in inspect.signature(HFGemma4RotaryEmbedding).parameters:
            reference = HFGemma4RotaryEmbedding(config, layer_type=layer_type)
            cos, sin = reference(torch.zeros(1), positions.unsqueeze(0), layer_type)
        else:
            reference = HFGemma4RotaryEmbedding(config.per_layer_config[layer_idx])
            cos, sin = reference(
                torch.zeros(1), positions.unsqueeze(0), layer_type
            )
        torch.testing.assert_close(
            actual.cos_sin_cache[positions, 0, : head_dim // 2],
            cos[0, :, : head_dim // 2],
        )
        torch.testing.assert_close(
            actual.cos_sin_cache[positions, 0, head_dim // 2 :],
            sin[0, :, : head_dim // 2],
        )


def test_gemma4_rope_cache_separates_same_type_head_dims():
    config = _config(
        num_hidden_layers=3,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        per_layer_config={
            "0": {"head_dim": 4},
            "1": {"head_dim": 8},
            "2": {"head_dim": 8},
        },
    )
    with _patch_parallel_context():
        model = Gemma4Model(config, TorchGemma4OperatorProvider())
    assert len(model.rotary_embeddings) == 3
    assert model.layers[0].self_attn.rotary_emb.cos_sin_cache.shape[-1] == 4
    assert model.layers[1].self_attn.rotary_emb.cos_sin_cache.shape[-1] == 8


def test_gemma4_rope_cache_separates_per_layer_parameters():
    parameters = {
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
        "full_attention": {
            "rope_type": "proportional",
            "rope_theta": 1000000.0,
            "partial_rotary_factor": 0.25,
        },
    }
    second_parameters = {
        **parameters,
        "sliding_attention": {"rope_type": "default", "rope_theta": 20000.0},
    }
    config = _config(
        num_hidden_layers=3,
        layer_types=["sliding_attention", "sliding_attention", "full_attention"],
        allow_global_per_layer_attribute_access=True,
        per_layer_config={
            "0": {"head_dim": 4, "rope_parameters": parameters},
            "1": {
                "head_dim": 4,
                "rope_parameters": second_parameters,
            },
            "2": {"head_dim": 8, "rope_parameters": parameters},
        },
    )
    config.allow_global_per_layer_attribute_access = False
    with _patch_parallel_context():
        model = Gemma4Model(config, TorchGemma4OperatorProvider())
    first = model.layers[0].self_attn.rotary_emb.cos_sin_cache
    second = model.layers[1].self_attn.rotary_emb.cos_sin_cache
    assert len(model.rotary_embeddings) == 3
    assert not torch.equal(first, second)
    expected = Gemma4RotaryEmbedding(
        config,
        "sliding_attention",
        4,
        second_parameters,
    )
    torch.testing.assert_close(second, expected.cos_sin_cache)


def test_gemma4_dense_mlp_matches_transformers():
    config = _config()
    with _patch_parallel_context():
        actual = Gemma4MLP(config, 0, TorchGemma4OperatorProvider())
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
        actual = Gemma4Router(
            config, TorchGemma4OperatorProvider(), TorchGemma4RouterProvider()
        )
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
            Gemma4RotaryEmbedding(config, "full_attention", 8),
            TorchGemma4OperatorProvider(),
        )
    loaded_key = torch.randn(
        8, config.hidden_size
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
        actual = Gemma4Model(config, TorchGemma4OperatorProvider())
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
            Gemma4RotaryEmbedding(config, "sliding_attention", 4),
            TorchGemma4OperatorProvider(),
        )
    assert attention.is_kv_shared_layer
    assert tuple(attention.qkv_proj.weight.shape) == (
        config.num_attention_heads * 4,
        config.hidden_size,
    )
    assert not hasattr(attention, "k_norm")
    assert not hasattr(attention, "v_norm")


def test_gemma4_shared_kv_rejects_per_layer_streaming_eviction():
    config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="gemma4_text",
            num_kv_shared_layers=18,
        ),
        sparse_method="streamingllm",
        prefill_sparse_method="",
    )
    with pytest.raises(NotImplementedError, match="KV-sharing"):
        normalize_sparse_methods(config)
