from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig

from sparsevllm.distributed import ParallelContext, ParallelGroup
from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.models.qwen3 import Qwen3Attention
from sparsevllm.models.qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoePackedExperts,
    Qwen3MoeSparseMoeBlock,
)
from sparsevllm.utils.loader import load_model


def _config(**overrides) -> Qwen3MoeConfig:
    values = {
        "vocab_size": 32,
        "hidden_size": 8,
        "intermediate_size": 16,
        "moe_intermediate_size": 6,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "num_experts": 4,
        "num_experts_per_tok": 2,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "norm_topk_prob": True,
        "hidden_act": "silu",
        "attention_bias": False,
        "max_position_embeddings": 32,
        "rope_theta": 10000.0,
        "tie_word_embeddings": False,
    }
    values.update(overrides)
    return Qwen3MoeConfig(**values)


def _ep_context(ep_rank: int, ep_size: int) -> ParallelContext:
    ranks = tuple(range(ep_size))
    return ParallelContext(
        world=ParallelGroup(None, ranks, ep_rank, ep_size),
        tensor=ParallelGroup(None, (ep_rank,), 0, 1),
        expert=ParallelGroup(None, ranks, ep_rank, ep_size),
        data=ParallelGroup(None, (ep_rank,), 0, 1),
    )


def _instantiate_model(config, context):
    with (
        patch("sparsevllm.models.qwen3_moe.get_parallel_context", return_value=context),
        patch("sparsevllm.models.qwen3.get_parallel_context", return_value=context),
        patch("sparsevllm.layers.linear.get_parallel_context", return_value=context),
        patch("sparsevllm.layers.embed_head.get_parallel_context", return_value=context),
    ):
        return Qwen3MoeForCausalLM(config)


def _rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    x_float = x.float()
    normalized = x_float * torch.rsqrt(
        x_float.square().mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * weight.float()).to(x.dtype)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_qwen3_rmsnorm_does_not_modify_input(dtype):
    pytest.importorskip("flashinfer")
    norm = RMSNorm(128).cuda().to(dtype)
    x = torch.randn(3, 128, device="cuda", dtype=dtype)
    original = x.clone()

    actual = norm(x)

    assert torch.equal(x, original)
    torch.testing.assert_close(
        actual,
        _rmsnorm_reference(x, norm.weight, norm.eps),
        rtol=1.0e-2,
        atol=3.0e-2,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("with_residual", [False, True])
def test_rmsnorm_matches_fp32_reference(dtype, with_residual):
    pytest.importorskip("flashinfer")
    torch.manual_seed(27)
    norm = RMSNorm(128).cuda().to(dtype)
    norm.weight.data.normal_(mean=1.0, std=0.2)
    x = torch.randn(7, 128, device="cuda", dtype=dtype)
    residual = torch.randn_like(x) if with_residual else None

    if residual is None:
        expected = _rmsnorm_reference(x, norm.weight, norm.eps)
        actual = norm(x)
    else:
        merged = x.float() + residual.float()
        expected_residual = merged.to(dtype)
        expected = _rmsnorm_reference(
            merged,
            norm.weight,
            norm.eps,
        ).to(dtype)
        actual, actual_residual = norm(x, residual)
        assert torch.equal(actual_residual, expected_residual)

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1.0e-2,
        atol=3.0e-2,
    )


def test_qwen3_attention_passes_raw_key_without_clone():
    class FixedProjection(torch.nn.Module):
        def __init__(self, output):
            super().__init__()
            self.output = output

        def forward(self, _):
            return self.output

    class PairIdentity(torch.nn.Module):
        def forward(self, _positions, query, key):
            return query, key

    class AttentionIdentity(torch.nn.Module):
        def forward(self, query, _key, _value):
            return query

    class CacheRecorder:
        def save_raw_kv_if_needed(self, _layer_idx, key, _value):
            self.raw_key = key
            self.saved_raw_key = key.clone()

        def save_rope_kv_if_needed(self, _layer_idx, _key, _value):
            pass

    attention = Qwen3Attention.__new__(Qwen3Attention)
    torch.nn.Module.__init__(attention)
    qkv = torch.randn(2, 16)
    expected_raw_key = qkv[:, 8:12].view(2, 1, 4)
    attention.qkv_proj = FixedProjection(qkv)
    attention.o_proj = torch.nn.Identity()
    attention.q_norm = torch.nn.Identity()
    attention.k_norm = torch.nn.Identity()
    attention.rotary_emb = PairIdentity()
    attention.attn = AttentionIdentity()
    attention.q_size = 8
    attention.kv_size = 4
    attention.num_heads = 2
    attention.num_kv_heads = 1
    attention.head_dim = 4
    attention.qkv_bias = False
    attention.proj_chunk_size = 16
    cache = CacheRecorder()
    context = SimpleNamespace(cache_manager=cache, now_layer_idx=0)

    with patch("sparsevllm.models.qwen3.get_context", return_value=context):
        attention(torch.arange(2), torch.empty(2, 8))

    assert cache.raw_key.data_ptr() == expected_raw_key.data_ptr()
    assert torch.equal(cache.saved_raw_key, expected_raw_key)


def test_moe_block_uses_triton_kernels():
    from sparsevllm.triton_kernel.moe_topk import topk_softmax

    config = _config()
    context = _ep_context(0, 1)
    with patch("sparsevllm.models.qwen3_moe.get_parallel_context", return_value=context):
        block = Qwen3MoeSparseMoeBlock(config)
    assert block.gate.topk_impl is topk_softmax
    hidden_states = torch.randn(3, config.hidden_size)
    expected = torch.randn_like(hidden_states)

    with (
        patch.object(block.gate, "forward", return_value=(
            torch.empty(3, config.num_experts),
            torch.empty(3, config.num_experts_per_tok),
            torch.empty(3, config.num_experts_per_tok, dtype=torch.int64),
        )),
        patch.object(block.experts, "forward", return_value=expected) as triton_forward,
    ):
        actual = block(hidden_states)

    assert torch.equal(actual, expected)
    triton_forward.assert_called_once()


def test_moe_block_reduces_in_activation_dtype():
    config = _config()
    context = _ep_context(0, 1)
    with patch("sparsevllm.models.qwen3_moe.get_parallel_context", return_value=context):
        block = Qwen3MoeSparseMoeBlock(config)
    hidden_states = torch.randn(3, config.hidden_size, dtype=torch.bfloat16)
    local_output = torch.randn(3, config.hidden_size, dtype=torch.bfloat16)

    with (
        patch.object(
            block.gate,
            "forward",
            return_value=(
                torch.empty(3, config.num_experts, dtype=torch.bfloat16),
                torch.empty(3, config.num_experts_per_tok, dtype=torch.bfloat16),
                torch.empty(3, config.num_experts_per_tok, dtype=torch.int64),
            ),
        ),
        patch.object(block.experts, "forward", return_value=local_output),
    ):
        output = block(hidden_states)

    assert output.dtype == hidden_states.dtype
    assert torch.equal(output, local_output)


def test_decoder_layer_broadcasts_attention_output_before_post_norm():
    config = _config()
    context = _ep_context(0, 2)
    model = _instantiate_model(config, context)
    layer = model.model.layers[0]
    hidden_states = torch.randn(3, config.hidden_size)
    residual = torch.randn_like(hidden_states)
    calls = []

    with (
        patch.object(
            layer.input_layernorm,
            "forward",
            return_value=(hidden_states, residual),
        ),
        patch.object(layer.self_attn, "forward", return_value=hidden_states),
        patch.object(
            ParallelContext,
            "ep_broadcast",
            side_effect=lambda state, **_: calls.append(("broadcast", state.shape)),
        ),
        patch.object(
            layer.post_attention_layernorm,
            "forward",
            side_effect=lambda state, res: (
                calls.append(("post_norm", state.shape)) or (state, res)
            ),
        ),
        patch.object(layer.mlp, "forward", return_value=hidden_states),
    ):
        layer(torch.arange(3), hidden_states, residual)

    assert calls == [
        ("broadcast", hidden_states.shape),
        ("post_norm", hidden_states.shape),
    ]


def test_moe_warmup_uses_one_local_decode_assignment():
    config = _config(num_experts_per_tok=3)
    context = _ep_context(1, 2)
    model = _instantiate_model(config, context)
    experts = model.model.layers[0].mlp.experts
    expected = torch.zeros(1, config.hidden_size)

    with (
        patch.object(experts, "forward", return_value=expected) as forward,
        patch(
            "sparsevllm.models.qwen3_moe.device_runtime.synchronize"
        ) as synchronize,
    ):
        model.warmup_moe()

    hidden_states, topk_ids, topk_weights = forward.call_args.args
    assert hidden_states.shape == (1, config.hidden_size)
    assert topk_ids.tolist() == [[2, 3, 2]]
    assert torch.allclose(topk_weights, torch.full((1, 3), 1 / 3))
    synchronize.assert_called_once()


def test_packed_expert_weight_mapping():
    torch.manual_seed(1)
    config = _config()
    context = _ep_context(0, 1)
    with patch("sparsevllm.models.qwen3_moe.get_parallel_context", return_value=context):
        experts = Qwen3MoePackedExperts(config)

    source_weights = {}
    for expert_id in range(config.num_experts):
        for projection, shape in {
            "gate_proj": (config.moe_intermediate_size, config.hidden_size),
            "up_proj": (config.moe_intermediate_size, config.hidden_size),
            "down_proj": (config.hidden_size, config.moe_intermediate_size),
        }.items():
            weight = torch.randn(shape)
            source_weights[(expert_id, projection)] = weight
            experts.load_expert_weight(expert_id, projection, weight)
    experts.validate_loaded_weights()

    for expert_id in range(config.num_experts):
        assert torch.equal(
            experts.w13_weight[expert_id, : config.moe_intermediate_size],
            source_weights[(expert_id, "gate_proj")],
        )
        assert torch.equal(
            experts.w13_weight[expert_id, config.moe_intermediate_size :],
            source_weights[(expert_id, "up_proj")],
        )
        assert torch.equal(
            experts.w2_weight[expert_id],
            source_weights[(expert_id, "down_proj")],
        )


def test_model_maps_only_local_experts_and_validates_all_weights():
    torch.manual_seed(3)
    config = _config()
    model = _instantiate_model(config, _ep_context(1, 2))

    for expert_id in range(config.num_experts):
        for projection, shape in {
            "gate_proj": (config.moe_intermediate_size, config.hidden_size),
            "up_proj": (config.moe_intermediate_size, config.hidden_size),
            "down_proj": (config.hidden_size, config.moe_intermediate_size),
        }.items():
            source_name = f"model.layers.0.mlp.experts.{expert_id}.{projection}.weight"
            target_name = model.map_weight_name(source_name)
            if expert_id < 2:
                assert target_name is None
            else:
                assert target_name is not None
                assert model.load_special_weight(
                    target_name,
                    torch.randn(shape),
                    None,
                ) == 1

    packed_names = {
        name
        for name, _ in model.named_parameters()
        if name.endswith(".mlp.experts.w13_weight")
        or name.endswith(".mlp.experts.w2_weight")
    }
    dense_names = {name for name, _ in model.named_parameters()} - packed_names
    model.validate_loaded_weights(dense_names)
    assert len(model._intentionally_skipped_expert_weights) == 6


def test_missing_local_expert_weight_fails_validation():
    config = _config()
    model = _instantiate_model(config, _ep_context(0, 2))
    experts = model.model.layers[0].mlp.experts
    with pytest.raises(ValueError, match="Missing local Qwen3MoE expert weights"):
        experts.validate_loaded_weights()


def test_checkpoint_loader_loads_local_experts_and_skips_remote(tmp_path):
    torch.manual_seed(5)
    config = _config()
    context = _ep_context(1, 2)
    template = _instantiate_model(config, context)
    checkpoint = {}
    for name, parameter in template.named_parameters():
        if name.endswith(".mlp.experts.w13_weight") or name.endswith(
            ".mlp.experts.w2_weight"
        ):
            continue
        value = torch.randn(parameter.shape, dtype=parameter.dtype)
        if name.endswith(".self_attn.qkv_proj.weight"):
            prefix = name[: -len("qkv_proj.weight")]
            q_size = config.num_attention_heads * config.head_dim
            kv_size = config.num_key_value_heads * config.head_dim
            checkpoint[prefix + "q_proj.weight"] = value[:q_size].clone()
            checkpoint[prefix + "k_proj.weight"] = value[q_size : q_size + kv_size].clone()
            checkpoint[prefix + "v_proj.weight"] = value[q_size + kv_size :].clone()
        else:
            checkpoint[name] = value

    local_sources = {}
    for expert_id in range(config.num_experts):
        for projection, shape in {
            "gate_proj": (config.moe_intermediate_size, config.hidden_size),
            "up_proj": (config.moe_intermediate_size, config.hidden_size),
            "down_proj": (config.hidden_size, config.moe_intermediate_size),
        }.items():
            name = f"model.layers.0.mlp.experts.{expert_id}.{projection}.weight"
            checkpoint[name] = torch.randn(shape)
            if expert_id >= 2:
                local_sources[(expert_id, projection)] = checkpoint[name]
    save_file(checkpoint, tmp_path / "model.safetensors")

    target = _instantiate_model(config, context)
    load_model(target, str(tmp_path), tp_rank=0, tp_size=1)

    experts = target.model.layers[0].mlp.experts
    assert torch.equal(experts.w13_weight[0, : config.moe_intermediate_size], local_sources[(2, "gate_proj")])
    assert torch.equal(experts.w13_weight[0, config.moe_intermediate_size :], local_sources[(2, "up_proj")])
    assert torch.equal(experts.w2_weight[1], local_sources[(3, "down_proj")])
    assert len(target._intentionally_skipped_expert_weights) == 6
