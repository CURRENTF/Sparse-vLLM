from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file
from transformers import Qwen3MoeConfig
from transformers.models.qwen3_moe.modeling_qwen3_moe import (
    Qwen3MoeSparseMoeBlock as HFQwen3MoeSparseMoeBlock,
)

from sparsevllm.distributed import ParallelContext, ParallelGroup
from sparsevllm.models.qwen3_moe import (
    Qwen3MoeForCausalLM,
    Qwen3MoePackedExperts,
    Qwen3MoeRouter,
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


def test_router_matches_qwen3_moe_reference_math():
    torch.manual_seed(0)
    config = _config()
    router = Qwen3MoeRouter(config)
    router.weight.data.normal_(mean=0.0, std=0.2)
    hidden_states = torch.randn(7, config.hidden_size)

    logits, topk_weights, topk_ids = router(hidden_states)

    expected_logits = torch.nn.functional.linear(hidden_states, router.weight)
    expected_probs = torch.softmax(expected_logits, dtype=torch.float32, dim=-1)
    expected_weights, expected_ids = torch.topk(
        expected_probs,
        config.num_experts_per_tok,
        dim=-1,
    )
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    assert torch.equal(logits, expected_logits)
    assert torch.equal(topk_ids, expected_ids)
    assert torch.allclose(topk_weights, expected_weights.to(logits.dtype))


def test_pytorch_moe_block_matches_transformers_reference():
    torch.manual_seed(4)
    config = _config()
    reference = HFQwen3MoeSparseMoeBlock(config)
    reference.gate.weight.data.normal_(mean=0.0, std=0.2)
    reference.experts.gate_up_proj.data.normal_(mean=0.0, std=0.2)
    reference.experts.down_proj.data.normal_(mean=0.0, std=0.2)
    context = _ep_context(0, 1)
    with patch("sparsevllm.models.qwen3_moe.get_parallel_context", return_value=context):
        actual = Qwen3MoeSparseMoeBlock(config)
    actual.gate.weight.data.copy_(reference.gate.weight)
    actual.experts.w13_weight.data.copy_(reference.experts.gate_up_proj)
    actual.experts.w2_weight.data.copy_(reference.experts.down_proj)
    hidden_states = torch.randn(9, config.hidden_size)

    expected = reference(hidden_states.unsqueeze(0)).squeeze(0)
    output = actual(hidden_states)

    assert torch.allclose(output, expected, atol=1e-6, rtol=1e-6)


def test_moe_block_dispatches_only_the_selected_backend():
    config = _config()
    config.moe_backend = "triton"
    context = _ep_context(0, 1)
    with patch("sparsevllm.models.qwen3_moe.get_parallel_context", return_value=context):
        block = Qwen3MoeSparseMoeBlock(config)
    hidden_states = torch.randn(3, config.hidden_size)
    expected = torch.randn_like(hidden_states)

    with (
        patch.object(block.gate, "forward", return_value=(
            torch.empty(3, config.num_experts),
            torch.empty(3, config.num_experts_per_tok),
            torch.empty(3, config.num_experts_per_tok, dtype=torch.int64),
        )),
        patch.object(block.experts, "forward_pytorch") as pytorch_forward,
        patch.object(block.experts, "forward_triton", return_value=expected) as triton_forward,
    ):
        actual = block(hidden_states)

    assert torch.equal(actual, expected)
    triton_forward.assert_called_once()
    pytorch_forward.assert_not_called()


def test_packed_expert_weight_mapping_and_oracle():
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

    hidden_states = torch.randn(5, config.hidden_size)
    topk_ids = torch.tensor([[0, 1], [1, 1], [2, 3], [3, 0], [2, 0]])
    topk_weights = torch.rand(5, 2)
    actual = experts.forward_pytorch(hidden_states, topk_ids, topk_weights)

    expected = torch.zeros_like(hidden_states)
    for token_id in range(hidden_states.shape[0]):
        for topk_slot in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_id, topk_slot])
            gate = torch.nn.functional.linear(
                hidden_states[token_id],
                source_weights[(expert_id, "gate_proj")],
            )
            up = torch.nn.functional.linear(
                hidden_states[token_id],
                source_weights[(expert_id, "up_proj")],
            )
            output = torch.nn.functional.linear(
                torch.nn.functional.silu(gate) * up,
                source_weights[(expert_id, "down_proj")],
            )
            expected[token_id] += output * topk_weights[token_id, topk_slot]
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_ep_local_contributions_sum_to_full_expert_output():
    torch.manual_seed(2)
    config = _config()
    with patch(
        "sparsevllm.models.qwen3_moe.get_parallel_context",
        return_value=_ep_context(0, 1),
    ):
        full = Qwen3MoePackedExperts(config)
    with patch(
        "sparsevllm.models.qwen3_moe.get_parallel_context",
        return_value=_ep_context(0, 2),
    ):
        rank0 = Qwen3MoePackedExperts(config)
    with patch(
        "sparsevllm.models.qwen3_moe.get_parallel_context",
        return_value=_ep_context(1, 2),
    ):
        rank1 = Qwen3MoePackedExperts(config)

    full.w13_weight.data.normal_(mean=0.0, std=0.2)
    full.w2_weight.data.normal_(mean=0.0, std=0.2)
    rank0.w13_weight.data.copy_(full.w13_weight[:2])
    rank0.w2_weight.data.copy_(full.w2_weight[:2])
    rank1.w13_weight.data.copy_(full.w13_weight[2:])
    rank1.w2_weight.data.copy_(full.w2_weight[2:])
    hidden_states = torch.randn(6, config.hidden_size)
    topk_ids = torch.tensor([[0, 1], [2, 3], [0, 0], [3, 3], [1, 2], [0, 1]])
    topk_weights = torch.rand(6, 2)

    expected = full.forward_pytorch(hidden_states, topk_ids, topk_weights)
    actual = rank0.forward_pytorch(hidden_states, topk_ids, topk_weights)
    actual += rank1.forward_pytorch(hidden_states, topk_ids, topk_weights)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)


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
