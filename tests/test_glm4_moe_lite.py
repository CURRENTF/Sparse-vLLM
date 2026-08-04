from __future__ import annotations

from contextlib import ExitStack, contextmanager
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from transformers import Glm4MoeLiteConfig
from transformers.models.glm4_moe_lite.modeling_glm4_moe_lite import (
    apply_rotary_pos_emb_interleave,
)

from sparsevllm.config import QuantizationConfig
from sparsevllm.debug.tiny_random import (
    build_tiny_random_hf_model,
    initialize_sparse_model,
)
from sparsevllm.distributed import ParallelContext, ParallelGroup
from sparsevllm.layers.rotary_embedding import apply_interleaved_rotary_emb
from sparsevllm.models.glm4_moe_lite import (
    Glm4MoeLiteAttention,
    Glm4MoeLiteForCausalLM,
    Glm4MoeLiteRouter,
    Glm4MoeLiteSparseMoeBlock,
    _expected_mtp_weight_names,
)
from sparsevllm.models.qwen3 import Qwen3MLP
from sparsevllm.operators.mla_attention import MlaAttentionOpSpec
from sparsevllm.operators.moe import TritonMoeProvider


def _config(**overrides) -> Glm4MoeLiteConfig:
    values = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 16,
        "num_hidden_layers": 2,
        "num_attention_heads": 20,
        "num_key_value_heads": 20,
        "n_shared_experts": 1,
        "n_routed_experts": 64,
        "num_experts_per_tok": 4,
        "routed_scaling_factor": 1.8,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "kv_lora_rank": 512,
        "q_lora_rank": 768,
        "qk_rope_head_dim": 64,
        "v_head_dim": 256,
        "qk_nope_head_dim": 192,
        "max_position_embeddings": 32,
        "dtype": torch.bfloat16,
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 1_000_000.0,
        },
    }
    values.update(overrides)
    config = Glm4MoeLiteConfig(**values)
    config.mlp_chunk_size = 8
    config.quantization_config = QuantizationConfig.disabled()
    config.decode_cuda_graph = False
    return config


def _tp_context(tp_rank: int = 0, tp_size: int = 1) -> ParallelContext:
    ranks = tuple(range(tp_size))
    return ParallelContext(
        world=ParallelGroup(None, ranks, tp_rank, tp_size),
        tensor=ParallelGroup(None, ranks, tp_rank, tp_size),
        expert=ParallelGroup(None, (tp_rank,), 0, 1),
        data=ParallelGroup(None, (tp_rank,), 0, 1),
    )


def _fake_mla(tp_size: int = 1):
    spec = MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=tp_size,
        cuda_graph=False,
    )
    return SimpleNamespace(
        spec=spec,
        provider=SimpleNamespace(name="test"),
        hidden_size=64,
        projection_chunk_size=8,
    )


@contextmanager
def _construction_context(context: ParallelContext):
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "sparsevllm.models.glm4_moe_lite.get_parallel_context",
                return_value=context,
            )
        )
        stack.enter_context(
            patch("sparsevllm.layers.linear.get_parallel_context", return_value=context)
        )
        stack.enter_context(
            patch(
                "sparsevllm.layers.embed_head.get_parallel_context",
                return_value=context,
            )
        )
        stack.enter_context(
            patch(
                "sparsevllm.models.glm4_moe_lite.resolve_moe_provider",
                return_value=TritonMoeProvider(),
            )
        )
        stack.enter_context(torch.device("cpu"))
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            yield
        finally:
            torch.set_default_dtype(previous_dtype)


def _model(config=None, *, tp_rank: int = 0, tp_size: int = 1):
    config = _config() if config is None else config
    context = _tp_context(tp_rank, tp_size)
    with _construction_context(context):
        return Glm4MoeLiteForCausalLM(
            config,
            mla_attention=_fake_mla(tp_size),
            mlp_chunk_size=config.mlp_chunk_size,
            decode_cuda_graph=config.decode_cuda_graph,
            expect_mtp_weights=False,
        )


def test_glm_topology_reuses_one_mla_object_and_qwen_dense_mlp() -> None:
    model = _model()

    assert len(model.model.layers) == 2
    assert isinstance(model.model.layers[0].mlp, Qwen3MLP)
    assert isinstance(model.model.layers[1].mlp, Glm4MoeLiteSparseMoeBlock)
    assert model.model.layers[0].self_attn.mla_attention is model.model.mla_attention
    assert model.model.layers[1].self_attn.mla_attention is model.model.mla_attention
    assert model.model.rotary_emb.interleaved is True
    assert model.model.layers[1].mlp.experts.op_spec.routing_method == "biased_sigmoid"


def test_glm_interleaved_rope_matches_transformers() -> None:
    torch.manual_seed(13)
    q = torch.randn(1, 3, 5, 64)
    k = torch.randn(1, 1, 5, 64)
    angles = torch.randn(1, 5, 32)
    cos_half = angles.cos()
    sin_half = angles.sin()
    cos = torch.cat((cos_half, cos_half), dim=-1)
    sin = torch.cat((sin_half, sin_half), dim=-1)

    expected_q, expected_k = apply_rotary_pos_emb_interleave(q, k, cos, sin)
    actual_q = apply_interleaved_rotary_emb(
        q,
        cos_half.unsqueeze(1),
        sin_half.unsqueeze(1),
    )
    actual_k = apply_interleaved_rotary_emb(
        k,
        cos_half.unsqueeze(1),
        sin_half.unsqueeze(1),
    )

    torch.testing.assert_close(actual_q, expected_q)
    torch.testing.assert_close(actual_k, expected_k)


def test_glm_tp_projection_slices_follow_local_heads() -> None:
    config = _config()
    context = _tp_context(tp_rank=2, tp_size=4)
    with _construction_context(context):
        attention = Glm4MoeLiteAttention(
            config,
            _fake_mla(tp_size=4),
            projection_chunk_size=config.mlp_chunk_size,
        )

    q_source = (
        torch.arange(20 * 256 * 768).remainder(127).to(torch.bfloat16)
    ).view(20 * 256, 768)
    kv_source = (
        torch.arange(20 * 448 * 512).remainder(127).to(torch.bfloat16)
    ).view(20 * 448, 512)
    o_source = (
        torch.arange(64 * 20 * 256).remainder(127).to(torch.bfloat16)
    ).view(64, 20 * 256)
    attention.q_b_proj.weight_loader(attention.q_b_proj.weight, q_source)
    attention.kv_b_proj.weight_loader(attention.kv_b_proj.weight, kv_source)
    attention.o_proj.weight_loader(attention.o_proj.weight, o_source)

    assert attention.local_heads == 5
    assert torch.equal(attention.q_b_proj.weight, q_source[2560:3840])
    assert torch.equal(attention.kv_b_proj.weight, kv_source[4480:6720])
    assert torch.equal(attention.o_proj.weight, o_source[:, 2560:3840])


def test_glm_decode_absorption_and_value_reconstruction_match_linear_algebra() -> None:
    config = _config()
    context = _tp_context()
    with _construction_context(context):
        attention = Glm4MoeLiteAttention(
            config,
            _fake_mla(),
            projection_chunk_size=config.mlp_chunk_size,
        )
    torch.manual_seed(19)
    attention.kv_b_proj.weight.data.normal_(mean=0.0, std=0.02)
    q_nope = torch.randn(3, 20, 192, dtype=torch.bfloat16)
    latent_output = torch.randn(3, 20, 512, dtype=torch.bfloat16)
    weight = attention.kv_b_proj.weight.view(20, 448, 512)

    absorbed = attention._decode_absorbed_query(q_nope)
    reconstructed = attention._reconstruct_decode_values(latent_output)

    expected_absorbed = torch.einsum(
        "bhd,hdr->bhr",
        q_nope,
        weight[:, :192],
    )
    expected_reconstructed = torch.einsum(
        "bhr,hvr->bhv",
        latent_output,
        weight[:, 192:],
    )
    torch.testing.assert_close(absorbed, expected_absorbed)
    torch.testing.assert_close(reconstructed, expected_reconstructed)


def test_glm_router_uses_bias_only_for_selection_and_scales_weights() -> None:
    torch.manual_seed(23)
    router = Glm4MoeLiteRouter(_config())
    router.weight.data.normal_(mean=0.0, std=0.1)
    router.e_score_correction_bias.data.zero_()
    router.e_score_correction_bias.data[:4] = 100.0

    def torch_topk(logits, correction_bias, *, top_k):
        scores = logits.sigmoid()
        ids = torch.topk(
            scores + correction_bias,
            top_k,
            dim=-1,
            sorted=False,
        ).indices
        weights = scores.gather(1, ids)
        return weights / (weights.sum(dim=-1, keepdim=True) + 1e-20), ids

    router.topk_impl = torch_topk
    hidden_states = torch.randn(7, 64, dtype=torch.bfloat16)
    logits, weights, ids = router(hidden_states)
    original_scores = logits.sigmoid()
    expected = original_scores.gather(1, ids)
    expected = expected / (expected.sum(dim=-1, keepdim=True) + 1e-20) * 1.8

    assert all(set(row.tolist()) == {0, 1, 2, 3} for row in ids)
    torch.testing.assert_close(weights, expected)
    torch.testing.assert_close(weights.sum(dim=-1), torch.full((7,), 1.8))


def test_tiny_transformers_weights_load_through_strict_glm_mapping() -> None:
    config = _config()
    context = _tp_context()
    with _construction_context(context):
        model = Glm4MoeLiteForCausalLM(
            config,
            mla_attention=_fake_mla(),
            mlp_chunk_size=config.mlp_chunk_size,
            decode_cuda_graph=config.decode_cuda_graph,
            expect_mtp_weights=False,
        )
        initialize_sparse_model(model, config, seed=29)
    reference = build_tiny_random_hf_model(config, seed=29)
    reference_experts = reference.model.layers[1].mlp.experts
    target_experts = model.model.layers[1].mlp.experts

    torch.testing.assert_close(
        model.model.layers[0].self_attn.q_b_proj.weight.cpu(),
        reference.model.layers[0].self_attn.q_b_proj.weight,
    )
    torch.testing.assert_close(
        target_experts.w13_weight[:, :16].cpu(),
        reference_experts.gate_up_proj[:, :16],
    )
    torch.testing.assert_close(
        target_experts.w13_weight[:, 16:].cpu(),
        reference_experts.gate_up_proj[:, 16:],
    )
    torch.testing.assert_close(
        target_experts.w2_weight.cpu(),
        reference_experts.down_proj,
    )
    assert len(target_experts._loaded_expert_shards) == 64 * 3
    assert model.model.layers[1].mlp.gate.e_score_correction_bias.dtype == torch.float32


def test_glm_mtp_skip_set_is_exact() -> None:
    config = _config()
    context = _tp_context()
    with _construction_context(context):
        model = Glm4MoeLiteForCausalLM(
            config,
            mla_attention=_fake_mla(),
            mlp_chunk_size=config.mlp_chunk_size,
            decode_cuda_graph=config.decode_cuda_graph,
            expect_mtp_weights=True,
        )
    dense_names = {
        name
        for name, _ in model.named_parameters()
        if not name.endswith(".mlp.experts.w13_weight")
        and not name.endswith(".mlp.experts.w2_weight")
    }
    for block in (
        layer.mlp
        for layer in model.model.layers
        if isinstance(layer.mlp, Glm4MoeLiteSparseMoeBlock)
    ):
        block.experts._loaded_expert_shards = {
            (expert_id, projection)
            for expert_id in range(64)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
    expected_mtp = _expected_mtp_weight_names(64)
    for name in expected_mtp:
        model.record_skipped_weight(name, (1,), "BF16", None, None)

    model.validate_loaded_weights(dense_names)
    assert len(model._intentionally_skipped_mtp_weights) == 212

    model._intentionally_skipped_mtp_weights.remove(next(iter(expected_mtp)))
    with pytest.raises(ValueError, match="MTP skip set is inconsistent"):
        model.validate_loaded_weights(dense_names)
