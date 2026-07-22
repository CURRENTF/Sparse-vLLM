from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file
from transformers import (
    MiniMaxM2Config as TransformersMiniMaxM2Config,
    MiniMaxM2ForCausalLM as TransformersMiniMaxM2ForCausalLM,
)

from sparsevllm.config import QuantizationConfig
from sparsevllm.distributed import ParallelContext, ParallelGroup
from sparsevllm.models.minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2ForCausalLM,
    MiniMaxM2PackedExperts,
    MiniMaxM2Router,
)
from sparsevllm.quantization.fp8 import (
    fp8_blockwise_dequantize,
    fp8_blockwise_linear_reference,
)
from sparsevllm.utils.loader import load_model


def _config(**overrides):
    values = {
        "vocab_size": 32,
        "hidden_size": 128,
        "intermediate_size": 128,
        "num_hidden_layers": 1,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "head_dim": 64,
        "rotary_dim": 32,
        "num_local_experts": 4,
        "num_experts_per_tok": 2,
        "max_position_embeddings": 32,
        "rope_theta": 5_000_000.0,
        "rms_norm_eps": 1.0e-6,
        "hidden_act": "silu",
        "tie_word_embeddings": False,
        "torch_dtype": torch.float32,
        "moe_backend": "pytorch",
        "quantization_config": QuantizationConfig(
            enabled=True,
            quant_method="fp8",
            weight_dtype="e4m3",
            activation_scheme="dynamic",
            weight_block_size=(128, 128),
            backend="auto",
            model_name="MiniMax M2.7",
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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
        patch(
            "sparsevllm.models.minimax_m2.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparsevllm.models.qwen3.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparsevllm.layers.linear.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparsevllm.layers.embed_head.get_parallel_context",
            return_value=context,
        ),
    ):
        return MiniMaxM2ForCausalLM(config)


def _random_fp8(shape):
    return torch.randn(shape).clamp(-4.0, 4.0).to(torch.float8_e4m3fn)


def test_router_matches_official_biased_sigmoid_math():
    torch.manual_seed(0)
    config = _config()
    router = MiniMaxM2Router(config)
    router.weight.data.normal_(mean=0.0, std=0.1)
    router.e_score_correction_bias.data.normal_(mean=0.0, std=0.05)
    hidden_states = torch.randn(7, config.hidden_size)

    logits, weights, ids = router(hidden_states)

    expected_logits = F.linear(hidden_states.float(), router.weight)
    routing_weights = torch.sigmoid(expected_logits)
    _, expected_ids = torch.topk(
        routing_weights + router.e_score_correction_bias,
        config.num_experts_per_tok,
        dim=-1,
        sorted=False,
    )
    expected_weights = routing_weights.gather(1, expected_ids)
    expected_weights /= expected_weights.sum(dim=-1, keepdim=True)
    assert torch.equal(logits, expected_logits)
    assert torch.equal(ids, expected_ids)
    assert torch.equal(weights, expected_weights)


def _load_random_experts(experts):
    source = {}
    for expert_id in range(experts.local_expert_start, experts.local_expert_end):
        for projection, shape in {
            "w1": (experts.intermediate_size, experts.hidden_size),
            "w2": (experts.hidden_size, experts.intermediate_size),
            "w3": (experts.intermediate_size, experts.hidden_size),
        }.items():
            weight = _random_fp8(shape)
            scale = torch.rand(
                shape[0] // 128,
                shape[1] // 128,
                dtype=torch.float32,
            ) + 0.1
            experts.load_expert_weight(expert_id, projection, weight, scale)
            source[(expert_id, projection)] = (weight, scale)
    experts.validate_loaded_weights()
    return source


def test_reference_packed_experts_match_explicit_oracle():
    torch.manual_seed(1)
    config = _config()
    with patch(
        "sparsevllm.models.minimax_m2.get_parallel_context",
        return_value=_ep_context(0, 1),
    ):
        experts = MiniMaxM2PackedExperts(config)
    source = _load_random_experts(experts)
    hidden_states = torch.randn(5, config.hidden_size)
    topk_ids = torch.tensor([[0, 1], [1, 1], [2, 3], [3, 0], [2, 0]])
    topk_weights = torch.rand(5, 2)

    actual = experts.forward_reference(hidden_states, topk_ids, topk_weights)

    expected = torch.zeros_like(hidden_states)
    for token_id in range(hidden_states.shape[0]):
        for topk_slot in range(topk_ids.shape[1]):
            expert_id = int(topk_ids[token_id, topk_slot])
            gate = fp8_blockwise_linear_reference(
                hidden_states[token_id], *source[(expert_id, "w1")]
            )
            up = fp8_blockwise_linear_reference(
                hidden_states[token_id], *source[(expert_id, "w3")]
            )
            expert_output = fp8_blockwise_linear_reference(
                F.silu(gate) * up,
                *source[(expert_id, "w2")],
            )
            expected[token_id] += (
                expert_output * topk_weights[token_id, topk_slot]
            )
    torch.testing.assert_close(actual, expected, atol=2.0e-5, rtol=2.0e-5)


def test_ep2_reference_local_outputs_sum_to_ep1_oracle():
    torch.manual_seed(2)
    config = _config()
    with patch(
        "sparsevllm.models.minimax_m2.get_parallel_context",
        return_value=_ep_context(0, 1),
    ):
        full = MiniMaxM2PackedExperts(config)
    with patch(
        "sparsevllm.models.minimax_m2.get_parallel_context",
        return_value=_ep_context(0, 2),
    ):
        rank0 = MiniMaxM2PackedExperts(config)
    with patch(
        "sparsevllm.models.minimax_m2.get_parallel_context",
        return_value=_ep_context(1, 2),
    ):
        rank1 = MiniMaxM2PackedExperts(config)

    full.w13_weight.data.copy_(_random_fp8(full.w13_weight.shape))
    full.w2_weight.data.copy_(_random_fp8(full.w2_weight.shape))
    full.w13_scale_inv.copy_(torch.rand_like(full.w13_scale_inv) + 0.1)
    full.w2_scale_inv.copy_(torch.rand_like(full.w2_scale_inv) + 0.1)
    rank0.w13_weight.data.copy_(full.w13_weight[:2])
    rank0.w2_weight.data.copy_(full.w2_weight[:2])
    rank0.w13_scale_inv.copy_(full.w13_scale_inv[:2])
    rank0.w2_scale_inv.copy_(full.w2_scale_inv[:2])
    rank1.w13_weight.data.copy_(full.w13_weight[2:])
    rank1.w2_weight.data.copy_(full.w2_weight[2:])
    rank1.w13_scale_inv.copy_(full.w13_scale_inv[2:])
    rank1.w2_scale_inv.copy_(full.w2_scale_inv[2:])
    hidden_states = torch.randn(6, config.hidden_size)
    topk_ids = torch.tensor([[0, 1], [2, 3], [0, 0], [3, 3], [1, 2], [0, 1]])
    topk_weights = torch.rand(6, 2)

    expected = full.forward_reference(hidden_states, topk_ids, topk_weights)
    actual = rank0.forward_reference(hidden_states, topk_ids, topk_weights)
    actual += rank1.forward_reference(hidden_states, topk_ids, topk_weights)
    torch.testing.assert_close(actual, expected, atol=1.0e-6, rtol=1.0e-6)


class _FixedProjection(torch.nn.Module):
    def __init__(self, output):
        super().__init__()
        self.output = output

    def forward(self, _hidden_states):
        return self.output


class _CaptureAttention(torch.nn.Module):
    def forward(self, query, key, value):
        self.query = query
        self.key = key
        self.value = value
        return query


class _TinyCausalAttention(torch.nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = float(scale)

    def forward(self, query, key, value):
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        causal_mask = torch.triu(
            torch.ones(
                scores.shape[-2:],
                dtype=torch.bool,
                device=scores.device,
            ),
            diagonal=1,
        )
        scores = scores.masked_fill(causal_mask, torch.finfo(scores.dtype).min)
        probabilities = torch.softmax(scores, dim=-1, dtype=torch.float32).to(
            query.dtype
        )
        return torch.matmul(probabilities, value).transpose(0, 1).contiguous()


class _CacheRecorder:
    def save_raw_kv_if_needed(self, _layer_idx, key, value):
        self.raw_key = key.clone()
        self.raw_value = value.clone()

    def save_rope_kv_if_needed(self, _layer_idx, key, value):
        self.rope_key = key.clone()
        self.rope_value = value.clone()


def test_attention_uses_flat_qk_norm_and_partial_rope():
    torch.manual_seed(3)
    config = _config()
    context = _ep_context(0, 1)
    with (
        patch(
            "sparsevllm.models.minimax_m2.get_parallel_context",
            return_value=context,
        ),
        patch(
            "sparsevllm.layers.linear.get_parallel_context",
            return_value=context,
        ),
    ):
        attention = MiniMaxM2Attention(config)
    qkv = torch.randn(3, 3 * config.hidden_size)
    qkv[:, :64] *= 0.1
    qkv[:, 64:128] *= 10.0
    attention.qkv_proj = _FixedProjection(qkv)
    attention.o_proj = torch.nn.Identity()
    capture = _CaptureAttention()
    attention.attn = capture
    cache = _CacheRecorder()
    runtime_context = SimpleNamespace(now_layer_idx=0, cache_manager=cache)
    positions = torch.tensor([0, 1, 2])

    with patch(
        "sparsevllm.models.minimax_m2.get_context",
        return_value=runtime_context,
    ):
        attention(positions, torch.empty(3, config.hidden_size))

    raw_q, raw_k, raw_v = qkv.split([128, 128, 128], dim=-1)
    expected_flat_q = attention.q_norm._rms_forward_impl(raw_q).view(3, 2, 64)
    expected_flat_k = attention.k_norm._rms_forward_impl(raw_k).view(3, 2, 64)
    per_head_q = raw_q.view(3, 2, 64)
    per_head_q = per_head_q * torch.rsqrt(
        per_head_q.pow(2).mean(dim=-1, keepdim=True) + attention.q_norm.eps
    )
    cos, sin = attention.rotary_emb.cos_sin_cache[positions].chunk(2, dim=-1)
    expected_q_prefix = torch.cat(
        (
            expected_flat_q[..., :16].float() * cos
            - expected_flat_q[..., 16:32].float() * sin,
            expected_flat_q[..., 16:32].float() * cos
            + expected_flat_q[..., :16].float() * sin,
        ),
        dim=-1,
    ).to(expected_flat_q.dtype)

    assert attention.q_norm.weight.shape == (128,)
    assert attention.k_norm.weight.shape == (128,)
    assert torch.equal(cache.raw_key, raw_k.view(3, 2, 64))
    assert torch.equal(cache.raw_value, raw_v.view(3, 2, 64))
    assert not torch.allclose(expected_flat_q, per_head_q)
    assert torch.equal(capture.query[..., :32], expected_q_prefix)
    assert torch.equal(capture.query[..., 32:], expected_flat_q[..., 32:])
    assert torch.equal(capture.key[..., 32:], expected_flat_k[..., 32:])


def _tiny_checkpoint(model, config):
    checkpoint = {}
    for name, parameter in model.named_parameters():
        if name.endswith(".block_sparse_moe.experts.w13_weight") or name.endswith(
            ".block_sparse_moe.experts.w2_weight"
        ):
            continue
        if name.endswith(".self_attn.qkv_proj.weight"):
            prefix = name[: -len("qkv_proj.weight")]
            for shard_id in ("q", "k", "v"):
                checkpoint[prefix + f"{shard_id}_proj.weight"] = _random_fp8(
                    (128, 128)
                )
                checkpoint[
                    prefix + f"{shard_id}_proj.weight_scale_inv"
                ] = torch.rand(1, 1) + 0.1
        elif name.endswith(".self_attn.o_proj.weight"):
            checkpoint[name] = _random_fp8(parameter.shape)
            checkpoint[name[: -len(".weight")] + ".weight_scale_inv"] = (
                torch.rand(1, 1) + 0.1
            )
        elif name.endswith(".gate.e_score_correction_bias"):
            source_name = name.replace(
                ".gate.e_score_correction_bias",
                ".e_score_correction_bias",
            )
            checkpoint[source_name] = torch.randn(parameter.shape)
        else:
            checkpoint[name] = torch.randn(parameter.shape, dtype=parameter.dtype)

    for expert_id in range(config.num_local_experts):
        for projection in ("w1", "w2", "w3"):
            name = (
                f"model.layers.0.block_sparse_moe.experts.{expert_id}."
                f"{projection}.weight"
            )
            checkpoint[name] = _random_fp8((128, 128))
            checkpoint[name[: -len(".weight")] + ".weight_scale_inv"] = (
                torch.rand(1, 1) + 0.1
            )
    return checkpoint


def test_checkpoint_loader_loads_local_fp8_experts_and_tracks_remote_scales(
    tmp_path,
):
    torch.manual_seed(4)
    config = _config()
    context = _ep_context(1, 2)
    template = _instantiate_model(config, context)
    checkpoint = _tiny_checkpoint(template, config)
    save_file(checkpoint, tmp_path / "model.safetensors")
    target = _instantiate_model(config, context)

    load_model(target, str(tmp_path), tp_rank=0, tp_size=1)

    experts = target.model.layers[0].block_sparse_moe.experts
    expected_w1 = checkpoint[
        "model.layers.0.block_sparse_moe.experts.2.w1.weight"
    ]
    expected_w3 = checkpoint[
        "model.layers.0.block_sparse_moe.experts.2.w3.weight"
    ]
    assert torch.equal(experts.w13_weight[0, :128], expected_w1)
    assert torch.equal(experts.w13_weight[0, 128:], expected_w3)
    assert len(target._intentionally_skipped_expert_weights) == 6
    assert len(target._intentionally_skipped_expert_scales) == 6


def test_remote_expert_without_scale_fails_immediately():
    config = _config()
    model = _instantiate_model(config, _ep_context(1, 2))
    source_name = "model.layers.0.block_sparse_moe.experts.0.w1.weight"

    assert model.map_weight_name(source_name) is None
    with pytest.raises(ValueError, match="missing weight_scale_inv"):
        model.record_skipped_weight(source_name, (128, 128), "F8_E4M3", None, None)


def test_remote_expert_metadata_rejects_bad_weight_dtype_and_shape():
    config = _config()
    model = _instantiate_model(config, _ep_context(1, 2))
    source_name = "model.layers.0.block_sparse_moe.experts.0.w1.weight"

    with pytest.raises(TypeError, match="must be FP8 E4M3"):
        model.record_skipped_weight(
            source_name,
            (128, 128),
            "BF16",
            (1, 1),
            "F32",
        )
    with pytest.raises(ValueError, match="weight shape mismatch"):
        model.record_skipped_weight(
            source_name,
            (64, 128),
            "F8_E4M3",
            (1, 1),
            "F32",
        )


def test_local_expert_loader_rejects_missing_duplicate_and_bad_tensors():
    config = _config()
    with patch(
        "sparsevllm.models.minimax_m2.get_parallel_context",
        return_value=_ep_context(0, 1),
    ):
        experts = MiniMaxM2PackedExperts(config)
    weight = _random_fp8((128, 128))
    scale = torch.ones(1, 1, dtype=torch.float32)

    with pytest.raises(ValueError, match="Missing FP8 weight_scale_inv"):
        experts.load_expert_weight(0, "w1", weight, None)
    with pytest.raises(TypeError, match="FP8 E4M3"):
        experts.load_expert_weight(0, "w1", weight.float(), scale)
    with pytest.raises(TypeError, match="must be FP32"):
        experts.load_expert_weight(0, "w1", weight, scale.bfloat16())
    with pytest.raises(ValueError, match="shape mismatch"):
        experts.load_expert_weight(0, "w1", _random_fp8((128, 256)), scale)

    experts.load_expert_weight(0, "w1", weight, scale)
    with pytest.raises(ValueError, match="Duplicate MiniMax expert weight"):
        experts.load_expert_weight(0, "w1", weight, scale)
    with pytest.raises(ValueError, match="Missing local MiniMax expert"):
        experts.validate_loaded_weights()


def test_checkpoint_loader_fails_when_local_expert_scale_is_missing(tmp_path):
    torch.manual_seed(6)
    config = _config()
    context = _ep_context(0, 1)
    template = _instantiate_model(config, context)
    checkpoint = _tiny_checkpoint(template, config)
    del checkpoint[
        "model.layers.0.block_sparse_moe.experts.0.w1.weight_scale_inv"
    ]
    save_file(checkpoint, tmp_path / "model.safetensors")
    target = _instantiate_model(config, context)

    with pytest.raises(ValueError, match="Missing FP8 weight_scale_inv"):
        load_model(target, str(tmp_path), tp_rank=0, tp_size=1)


def _initialize_tiny_reference_weights(model):
    for parameter in model.parameters():
        if parameter.dtype == torch.float8_e4m3fn:
            parameter.data.copy_(
                (torch.randn(parameter.shape) * 0.25).to(torch.float8_e4m3fn)
            )
        elif parameter.ndim == 1 and parameter.shape[0] == 128:
            parameter.data.uniform_(0.9, 1.1)
        else:
            parameter.data.normal_(mean=0.0, std=0.05)
    for name, buffer in model.named_buffers():
        if name.endswith("weight_scale_inv"):
            buffer.uniform_(0.08, 0.12)


def _copy_to_transformers_reference(source, target):
    source_layer = source.model.layers[0]
    target_layer = target.model.layers[0]
    source_attention = source_layer.self_attn
    target_attention = target_layer.self_attn
    source_experts = source_layer.block_sparse_moe.experts

    with torch.no_grad():
        target.model.embed_tokens.weight.copy_(source.model.embed_tokens.weight)
        qkv_ranges = (
            (target_attention.q_proj, 0, source_attention.q_size),
            (
                target_attention.k_proj,
                source_attention.q_size,
                source_attention.kv_size,
            ),
            (
                target_attention.v_proj,
                source_attention.q_size + source_attention.kv_size,
                source_attention.kv_size,
            ),
        )
        for target_projection, row_start, row_count in qkv_ranges:
            scale_start = row_start // 128
            scale_count = row_count // 128
            target_projection.weight.copy_(
                fp8_blockwise_dequantize(
                    source_attention.qkv_proj.weight[
                        row_start : row_start + row_count
                    ],
                    source_attention.qkv_proj.weight_scale_inv[
                        scale_start : scale_start + scale_count
                    ],
                )
            )
        target_attention.o_proj.weight.copy_(
            fp8_blockwise_dequantize(
                source_attention.o_proj.weight,
                source_attention.o_proj.weight_scale_inv,
            )
        )
        target_attention.q_norm.weight.copy_(source_attention.q_norm.weight)
        target_attention.k_norm.weight.copy_(source_attention.k_norm.weight)
        target_layer.input_layernorm.weight.copy_(
            source_layer.input_layernorm.weight
        )
        target_layer.post_attention_layernorm.weight.copy_(
            source_layer.post_attention_layernorm.weight
        )
        target_layer.mlp.gate.weight.copy_(
            source_layer.block_sparse_moe.gate.weight
        )
        target_layer.mlp.e_score_correction_bias.copy_(
            source_layer.block_sparse_moe.gate.e_score_correction_bias
        )
        for expert_id in range(source_experts.num_experts):
            target_layer.mlp.experts.gate_up_proj[expert_id].copy_(
                fp8_blockwise_dequantize(
                    source_experts.w13_weight[expert_id],
                    source_experts.w13_scale_inv[expert_id],
                )
            )
            target_layer.mlp.experts.down_proj[expert_id].copy_(
                fp8_blockwise_dequantize(
                    source_experts.w2_weight[expert_id],
                    source_experts.w2_scale_inv[expert_id],
                )
            )
        target.model.norm.weight.copy_(source.model.norm.weight)
        target.lm_head.weight.copy_(source.lm_head.weight)


def test_tiny_dynamic_w8a8_model_stays_close_to_w8a16_transformers():
    torch.manual_seed(5)
    config = _config()
    context = _ep_context(0, 1)
    model = _instantiate_model(config, context).eval()
    _initialize_tiny_reference_weights(model)
    model.model.layers[0].self_attn.attn = _TinyCausalAttention(64**-0.5)

    transformers_config = TransformersMiniMaxM2Config(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        num_local_experts=config.num_local_experts,
        num_experts_per_tok=config.num_experts_per_tok,
        max_position_embeddings=config.max_position_embeddings,
        rms_norm_eps=config.rms_norm_eps,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": config.rope_theta,
            "partial_rotary_factor": config.rotary_dim / config.head_dim,
        },
        bos_token_id=None,
        eos_token_id=None,
        use_cache=False,
    )
    reference = TransformersMiniMaxM2ForCausalLM(transformers_config).eval()
    _copy_to_transformers_reference(model, reference)
    runtime_context = SimpleNamespace(
        now_layer_idx=0,
        cache_manager=_CacheRecorder(),
        is_prefill=False,
    )
    token_ids = torch.tensor([1, 7, 3, 9], dtype=torch.long)
    greedy_tokens = []

    for _ in range(2):
        positions = torch.arange(token_ids.shape[0], dtype=torch.long)
        with (
            patch(
                "sparsevllm.models.minimax_m2.get_context",
                return_value=runtime_context,
            ),
            patch(
                "sparsevllm.models.qwen3.get_context",
                return_value=runtime_context,
            ),
            patch(
                "sparsevllm.layers.embed_head.get_context",
                return_value=runtime_context,
            ),
        ):
            hidden_states = model(token_ids, positions)
            logits = model.compute_logits(hidden_states)
        reference_output = reference.model(
            input_ids=token_ids.view(1, -1),
            position_ids=positions.view(1, -1),
            use_cache=False,
        )
        reference_hidden = reference_output.last_hidden_state
        reference_logits = reference.lm_head(reference_hidden)

        reference_hidden = reference_hidden.squeeze(0)
        reference_logits = reference_logits.squeeze(0)
        hidden_relative_l2 = torch.linalg.vector_norm(
            hidden_states - reference_hidden
        ) / torch.linalg.vector_norm(reference_hidden)
        logits_relative_l2 = torch.linalg.vector_norm(
            logits - reference_logits
        ) / torch.linalg.vector_norm(reference_logits)
        assert 1.0e-3 < float(hidden_relative_l2.detach()) < 5.0e-2
        assert 1.0e-3 < float(logits_relative_l2.detach()) < 5.0e-2
        next_token = logits[-1].argmax()
        reference_next_token = reference_logits[-1].argmax()
        assert torch.equal(next_token, reference_next_token)
        greedy_tokens.append(int(next_token))
        token_ids = torch.cat((token_ids, next_token.view(1)))

    assert len(greedy_tokens) == 2
