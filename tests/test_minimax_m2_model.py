from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from safetensors.torch import save_file

from sparsevllm.config import QuantizationConfig
from sparsevllm.distributed import ParallelContext, ParallelGroup
from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.models.minimax_m2 import (
    MiniMaxM2Attention,
    MiniMaxM2ForCausalLM,
    MiniMaxM2PackedExperts,
)
from sparsevllm.utils.loader import load_model


def _rmsnorm_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    *,
    weight_offset: float = 0.0,
) -> torch.Tensor:
    x_float = x.float()
    normalized = x_float * torch.rsqrt(
        x_float.square().mean(dim=-1, keepdim=True) + eps
    )
    return (normalized * (weight.float() + weight_offset)).to(x.dtype)


class _ReferenceRMSNorm(torch.nn.Module):
    def __init__(self, norm: RMSNorm) -> None:
        super().__init__()
        self.eps = norm.eps
        self.weight = torch.nn.Parameter(norm.weight.detach().clone())

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return _rmsnorm_reference(x, self.weight, self.eps)
        merged = x.float() + residual.float()
        merged_output = merged.to(x.dtype)
        return _rmsnorm_reference(merged, self.weight, self.eps).to(x.dtype), merged_output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("hidden_size", [1024, 3072, 6144])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_flashinfer_rmsnorm_matches_reference_for_minimax_shapes(hidden_size, dtype):
    pytest.importorskip("flashinfer")
    torch.manual_seed(37)
    norm = RMSNorm(hidden_size).cuda().to(dtype)
    norm.weight.data.normal_(mean=1.0, std=0.2)
    storage = torch.randn(7, 8192, device="cuda", dtype=dtype)
    x = storage[:, :hidden_size]
    original = x.clone()
    expected = _rmsnorm_reference(x, norm.weight, norm.eps)

    actual = norm(x)

    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=2.0e-2)
    assert torch.equal(x, original)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_flashinfer_fused_add_rmsnorm_matches_reference():
    pytest.importorskip("flashinfer")
    torch.manual_seed(41)
    norm = RMSNorm(3072).cuda().to(torch.bfloat16)
    norm.weight.data.normal_(mean=1.0, std=0.2)
    x = torch.randn(7, 3072, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    merged = x.float() + residual.float()
    expected_residual = merged.to(x.dtype)
    expected = _rmsnorm_reference(merged, norm.weight, norm.eps).to(x.dtype)

    actual, actual_residual = norm(x, residual)

    torch.testing.assert_close(actual, expected, rtol=1.0e-2, atol=2.0e-2)
    assert torch.equal(actual_residual, expected_residual)
    assert actual is x
    assert actual_residual is residual


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
    attention.rotary_emb.backend = "torch"
    attention.q_norm = _ReferenceRMSNorm(attention.q_norm)
    attention.k_norm = _ReferenceRMSNorm(attention.k_norm)
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
    expected_flat_q = attention.q_norm(raw_q).view(3, 2, 64)
    expected_flat_k = attention.k_norm(raw_k).view(3, 2, 64)
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
    assert torch.equal(experts.w13_weight[0, :128], expected_w3)
    assert torch.equal(experts.w13_weight[0, 128:], expected_w1)
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
