from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparsevllm.config import Config
from sparsevllm.method_registry import (
    MINIMAX_M2_EP_COMPATIBILITY,
    MODEL_RUNTIME_COMPATIBILITY,
    validate_model_runtime_compatibility,
)


def _quantization_config(**overrides):
    values = {
        "quant_method": "fp8",
        "fmt": "float8_e4m3fn",
        "activation_scheme": "dynamic",
        "weight_block_size": [128, 128],
        "modules_to_not_convert": [
            "gate",
            "e_score_correction_bias",
            "lm_head",
        ],
    }
    values.update(overrides)
    return values


def _official_config(**overrides):
    values = {
        "architectures": ["MiniMaxM2ForCausalLM"],
        "model_type": "minimax_m2",
        "vocab_size": 200064,
        "hidden_size": 3072,
        "intermediate_size": 1536,
        "num_hidden_layers": 62,
        "num_attention_heads": 48,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "rotary_dim": 64,
        "num_local_experts": 256,
        "num_experts_per_tok": 8,
        "max_position_embeddings": 204800,
        "shared_intermediate_size": 0,
        "mtp_transformer_layers": 1,
        "num_mtp_modules": 3,
        "hidden_act": "silu",
        "qk_norm_type": "per_layer",
        "scoring_func": "sigmoid",
        "use_qk_norm": True,
        "use_routing_bias": True,
        "use_mtp": True,
        "tie_word_embeddings": False,
        "torch_dtype": torch.bfloat16,
        "quantization_config": _quantization_config(),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_config(tmp_path, hf_config=None, **kwargs):
    if hf_config is None:
        hf_config = _official_config()
    with patch(
        "sparsevllm.config.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        return Config(model=str(tmp_path), **kwargs)


@pytest.mark.parametrize("expert_parallel_size", [1, 2, 4, 8])
@pytest.mark.parametrize("moe_backend", ["pytorch", "native", "triton"])
def test_minimax_config_accepts_first_milestone_runtime(
    tmp_path,
    expert_parallel_size,
    moe_backend,
):
    config = _make_config(
        tmp_path,
        expert_parallel_size=expert_parallel_size,
        moe_backend=moe_backend,
        enable_prefix_caching=True,
        decode_cuda_graph=moe_backend == "triton",
        enforce_eager=moe_backend != "triton",
    )

    assert config.hf_config.model_type == "minimax_m2"
    assert config.quantization_config.model_name == "MiniMax M2.7"
    assert config.quantization_config.weight_block_size == (128, 128)
    assert config.moe_backend == moe_backend


@pytest.mark.parametrize("moe_backend", ["pytorch", "native"])
def test_minimax_config_rejects_graph_for_eager_expert_backends(
    tmp_path,
    moe_backend,
):
    with pytest.raises(ValueError, match="moe_backend='triton'"):
        _make_config(
            tmp_path,
            moe_backend=moe_backend,
            decode_cuda_graph=True,
            enforce_eager=False,
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("hidden_size", 4096),
        ("rotary_dim", 128),
        ("qk_norm_type", "per_head"),
        ("scoring_func", "softmax"),
        ("use_mtp", False),
        ("num_mtp_modules", 0),
    ],
)
def test_minimax_config_rejects_checkpoint_drift(
    tmp_path,
    field_name,
    invalid_value,
):
    hf_config = _official_config(**{field_name: invalid_value})
    with pytest.raises(ValueError, match=field_name):
        _make_config(tmp_path, hf_config=hf_config)


def test_minimax_config_requires_all_fp8_exclusions(tmp_path):
    hf_config = _official_config(
        quantization_config=_quantization_config(
            modules_to_not_convert=["gate", "lm_head"],
        )
    )
    with pytest.raises(ValueError, match="e_score_correction_bias"):
        _make_config(tmp_path, hf_config=hf_config)


@pytest.mark.parametrize(
    "parallel_kwargs",
    [
        {"tensor_parallel_size": 2},
        {"data_parallel_size": 2},
        {"expert_parallel_size": 3},
    ],
)
def test_minimax_config_rejects_unvalidated_parallel_layout(
    tmp_path,
    parallel_kwargs,
):
    with pytest.raises(ValueError, match="MiniMax M2.7"):
        _make_config(tmp_path, **parallel_kwargs)


def _validate(method="", **overrides):
    values = {
        "model_type": "minimax_m2",
        "sparse_method": method,
        "tensor_parallel_size": 1,
        "expert_parallel_size": 4,
        "data_parallel_size": 1,
        "enforce_eager": False,
        "decode_cuda_graph": True,
        "enable_prefix_caching": True,
    }
    values.update(overrides)
    return validate_model_runtime_compatibility(**values)


def test_minimax_compatibility_matches_qwen3_moe_sparse_runtime():
    assert MODEL_RUNTIME_COMPATIBILITY["minimax_m2"] is MINIMAX_M2_EP_COMPATIBILITY
    assert MINIMAX_M2_EP_COMPATIBILITY.parallel_mode == "ep_replicated_kv"
    assert MINIMAX_M2_EP_COMPATIBILITY.sparse_methods == {
        "",
        "streamingllm",
        "snapkv",
        "pyramidkv",
        "omnikv",
        "quest",
        "rkv",
    }
    assert MINIMAX_M2_EP_COMPATIBILITY.prefix_cache_methods == {
        "",
        "omnikv",
        "quest",
    }
    assert MINIMAX_M2_EP_COMPATIBILITY.decode_cuda_graph_methods == {""}
    assert _validate() is MINIMAX_M2_EP_COMPATIBILITY


@pytest.mark.parametrize("method", sorted(MINIMAX_M2_EP_COMPATIBILITY.sparse_methods))
def test_minimax_compatibility_accepts_non_deltakv_sparse_methods(method):
    assert (
        _validate(method, decode_cuda_graph=False, enable_prefix_caching=False)
        is MINIMAX_M2_EP_COMPATIBILITY
    )


@pytest.mark.parametrize("method", ["", "omnikv", "quest"])
def test_minimax_compatibility_accepts_prefix_cache_methods(method):
    assert (
        _validate(method, decode_cuda_graph=False, enable_prefix_caching=True)
        is MINIMAX_M2_EP_COMPATIBILITY
    )


@pytest.mark.parametrize("method", ["streamingllm", "snapkv", "pyramidkv", "rkv"])
def test_minimax_compatibility_rejects_unvalidated_prefix_cache_methods(method):
    with pytest.raises(ValueError, match="prefix caching is validated only"):
        _validate(method, decode_cuda_graph=False, enable_prefix_caching=True)


@pytest.mark.parametrize("method", ["deltakv", "skipkv"])
def test_minimax_compatibility_rejects_out_of_scope_sparse_methods(method):
    with pytest.raises(ValueError, match="validated methods"):
        _validate(method, decode_cuda_graph=False, enable_prefix_caching=False)


@pytest.mark.parametrize(
    "method",
    ["streamingllm", "snapkv", "pyramidkv", "omnikv", "quest", "rkv"],
)
def test_minimax_config_accepts_non_deltakv_sparse_methods(tmp_path, method):
    config = _make_config(
        tmp_path,
        vllm_sparse_method=method,
        enforce_eager=True,
        decode_cuda_graph=False,
        enable_prefix_caching=False,
    )

    assert config.vllm_sparse_method == method


@pytest.mark.parametrize(
    "method",
    ["streamingllm", "snapkv", "pyramidkv", "omnikv", "quest", "rkv"],
)
def test_minimax_sparse_methods_require_eager_decode(tmp_path, method):
    with pytest.raises(ValueError, match="decode_cuda_graph is validated only"):
        _make_config(
            tmp_path,
            vllm_sparse_method=method,
            moe_backend="triton",
            enforce_eager=False,
            decode_cuda_graph=True,
            enable_prefix_caching=False,
        )
