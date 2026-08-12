from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparsevllm.config import Config
from sparsevllm.debug.tiny_random import apply_tiny_random_overrides
from sparsevllm.distributed import ParallelMode
from sparsevllm.method_registry import (
    DEEPSEEK_V4_DPA_EP_COMPATIBILITY,
    MODEL_RUNTIME_COMPATIBILITY,
)


def _official_config(**overrides):
    values = {
        "architectures": ["DeepseekV4ForCausalLM"],
        "model_type": "deepseek_v4",
        "vocab_size": 129280,
        "hidden_size": 4096,
        "intermediate_size": 2048,
        "moe_intermediate_size": 2048,
        "num_hidden_layers": 43,
        "num_attention_heads": 64,
        "num_key_value_heads": 1,
        "head_dim": 512,
        "q_lora_rank": 1024,
        "o_lora_rank": 1024,
        "o_groups": 8,
        "qk_rope_head_dim": 64,
        "index_n_heads": 64,
        "index_head_dim": 128,
        "index_topk": 512,
        "n_routed_experts": 256,
        "num_local_experts": 256,
        "num_experts_per_tok": 6,
        "expert_dtype": "fp4",
        "hc_mult": 4,
        "sliding_window": 128,
        "compress_rates": {
            "compressed_sparse_attention": 4,
            "heavily_compressed_attention": 128,
        },
        "num_nextn_predict_layers": 1,
        "max_position_embeddings": 1048576,
        "torch_dtype": torch.bfloat16,
        "layer_types": ["sliding_attention", "sliding_attention"]
        + ["compressed_sparse_attention", "heavily_compressed_attention"] * 20
        + ["compressed_sparse_attention"],
        "mlp_layer_types": ["hash_moe"] * 3 + ["moe"] * 40,
        "quantization_config": {
            "activation_scheme": "dynamic",
            "fmt": "e4m3",
            "quant_method": "fp8",
            "scale_fmt": "ue8m0",
            "weight_block_size": [128, 128],
        },
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_config(tmp_path, hf_config=None, **kwargs):
    hf_config = hf_config or _official_config()
    with patch(
        "sparsevllm.configs.runtime.AutoConfig.from_pretrained",
        return_value=hf_config,
    ):
        return Config(model=str(tmp_path), **kwargs)


@pytest.mark.parametrize("parallel_size", [1, 2, 4])
def test_deepseek_v4_accepts_overlapping_dpa_ep(tmp_path, parallel_size):
    config = _make_config(
        tmp_path,
        data_parallel_size=parallel_size,
        expert_parallel_size=parallel_size,
        decode_cuda_graph=True,
        enforce_eager=False,
    )

    assert config.uses_dpa_ep_layout
    assert config.world_size == parallel_size
    assert config.attention_tensor_parallel_size == 1
    assert config.moe_tensor_parallel_size == 1
    assert config.parallel_topology.mode is ParallelMode.DPA_EP
    assert (
        MODEL_RUNTIME_COMPATIBILITY[("deepseek_v4", ParallelMode.DPA_EP)]
        is DEEPSEEK_V4_DPA_EP_COMPATIBILITY
    )


@pytest.mark.parametrize(
    "parallel_kwargs",
    [
        {"tensor_parallel_size": 2},
        {"data_parallel_size": 2, "expert_parallel_size": 1},
        {"data_parallel_size": 1, "expert_parallel_size": 2},
        {"data_parallel_size": 3, "expert_parallel_size": 3},
    ],
)
def test_deepseek_v4_rejects_invalid_parallel_layout(tmp_path, parallel_kwargs):
    with pytest.raises(ValueError, match=r"DPA\+EP|DeepSeek V4"):
        _make_config(tmp_path, **parallel_kwargs)


def test_deepseek_v4_rejects_sparse_methods_and_prefix_cache(tmp_path):
    with pytest.raises(ValueError, match="validated methods"):
        _make_config(tmp_path, vllm_sparse_method="quest")
    with pytest.raises(ValueError, match="prefix caching"):
        _make_config(tmp_path, enable_prefix_caching=True)


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("head_dim", 128),
        ("index_topk", 256),
        ("n_routed_experts", 128),
        ("hc_mult", 2),
        ("num_nextn_predict_layers", 0),
        ("torch_dtype", torch.float16),
        ("expert_dtype", "bf16"),
    ],
)
def test_deepseek_v4_rejects_checkpoint_drift(tmp_path, field_name, invalid_value):
    expected_field = field_name.replace("n_routed", "routed").replace("torch_", "")
    with pytest.raises(ValueError, match=expected_field):
        _make_config(tmp_path, hf_config=_official_config(**{field_name: invalid_value}))


def test_deepseek_v4_tiny_random_shrinks_lists_and_disables_quantization(tmp_path):
    tiny_path = tmp_path / "tiny.json"
    tiny_path.write_text(
        '{"num_hidden_layers": 4, "hidden_size": 128, "intermediate_size": 64, '
        '"head_dim": 64, "num_attention_heads": 4, "q_lora_rank": 64, '
        '"o_lora_rank": 64, "o_groups": 2, "index_n_heads": 4, '
        '"index_head_dim": 64, "index_topk": 8, "num_local_experts": 8, '
        '"n_routed_experts": 8, "num_experts_per_tok": 2, "sliding_window": 8, '
        '"vocab_size": 128}',
        encoding="utf-8",
    )
    hf_config = _official_config()
    applied = apply_tiny_random_overrides(hf_config, str(tiny_path))

    assert applied["num_hidden_layers"] == 4
    assert hf_config.layer_types == [
        "sliding_attention",
        "sliding_attention",
        "compressed_sparse_attention",
        "heavily_compressed_attention",
    ]
    assert hf_config.mlp_layer_types == ["hash_moe", "hash_moe", "hash_moe", "moe"]

    config = _make_config(
        tmp_path,
        hf_config=_official_config(),
        tiny_random=True,
        tiny_random_config=str(tiny_path),
    )
    assert not config.quantization_config.enabled
