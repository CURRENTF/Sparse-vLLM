from __future__ import annotations

import pytest
import torch

from sparsevllm.engine.cache_manager.storage import CacheLayout
from sparsevllm.method_registry import (
    GLM4_MOE_LITE_EP_COMPATIBILITY,
    MODEL_RUNTIME_COMPATIBILITY,
)
from sparsevllm.distributed import ParallelMode

from glm_test_helpers import _glm_config


def test_glm_config_selects_mla_latent_layout():
    config = _glm_config()

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value
    assert config.mla_prefill_workspace_bytes == 2 * 1024**3


def test_mla_prefill_workspace_budget_must_be_positive():
    with pytest.raises(ValueError, match="mla_prefill_workspace_bytes"):
        _glm_config(mla_prefill_workspace_bytes=0)


@pytest.mark.parametrize(
    ("override", "error_type", "message"),
    [
        ({"vllm_sparse_method": "quest"}, ValueError, "Unsupported glm4_moe_lite"),
        (
            {"enforce_eager": False},
            ValueError,
            "requires enforce_eager=True",
        ),
    ],
)
def test_glm_config_rejects_unsupported_storage_combinations(
    override, error_type, message
):
    with pytest.raises(error_type, match=message):
        _glm_config(**override)


@pytest.mark.parametrize("expert_parallel_size", [2, 4])
def test_glm_config_accepts_replicated_attention_ep(expert_parallel_size):
    config = _glm_config(
        tensor_parallel_size=1,
        expert_parallel_size=expert_parallel_size,
        data_parallel_size=1,
    )

    assert config.world_size == expert_parallel_size
    assert config.tensor_parallel_size == 1
    assert config.expert_parallel_size == expert_parallel_size
    assert config.data_parallel_size == 1
    assert not config.uses_outer_tp_moe_layout


@pytest.mark.parametrize(
    (
        "tensor_parallel_size",
        "expert_parallel_size",
        "world_size",
        "moe_tensor_parallel_size",
    ),
    [
        (2, 2, 2, 1),
        (4, 2, 4, 2),
        (4, 4, 4, 1),
    ],
)
def test_glm_config_accepts_outer_tp_moe_ep_layout(
    tensor_parallel_size,
    expert_parallel_size,
    world_size,
    moe_tensor_parallel_size,
):
    config = _glm_config(
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
    )

    assert config.uses_outer_tp_moe_layout
    assert config.world_size == world_size
    assert config.moe_tensor_parallel_size == moe_tensor_parallel_size


def test_glm_hybrid_checks_routed_width_against_moe_tp_not_outer_tp():
    config = _glm_config(
        tensor_parallel_size=4,
        expert_parallel_size=2,
        hf_overrides={"moe_intermediate_size": 6},
    )

    assert config.moe_tensor_parallel_size == 2


def test_glm_hybrid_rejects_routed_width_not_divisible_by_moe_tp():
    with pytest.raises(ValueError, match="divisible by MoE TP"):
        _glm_config(
            tensor_parallel_size=4,
            expert_parallel_size=2,
            hf_overrides={"moe_intermediate_size": 5},
        )


def test_glm_config_rejects_nondivisible_outer_tp_moe_ep_layout():
    with pytest.raises(ValueError, match="TP divisible by EP"):
        _glm_config(tensor_parallel_size=2, expert_parallel_size=4)


def test_glm_config_rejects_data_parallelism():
    with pytest.raises(ValueError, match="does not support data parallelism"):
        _glm_config(data_parallel_size=2)


_GLM_PARALLEL_LAYOUTS = [
    (1, 1),
    (2, 1),
    (4, 1),
    (1, 2),
    (1, 4),
    (2, 2),
    (4, 2),
    (4, 4),
]


@pytest.mark.parametrize(
    ("tensor_parallel_size", "expert_parallel_size"),
    _GLM_PARALLEL_LAYOUTS,
)
@pytest.mark.parametrize(
    "method",
    ["", "streamingllm", "snapkv", "h2o", "omnikv", "rkv"],
)
def test_glm_config_accepts_parallel_sparse_graph_cross_product(
    tensor_parallel_size,
    expert_parallel_size,
    method,
):
    config = _glm_config(
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
        vllm_sparse_method=method,
        decode_cuda_graph=True,
    )

    assert config.decode_cuda_graph
    assert config.vllm_sparse_method == method


@pytest.mark.parametrize(
    ("tensor_parallel_size", "expert_parallel_size"),
    _GLM_PARALLEL_LAYOUTS,
)
@pytest.mark.parametrize(
    "method",
    ["", "streamingllm", "snapkv", "h2o", "omnikv", "rkv"],
)
def test_glm_config_accepts_parallel_sparse_graph_prefix_cross_product(
    tensor_parallel_size,
    expert_parallel_size,
    method,
):
    config = _glm_config(
        tensor_parallel_size=tensor_parallel_size,
        expert_parallel_size=expert_parallel_size,
        vllm_sparse_method=method,
        decode_cuda_graph=True,
        enable_prefix_caching=True,
    )

    assert config.enable_prefix_caching
    assert config.decode_cuda_graph
    assert config.resolved_prefix_cache_mode == (
        "radix" if method in {"", "omnikv"} else "chain"
    )


def test_glm_registry_selects_parallel_layout_contracts():
    assert MODEL_RUNTIME_COMPATIBILITY[
        ("glm4_moe_lite", ParallelMode.STANDARD)
    ] is GLM4_MOE_LITE_EP_COMPATIBILITY
    assert MODEL_RUNTIME_COMPATIBILITY[
        ("glm4_moe_lite", ParallelMode.OUTER_TP_MOE)
    ] is GLM4_MOE_LITE_EP_COMPATIBILITY


@pytest.mark.parametrize(
    "method",
    ["", "streamingllm", "snapkv", "h2o", "omnikv", "rkv"],
)
def test_glm_config_accepts_tp1_ep1_decode_cuda_graph(method):
    config = _glm_config(
        decode_cuda_graph=True,
        vllm_sparse_method=method,
    )

    assert config.decode_cuda_graph
    assert config.tensor_parallel_size == 1
    assert config.expert_parallel_size == 1
    assert config.vllm_sparse_method == method


def test_glm_config_accepts_vanilla_latent_prefix_cache():
    config = _glm_config(enable_prefix_caching=True)

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value
    assert config.enable_prefix_caching
    assert config.resolved_prefix_cache_mode == "radix"


def test_glm_config_accepts_omnikv_latent_prefix_cache_after_lifecycle_gate():
    config = _glm_config(
        vllm_sparse_method="omnikv",
        enable_prefix_caching=True,
    )

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value
    assert config.enable_prefix_caching
    assert config.resolved_prefix_cache_mode == "radix"


def test_glm_config_accepts_snapkv_latent_chain_prefix_after_lifecycle_gate():
    config = _glm_config(
        vllm_sparse_method="snapkv",
        enable_prefix_caching=True,
    )

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value
    assert config.enable_prefix_caching
    assert config.resolved_prefix_cache_mode == "chain"


def test_glm_config_accepts_h2o_latent_chain_prefix_after_lifecycle_gate():
    config = _glm_config(
        vllm_sparse_method="h2o",
        enable_prefix_caching=True,
    )

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value
    assert config.enable_prefix_caching
    assert config.resolved_prefix_cache_mode == "chain"


def test_glm_config_accepts_rkv_latent_chain_prefix_after_lifecycle_gate():
    config = _glm_config(
        vllm_sparse_method="rkv",
        enable_prefix_caching=True,
    )

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value
    assert config.enable_prefix_caching
    assert config.resolved_prefix_cache_mode == "chain"


def test_glm_config_accepts_streamingllm_chain_prefix_cache():
    config = _glm_config(
        vllm_sparse_method="streamingllm",
        enable_prefix_caching=True,
    )

    assert config.resolved_prefix_cache_mode == "chain"


@pytest.mark.parametrize(
    "method",
    ["streamingllm", "snapkv", "h2o", "omnikv", "rkv"],
)
def test_glm_config_accepts_sparse_latent_layout(method):
    config = _glm_config(
        vllm_sparse_method=method,
        num_sink_tokens=2,
        num_recent_tokens=3,
    )

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value
    assert config.vllm_sparse_method == method
