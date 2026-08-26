from __future__ import annotations

import pytest
from sparsevllm.engine.cache_manager.storage import CacheLayout
from glm_test_helpers import _glm_config


def test_glm_config_selects_mla_latent_layout():
    config = _glm_config()

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value


def test_mla_prefill_workspace_budget_must_be_positive():
    with pytest.raises(ValueError, match="mla_prefill_workspace_bytes"):
        _glm_config(mla_prefill_workspace_bytes=0)


def test_glm_config_accepts_latent_quest_without_prefix_cache():
    config = _glm_config(
        vllm_sparse_method="quest",
        decode_cuda_graph=True,
    )

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value
    assert config.vllm_sparse_method == "quest"
    assert config.decode_cuda_graph is True
    assert config.enable_prefix_caching is False


def test_glm_config_rejects_latent_quest_prefix_cache():
    with pytest.raises(ValueError, match="prefix caching is validated only"):
        _glm_config(
            vllm_sparse_method="quest",
            enable_prefix_caching=True,
        )


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
