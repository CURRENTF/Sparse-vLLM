import pytest

from sparsevllm.distributed import ParallelMode, ParallelTopology
from sparsevllm.method_registry import (
    DENSE_MODEL_COMPATIBILITY,
    MODEL_RUNTIME_COMPATIBILITY,
    QWEN35_MOE_COMPATIBILITY,
    QWEN3_MOE_EP_COMPATIBILITY,
    QWEN3_MOE_TP_COMPATIBILITY,
    QWEN3_MOE_TP_EP_COMPATIBILITY,
    validate_model_runtime_compatibility,
)


def _validate(method="", **overrides):
    values = {
        "model_type": "qwen3_moe",
        "sparse_method": method,
        "tensor_parallel_size": 1,
        "expert_parallel_size": 2,
        "data_parallel_size": 1,
        "decode_cuda_graph": False,
        "enable_prefix_caching": False,
    }
    values.update(overrides)
    tp_size = values.pop("tensor_parallel_size")
    ep_size = values.pop("expert_parallel_size")
    dp_size = values.pop("data_parallel_size")
    values["topology"] = ParallelTopology(
        tp_size,
        ep_size,
        dp_size,
        ParallelMode.OUTER_TP_MOE if tp_size > 1 else ParallelMode.STANDARD,
    )
    return validate_model_runtime_compatibility(**values)


def test_qwen3_moe_registry_lists_only_v1_validated_combinations():
    assert (
        MODEL_RUNTIME_COMPATIBILITY["qwen3_moe", ParallelMode.STANDARD]
        is QWEN3_MOE_EP_COMPATIBILITY
    )
    assert QWEN3_MOE_EP_COMPATIBILITY.sparse_methods == {
        "",
        "streamingllm",
        "snapkv",
        "h2o",
        "pyramidkv",
        "omnikv",
        "quest",
        "rkv",
    }
    assert QWEN3_MOE_EP_COMPATIBILITY.prefix_cache_methods == {
        "",
        "omnikv",
        "quest",
        "snapkv",
        "h2o",
        "pyramidkv",
        "rkv",
    }
    assert QWEN3_MOE_EP_COMPATIBILITY.decode_cuda_graph_methods == {
        "",
        "streamingllm",
        "snapkv",
        "h2o",
        "pyramidkv",
        "omnikv",
        "quest",
        "rkv",
    }


def test_qwen35_moe_registry_accepts_vanilla_prefix_cache():
    assert validate_model_runtime_compatibility(
        model_type="qwen3_5_moe",
        sparse_method="",
        topology=ParallelTopology(2, 2, 1, ParallelMode.OUTER_TP_MOE),
        decode_cuda_graph=False,
        enable_prefix_caching=True,
    ) is QWEN35_MOE_COMPATIBILITY


@pytest.mark.parametrize("method", sorted(QWEN3_MOE_EP_COMPATIBILITY.sparse_methods))
def test_qwen3_moe_registry_accepts_first_batch_sparse_methods(method):
    assert _validate(method) is QWEN3_MOE_EP_COMPATIBILITY


@pytest.mark.parametrize(
    "method",
    ["", "omnikv", "quest", "snapkv", "h2o", "pyramidkv", "rkv"],
)
def test_qwen3_moe_registry_accepts_explicit_prefix_cache_methods(method):
    assert _validate(method, enable_prefix_caching=True) is QWEN3_MOE_EP_COMPATIBILITY


@pytest.mark.parametrize("method", ["streamingllm"])
def test_qwen3_moe_registry_rejects_unvalidated_prefix_cache_methods(method):
    with pytest.raises(ValueError, match="prefix caching is validated only"):
        _validate(method, enable_prefix_caching=True)


def test_qwen3_moe_registry_rejects_conditional_and_out_of_scope_methods():
    with pytest.raises(ValueError, match="validated methods"):
        _validate("skipkv")
    with pytest.raises(ValueError, match="validated methods"):
        _validate("deltakv")


def test_qwen3_moe_registry_rejects_outer_tp_with_dp():
    with pytest.raises(ValueError, match="requires DP=1"):
        _validate(tensor_parallel_size=2, data_parallel_size=2)


def test_qwen3_moe_registry_accepts_vanilla_pure_tp_modes():
    for tp_size in (2, 4):
        assert _validate(
            tensor_parallel_size=tp_size,
            expert_parallel_size=1,
        ) is QWEN3_MOE_TP_COMPATIBILITY
        assert _validate(
            tensor_parallel_size=tp_size,
            expert_parallel_size=1,
            decode_cuda_graph=True,
            enable_prefix_caching=True,
        ) is QWEN3_MOE_TP_COMPATIBILITY


@pytest.mark.parametrize(
    "method",
    sorted(QWEN3_MOE_TP_EP_COMPATIBILITY.decode_cuda_graph_methods),
)
@pytest.mark.parametrize("expert_parallel_size", [1, 2])
def test_qwen3_moe_registry_accepts_tp_sparse_graph_modes(
    method,
    expert_parallel_size,
):
    assert _validate(
        method,
        tensor_parallel_size=4,
        expert_parallel_size=expert_parallel_size,
        decode_cuda_graph=True,
    ) is QWEN3_MOE_TP_COMPATIBILITY


def test_qwen3_moe_tp_sparse_methods_exclude_deltakv_and_skipkv():
    assert "deltakv" not in QWEN3_MOE_TP_EP_COMPATIBILITY.sparse_methods
    assert "skipkv" not in QWEN3_MOE_TP_EP_COMPATIBILITY.sparse_methods
    for method in ("deltakv", "skipkv"):
        with pytest.raises((ValueError, NotImplementedError)):
            _validate(method, tensor_parallel_size=4, expert_parallel_size=2)


@pytest.mark.parametrize(
    "method",
    sorted(QWEN3_MOE_EP_COMPATIBILITY.decode_cuda_graph_methods),
)
def test_qwen3_moe_registry_accepts_decode_cuda_graph(method):
    assert (
        _validate(method, decode_cuda_graph=True)
        is QWEN3_MOE_EP_COMPATIBILITY
    )
def test_dense_models_use_shared_runtime_compatibility():
    assert validate_model_runtime_compatibility(
        model_type="qwen3",
        sparse_method="deltakv",
        topology=ParallelTopology(1, 1, 1),
        decode_cuda_graph=False,
        enable_prefix_caching=False,
    ) is DENSE_MODEL_COMPATIBILITY


def test_all_dense_architectures_are_registered():
    assert {
        model_type
        for model_type, mode in MODEL_RUNTIME_COMPATIBILITY
        if mode is ParallelMode.STANDARD
    } >= {"qwen2", "qwen3", "qwen3_5", "llama"}
