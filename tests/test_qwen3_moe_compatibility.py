import pytest

from sparsevllm.method_registry import (
    MODEL_RUNTIME_COMPATIBILITY,
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
        "enforce_eager": True,
        "decode_cuda_graph": False,
        "enable_prefix_caching": False,
    }
    values.update(overrides)
    return validate_model_runtime_compatibility(**values)


def test_qwen3_moe_registry_lists_only_v1_validated_combinations():
    assert MODEL_RUNTIME_COMPATIBILITY["qwen3_moe"] is QWEN3_MOE_EP_COMPATIBILITY
    assert QWEN3_MOE_EP_COMPATIBILITY.parallel_mode == "ep_replicated_kv"
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
    assert QWEN3_MOE_EP_COMPATIBILITY.requires_eager is False
    assert QWEN3_MOE_EP_COMPATIBILITY.decode_cuda_graph_methods == {
        "",
        "streamingllm",
        "snapkv",
        "pyramidkv",
        "omnikv",
        "quest",
        "rkv",
    }


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
    with pytest.raises(NotImplementedError, match="steering asset"):
        _validate("skipkv")
    with pytest.raises(NotImplementedError, match="not part of the validated"):
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
            enforce_eager=False,
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
        enforce_eager=False,
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
        _validate(method, enforce_eager=False, decode_cuda_graph=True)
        is QWEN3_MOE_EP_COMPATIBILITY
    )


def test_qwen3_moe_registry_rejects_unvalidated_h2o_decode_cuda_graph():
    with pytest.raises(ValueError, match="decode_cuda_graph is validated only"):
        _validate("h2o", enforce_eager=False, decode_cuda_graph=True)


def test_dense_models_do_not_inherit_qwen3_moe_compatibility():
    assert validate_model_runtime_compatibility(
        model_type="qwen3",
        sparse_method="deltakv",
        tensor_parallel_size=1,
        expert_parallel_size=1,
        data_parallel_size=1,
        enforce_eager=True,
        decode_cuda_graph=False,
        enable_prefix_caching=False,
    ) is None
