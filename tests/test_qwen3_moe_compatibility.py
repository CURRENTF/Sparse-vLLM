import pytest

from sparsevllm.method_registry import (
    MODEL_RUNTIME_COMPATIBILITY,
    QWEN3_MOE_EP_COMPATIBILITY,
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
        "pyramidkv",
        "omnikv",
        "quest",
        "rkv",
    }
    assert QWEN3_MOE_EP_COMPATIBILITY.prefix_cache_methods == {"", "omnikv", "quest"}


@pytest.mark.parametrize("method", sorted(QWEN3_MOE_EP_COMPATIBILITY.sparse_methods))
def test_qwen3_moe_registry_accepts_first_batch_sparse_methods(method):
    assert _validate(method) is QWEN3_MOE_EP_COMPATIBILITY


@pytest.mark.parametrize("method", ["", "omnikv", "quest"])
def test_qwen3_moe_registry_accepts_explicit_prefix_cache_methods(method):
    assert _validate(method, enable_prefix_caching=True) is QWEN3_MOE_EP_COMPATIBILITY


@pytest.mark.parametrize("method", ["streamingllm", "snapkv", "pyramidkv", "rkv"])
def test_qwen3_moe_registry_rejects_unvalidated_prefix_cache_methods(method):
    with pytest.raises(ValueError, match="prefix caching is validated only"):
        _validate(method, enable_prefix_caching=True)


def test_qwen3_moe_registry_rejects_conditional_and_out_of_scope_methods():
    with pytest.raises(NotImplementedError, match="steering asset"):
        _validate("skipkv")
    with pytest.raises(NotImplementedError, match="not part of the validated"):
        _validate("deltakv")


def test_qwen3_moe_registry_rejects_unvalidated_parallel_and_graph_modes():
    with pytest.raises(ValueError, match="requires TP=1 and DP=1"):
        _validate(tensor_parallel_size=2)
    with pytest.raises(ValueError, match="enforce_eager=True"):
        _validate(enforce_eager=False)
    with pytest.raises(ValueError, match="does not support decode_cuda_graph"):
        _validate(decode_cuda_graph=True)


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
