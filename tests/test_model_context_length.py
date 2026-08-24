from types import MethodType, SimpleNamespace

import pytest

from sparsevllm.configs.model import _finalize_model_config, _model_context_length
from sparsevllm.configs.runtime import Config
from sparsevllm.engine.cache_manager.standard import StandardCacheManager


def test_model_context_length_applies_rope_scaling():
    hf_config = SimpleNamespace(
        max_position_embeddings=32768,
        rope_scaling={"rope_type": "linear", "factor": 4.0},
    )

    assert _model_context_length(hf_config) == 131072


def test_model_context_length_accepts_rope_parameters():
    hf_config = SimpleNamespace(
        max_position_embeddings=32768,
        rope_parameters={"rope_type": "linear", "factor": 2.0},
    )

    assert _model_context_length(hf_config) == 65536


def test_model_context_length_accepts_per_layer_rope_parameters():
    hf_config = SimpleNamespace(
        max_position_embeddings=262144,
        rope_parameters={
            "full_attention": {"rope_type": "proportional"},
            "sliding_attention": {"rope_type": "default"},
        },
    )

    assert _model_context_length(hf_config) == 262144


@pytest.mark.parametrize(
    "rope_scaling",
    [
        {"rope_type": "llama3", "factor": 8.0},
        {"factor": 4.0, "original_max_position_embeddings": 32768},
    ],
)
def test_model_context_length_does_not_double_apply_rope_scaling(rope_scaling):
    hf_config = SimpleNamespace(
        max_position_embeddings=131072,
        rope_scaling=rope_scaling,
    )

    assert _model_context_length(hf_config) == 131072


def test_auto_max_model_len_uses_model_context_length(monkeypatch):
    config = SimpleNamespace(
        hf_config=SimpleNamespace(max_position_embeddings=32768),
        max_model_len=None,
        max_num_seqs_in_batch=32,
    )
    monkeypatch.setattr(
        "sparsevllm.configs.model.RuntimeLayout.from_config",
        lambda *_args, **_kwargs: object(),
    )

    _finalize_model_config(config, SimpleNamespace(mixed_attention=False))

    assert config.max_model_len == 32768
    assert config.max_model_len_auto is True


def test_explicit_max_model_len_cannot_exceed_model_context(monkeypatch):
    config = SimpleNamespace(
        hf_config=SimpleNamespace(max_position_embeddings=32768),
        max_model_len=32769,
        max_num_seqs_in_batch=32,
    )
    monkeypatch.setattr(
        "sparsevllm.configs.model.RuntimeLayout.from_config",
        lambda *_args, **_kwargs: object(),
    )

    with pytest.raises(ValueError, match="exceeds the model context length"):
        _finalize_model_config(config, SimpleNamespace(mixed_attention=False))


class _Storage:
    def allocate(self, **kwargs):
        self.allocation = kwargs


def _standard_manager(*, max_model_len, auto):
    config = SimpleNamespace(
        max_model_len=max_model_len,
        max_model_len_auto=auto,
        decode_graph=False,
        decode_graph_context_sizes_auto=False,
        prefix_cache_max_blocks=None,
    )
    config.limit_auto_max_model_len = MethodType(Config.limit_auto_max_model_len, config)
    manager = object.__new__(StandardCacheManager)
    manager.config = config
    manager.num_kv_layers = 2
    manager.max_buffer_rows = 2
    manager.max_model_len = max_model_len
    manager.device = "cpu"
    manager.attention_cache_storage = _Storage()
    manager._get_available_slots_info = lambda: (1000, 10)
    return manager


def test_standard_cache_limits_auto_length_to_runtime_capacity():
    manager = _standard_manager(max_model_len=100, auto=True)

    manager.allocate_kv_cache()

    assert manager.max_model_len == 31
    assert manager.config.num_kvcache_slots == 31
    assert manager.attention_cache_storage.allocation["num_slots"] == 31


def test_runtime_capacity_bounds_auto_cuda_graph_context_sizes():
    config = SimpleNamespace(
        max_model_len=262144,
        max_model_len_auto=True,
        decode_graph=True,
        decode_graph_context_sizes_auto=True,
        decode_graph_context_sizes=[1024, 262144],
    )

    Config.limit_auto_max_model_len(config, 9000)

    assert config.max_model_len == 9000
    context_sizes = config.decode_graph_context_sizes
    assert context_sizes == sorted(set(context_sizes))
    assert context_sizes[-1] == config.max_model_len
    assert all(0 < size <= config.max_model_len for size in context_sizes)


def test_standard_cache_rejects_explicit_length_above_runtime_capacity():
    manager = _standard_manager(max_model_len=100, auto=False)

    with pytest.raises(RuntimeError, match="capacity is smaller than max_model_len"):
        manager.allocate_kv_cache()
