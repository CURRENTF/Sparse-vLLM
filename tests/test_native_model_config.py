import json
from types import SimpleNamespace

import pytest
import torch

from sparsevllm.configs.model import _load_model_config
from sparsevllm.method_registry import resolve_model_sparse_method


def test_native_checkpoint_metadata_survives_transformers_config_conversion(tmp_path, monkeypatch):
    # Transformers renames compression/hash fields and nests the native YaRN
    # metadata. The model consumes the original checkpoint contract.
    raw = {
        "model_type": "deepseek_v4",
        "num_hash_layers": 2,
        "compress_ratios": [0, 4, 128],
        "rope_scaling": {"type": "yarn", "factor": 16},
    }
    (tmp_path / "config.json").write_text(json.dumps(raw))
    monkeypatch.setattr(
        "sparsevllm.configs.model.AutoConfig.from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(
            model_type="deepseek_v4", compress_rates=raw["compress_ratios"],
            rope_scaling={"compress": raw["rope_scaling"]},
        ),
    )

    config = _load_model_config(str(tmp_path))

    for field, value in raw.items():
        assert getattr(config, field) == value


def test_native_attention_is_resolved_when_no_method_is_requested():
    resolved = resolve_model_sparse_method("deepseek_v4", "")
    assert resolved
    assert resolve_model_sparse_method("deepseek_v4", resolved) == resolved


@pytest.mark.parametrize("method", ["h2o", "quest", "kivi"])
def test_native_attention_rejects_method_replacement(method):
    with pytest.raises(ValueError, match="requires its native"):
        resolve_model_sparse_method("deepseek_v4", method)


def test_native_index_binds_to_cache_device_instead_of_current_device(monkeypatch):
    # A cache owner can be constructed on an explicit device while another
    # device is current; selection scratch must follow the cache's device.
    from sparsevllm.engine.sparse_methods.deepseek_v4 import DeepseekV4Runtime

    bound = []
    provider = SimpleNamespace()

    def prepare(spec, *, device_index):
        bound.append(device_index)
        return provider

    monkeypatch.setattr("sparsevllm.engine.sparse_methods.native_index.prepare_compressed_index", prepare)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    cache = SimpleNamespace(device=torch.device("cuda", 2),
                            parallel_context=SimpleNamespace(attn_tp_size=1))
    config = SimpleNamespace(
        sparse_method="deepseek_v4", obs_layer_ids=(), full_attention_layers=(),
        sink_keep_tokens=0, recent_keep_tokens=0, decode_keep_tokens=0,
        max_model_len=4096, max_num_batched_tokens=128,
        runtime_layout=SimpleNamespace(kv_head_dims=(512,)),
        hf_config=SimpleNamespace(index_n_heads=64, index_head_dim=128, index_topk=512, num_hidden_layers=1),
    )
    runtime = DeepseekV4Runtime(config, cache)
    assert runtime.selection.provider is provider
    assert bound == [cache.device.index]


def test_native_compressor_providers_follow_parameter_device(monkeypatch):
    # Explicit tensor placement can differ from the ambient CUDA device.
    # Provider mocks check ownership only; CUDA tests cover the arithmetic.
    from sparsevllm.models.deepseek_v4 import compression

    bound = []
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(compression, "ReplicatedLinear", torch.nn.Linear)

    def prepare(spec, *, device_index):
        bound.append(device_index)
        return SimpleNamespace()

    for name in ("prepare_float32_linear", "prepare_compression", "prepare_native_rotary", "prepare_rotated_mxfp4"):
        monkeypatch.setattr(compression, name, prepare)
    with torch.device("meta"):
        model = compression.DeepseekV4Compressor(
            SimpleNamespace(hidden_size=256, qk_rope_head_dim=64, rms_norm_eps=1e-6),
            ratio=4, head_dim=128, rotate=True, max_num_tokens=8, max_num_requests=2,
        )
    assert bound and all(index == model.wkv.weight.device.index for index in bound)


def test_native_expert_storage_follows_explicit_owner_device(monkeypatch):
    from sparsevllm.models.deepseek_v4 import moe

    bound = []
    storage = moe.Mxfp4ExpertWeights(*(torch.empty(0) for _ in range(4)))

    def resolve(spec, *, device_index):
        bound.append(device_index)
        return SimpleNamespace(allocate_weights=lambda: storage)

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(moe, "resolve_mxfp4_moe_provider", resolve)
    model = moe.DeepseekV4Experts(SimpleNamespace(), device_index=2)
    assert bound == [2]
    assert model.storage.gate_up is storage.gate_up


def test_native_router_providers_follow_parameter_device(monkeypatch):
    from sparsevllm.models.deepseek_v4 import moe

    bound = []

    def prepare(spec, *, device_index):
        bound.append(device_index)
        return SimpleNamespace()

    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(moe, "resolve_moe_router_provider", prepare)
    monkeypatch.setattr(moe, "prepare_float32_linear", prepare)
    config = SimpleNamespace(num_hash_layers=0, routed_scaling_factor=1.,
                             n_routed_experts=4, hidden_size=256,
                             num_experts_per_tok=2, norm_topk_prob=True)
    with torch.device("meta"):
        model = moe.DeepseekV4Router(config, 0, max_num_tokens=4, cuda_graph=True)
    assert bound and all(index == model.weight.device.index for index in bound)
