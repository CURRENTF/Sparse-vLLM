from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sparsevllm.config import Config, RuntimeLayout
from sparsevllm.engine.cache_manager import (
    ExplicitKVPayload,
    ExplicitKVWrite,
    LayerBatchStates,
    MlaLatentPayload,
    MlaLatentWrite,
)
from sparsevllm.engine.cache_manager.standard import StandardCacheManager
from sparsevllm.engine.cache_manager.storage import (
    CacheLayout,
    ExplicitKVStorage,
    MlaLatentStorage,
    create_attention_cache_storage,
)


def _glm_hf_config(**overrides):
    values = {
        "model_type": "glm4_moe_lite",
        "architectures": ["Glm4MoeLiteForCausalLM"],
        "torch_dtype": torch.bfloat16,
        "max_position_embeddings": 4096,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "vocab_size": 128,
        "kv_lora_rank": 512,
        "qk_rope_head_dim": 64,
        "quantization_config": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _glm_config(*, hf_overrides=None, **overrides) -> Config:
    kwargs = {
        "model": str(Path(__file__).resolve().parents[1]),
        "max_model_len": 128,
        "max_num_batched_tokens": 64,
        "chunk_prefill_size": 64,
        "enforce_eager": True,
    }
    kwargs.update(overrides)
    with patch(
        "sparsevllm.configs.runtime.AutoConfig.from_pretrained",
        return_value=_glm_hf_config(**(hf_overrides or {})),
    ):
        return Config(**kwargs)


def test_glm_config_selects_mla_latent_layout():
    config = _glm_config()

    assert config.attention_cache_layout == CacheLayout.MLA_LATENT.value
    assert config.mla_prefill_workspace_bytes == 2 * 1024**3


def test_mla_prefill_workspace_budget_must_be_positive():
    with pytest.raises(ValueError, match="mla_prefill_workspace_bytes"):
        _glm_config(mla_prefill_workspace_bytes=0)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"vllm_sparse_method": "snapkv"}, "only vanilla attention"),
        ({"enable_prefix_caching": True}, "does not support prefix caching"),
        ({"decode_cuda_graph": True}, "does not support decode CUDA Graph"),
        ({"enforce_eager": False}, "requires enforce_eager=True"),
    ],
)
def test_glm_config_rejects_unsupported_storage_combinations(override, message):
    with pytest.raises(NotImplementedError, match=message):
        _glm_config(**override)


@pytest.mark.parametrize(
    ("hf_override", "message"),
    [
        ({"torch_dtype": torch.float16}, "requires torch.bfloat16"),
        ({"kv_lora_rank": 256}, "requires kv_lora_rank=512"),
        ({"qk_rope_head_dim": 32}, "qk_rope_head_dim=64"),
    ],
)
def test_glm_config_rejects_unsupported_mla_storage_contract(hf_override, message):
    with pytest.raises(NotImplementedError, match=message):
        _glm_config(hf_overrides=hf_override)


def test_explicit_storage_preserves_legacy_tensor_layout_and_size():
    storage = ExplicitKVStorage(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float16,
    )
    storage.allocate(num_layers=3, num_slots=5, device=torch.device("cpu"))

    assert storage.layout is CacheLayout.EXPLICIT_KV
    assert storage.cache.shape == (2, 3, 5, 2, 8)
    assert storage.bytes_per_slot_per_layer() == 2 * 2 * 8 * 2
    assert storage.cache.untyped_storage().nbytes() == 3 * 5 * 2 * 2 * 8 * 2
    payload = storage.layer_payload(1)
    assert isinstance(payload, ExplicitKVPayload)
    assert payload.k_cache.data_ptr() == storage.cache[0, 1].data_ptr()
    assert payload.v_cache.data_ptr() == storage.cache[1, 1].data_ptr()
    accounting_tensors = storage.accounting_tensors()
    assert len(accounting_tensors) == 1
    assert accounting_tensors[0] is storage.cache


def test_mla_storage_uses_576_bf16_values_per_slot_per_layer():
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=2, num_slots=3, device=torch.device("cpu"))

    assert storage.layout is CacheLayout.MLA_LATENT
    assert storage.latent_cache is not None
    assert storage.rope_cache is not None
    assert storage.latent_cache.shape == (2, 3, 1, 512)
    assert storage.rope_cache.shape == (2, 3, 1, 64)
    assert storage.bytes_per_slot_per_layer() == 576 * 2
    assert sum(t.untyped_storage().nbytes() for t in storage.accounting_tensors()) == (
        2 * 3 * 576 * 2
    )
    payload = storage.layer_payload(1)
    assert isinstance(payload, MlaLatentPayload)
    assert payload.latent_cache.data_ptr() == storage.latent_cache[1].data_ptr()
    assert payload.rope_cache.data_ptr() == storage.rope_cache[1].data_ptr()


def test_storage_factory_uses_configured_layout():
    explicit_config = SimpleNamespace(
        attention_cache_layout="explicit_kv",
        hf_config=SimpleNamespace(torch_dtype=torch.float16),
    )
    mla_config = SimpleNamespace(
        attention_cache_layout="mla_latent",
        hf_config=SimpleNamespace(
            torch_dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        ),
    )

    assert isinstance(
        create_attention_cache_storage(
            explicit_config,
            num_kv_heads=2,
            head_dim=8,
        ),
        ExplicitKVStorage,
    )
    assert isinstance(
        create_attention_cache_storage(
            mla_config,
            num_kv_heads=4,
            head_dim=64,
        ),
        MlaLatentStorage,
    )

    inferred_mla_config = SimpleNamespace(
        hf_config=SimpleNamespace(
            model_type="glm4_moe_lite",
            torch_dtype=torch.bfloat16,
            kv_lora_rank=512,
            qk_rope_head_dim=64,
        )
    )
    assert isinstance(
        create_attention_cache_storage(
            inferred_mla_config,
            num_kv_heads=4,
            head_dim=64,
        ),
        MlaLatentStorage,
    )


@pytest.mark.parametrize(
    ("storage", "num_layers", "num_slots", "expected_shape"),
    [
        (
            ExplicitKVStorage(
                num_kv_heads=2,
                head_dim=8,
                dtype=torch.float16,
            ),
            3,
            5,
            (2, 3, 5, 2, 8),
        ),
        (
            MlaLatentStorage(
                kv_lora_rank=512,
                rope_dim=64,
                dtype=torch.bfloat16,
            ),
            2,
            7,
            ((2, 7, 1, 512), (2, 7, 1, 64)),
        ),
    ],
)
def test_standard_manager_derives_capacity_and_allocates_through_storage(
    storage,
    num_layers,
    num_slots,
    expected_shape,
):
    manager = object.__new__(StandardCacheManager)
    manager.attention_cache_storage = storage
    manager.num_kv_layers = num_layers
    manager.device = torch.device("cpu")
    manager.config = SimpleNamespace(num_kvcache_slots=-1)
    slot_bytes = storage.bytes_per_slot_per_layer()
    manager._get_available_slots_info = lambda: (
        num_layers * num_slots * slot_bytes,
        slot_bytes,
    )

    manager.allocate_kv_cache()

    assert manager.config.num_kvcache_slots == num_slots
    if isinstance(storage, ExplicitKVStorage):
        assert storage.cache.shape == expected_shape
        assert manager.kv_cache is storage.cache
    else:
        assert storage.latent_cache is not None
        assert storage.rope_cache is not None
        assert (storage.latent_cache.shape, storage.rope_cache.shape) == expected_shape
        assert manager.kv_cache is None


def test_storage_store_payload_types_are_not_interchangeable():
    explicit = ExplicitKVStorage(
        num_kv_heads=1,
        head_dim=4,
        dtype=torch.float16,
    )
    explicit.allocate(num_layers=1, num_slots=2, device=torch.device("cpu"))
    mla = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    mla.allocate(num_layers=1, num_slots=2, device=torch.device("cpu"))
    slots = torch.tensor([0], dtype=torch.int32)

    with pytest.raises(TypeError, match="ExplicitKVWrite"):
        explicit.store(
            0,
            slots,
            MlaLatentWrite(
                latent=torch.empty(1, 1, 512, dtype=torch.bfloat16),
                rope=torch.empty(1, 1, 64, dtype=torch.bfloat16),
            ),
        )
    with pytest.raises(TypeError, match="MlaLatentWrite"):
        mla.store(
            0,
            slots,
            ExplicitKVWrite(
                key=torch.empty(1, 1, 4, dtype=torch.float16),
                value=torch.empty(1, 1, 4, dtype=torch.float16),
            ),
        )


def test_mla_storage_reuses_one_manager_validation_across_layers():
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=2, num_slots=2, device=torch.device("cpu"))
    slots = torch.tensor([0], dtype=torch.int32)
    write = MlaLatentWrite(
        latent=torch.empty(1, 1, 512, dtype=torch.bfloat16),
        rope=torch.empty(1, 1, 64, dtype=torch.bfloat16),
    )
    storage.validate_slot_mapping(slots)

    with patch(
        "sparsevllm.engine.cache_manager.storage.mla_latent.copy_latent_to_cache"
    ) as copy:
        storage.store(0, slots, write)
        storage.store(1, slots, write)
        storage.store(0, slots, write)

    assert [call.kwargs["validate_slots"] for call in copy.call_args_list] == [
        False,
        False,
        True,
    ]


def test_standard_manager_delegates_payload_store_and_compute_view():
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=2, num_slots=4, device=torch.device("cpu"))
    manager = object.__new__(StandardCacheManager)
    manager.attention_cache_storage = storage
    manager.runtime_layout = RuntimeLayout.dense(2)
    manager.layer_batch_state = LayerBatchStates(
        slot_mapping=torch.tensor([1], dtype=torch.int32)
    )
    write = MlaLatentWrite(
        latent=torch.empty(1, 1, 512, dtype=torch.bfloat16),
        rope=torch.empty(1, 1, 64, dtype=torch.bfloat16),
    )

    with patch.object(storage, "store") as store:
        returned_slots = manager.store_attention_payload(1, write)
    store.assert_called_once_with(1, manager.layer_batch_state.slot_mapping, write)
    assert returned_slots is manager.layer_batch_state.slot_mapping

    active_slots = torch.tensor([[0, 1]], dtype=torch.int32)
    req_indices = torch.tensor([0], dtype=torch.int32)
    context_lens = torch.tensor([2], dtype=torch.int32)
    payload, actual_slots, actual_rows, actual_lens = manager.get_layer_compute_payload(
        1,
        active_slots,
        req_indices,
        context_lens,
    )
    assert isinstance(payload, MlaLatentPayload)
    assert actual_slots is active_slots
    assert actual_rows is req_indices
    assert actual_lens is context_lens


def test_standard_manager_accounts_storage_tensors_explicitly():
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=2, num_slots=3, device=torch.device("cpu"))
    manager = object.__new__(StandardCacheManager)
    manager.attention_cache_storage = storage
    manager.kv_cache = None
    manager.config = SimpleNamespace(
        num_kvcache_slots=3,
        max_num_seqs_in_gpu=1,
        memory_expected_savings=None,
    )
    manager.hf_config = SimpleNamespace(torch_dtype=torch.bfloat16)
    manager.num_layers = 2
    manager.num_kv_layers = 2
    manager.num_kv_heads = 4
    manager.head_dim = 64
    manager.row_seq_lens = np.array([2], dtype=np.int32)

    accounting = manager.memory_accounting()

    assert accounting["kv_or_latent_tensor_bytes"] == 2 * 3 * 576 * 2
    assert accounting["logical_live_kv_bytes"] == 2 * 2 * 576 * 2
    assert accounting["tensor_count"] == 2
    assert {item["path"] for item in accounting["tensors"]} == {
        "attention_cache_storage.mla_latent.0_cache",
        "attention_cache_storage.mla_latent.1_cache",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mla_storage_store_skips_padding_and_overwrites_reused_slot():
    device = torch.device("cuda")
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=1, num_slots=4, device=device)
    assert storage.latent_cache is not None
    assert storage.rope_cache is not None
    storage.latent_cache.fill_(-7)
    storage.rope_cache.fill_(-7)

    latent = torch.stack(
        [torch.full((1, 512), value, dtype=torch.bfloat16, device=device) for value in (1, 2, 3)]
    )
    rope = torch.stack(
        [torch.full((1, 64), value, dtype=torch.bfloat16, device=device) for value in (4, 5, 6)]
    )
    slot_mapping = torch.tensor([1, -1, 3], dtype=torch.int32, device=device)
    storage.validate_slot_mapping(slot_mapping)
    storage.store(
        0,
        slot_mapping,
        MlaLatentWrite(latent=latent, rope=rope),
    )

    assert torch.equal(storage.latent_cache[0, 0], torch.full_like(storage.latent_cache[0, 0], -7))
    assert torch.equal(storage.latent_cache[0, 1], latent[0])
    assert torch.equal(storage.latent_cache[0, 2], torch.full_like(storage.latent_cache[0, 2], -7))
    assert torch.equal(storage.latent_cache[0, 3], latent[2])
    assert torch.equal(storage.rope_cache[0, 1], rope[0])
    assert torch.equal(storage.rope_cache[0, 3], rope[2])

    replacement = MlaLatentWrite(
        latent=torch.full((1, 1, 512), 9, dtype=torch.bfloat16, device=device),
        rope=torch.full((1, 1, 64), 10, dtype=torch.bfloat16, device=device),
    )
    storage.store(
        0,
        torch.tensor([1], dtype=torch.int32, device=device),
        replacement,
    )
    assert torch.equal(storage.latent_cache[0, 1], replacement.latent[0])
    assert torch.equal(storage.rope_cache[0, 1], replacement.rope[0])
