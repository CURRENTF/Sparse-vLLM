from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch

from sparsevllm.config import RuntimeLayout
from sparsevllm.engine.cache_manager import (
    ExplicitKVPayload,
    ExplicitKVWrite,
    LayerBatchStates,
    MlaLatentPayload,
    MlaLatentWrite,
)
from sparsevllm.engine.cache_manager.standard import StandardCacheManager
from sparsevllm.engine.cache_manager.snapkv import SnapKVCacheManager
from sparsevllm.engine.cache_manager.storage import (
    CacheLayout,
    ExplicitKVStorage,
    MlaLatentStorage,
    create_attention_cache_storage,
)
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


@pytest.mark.parametrize("layout", ["explicit_kv", "mla_latent"])
def test_attention_storage_copy_slots_is_overlap_safe(layout):
    if layout == "explicit_kv":
        storage = ExplicitKVStorage(
            num_kv_heads=1,
            head_dim=4,
            dtype=torch.float32,
        )
    else:
        storage = MlaLatentStorage(
            kv_lora_rank=512,
            rope_dim=64,
            dtype=torch.bfloat16,
        )
    storage.allocate(num_layers=1, num_slots=4, device=torch.device("cpu"))
    payload = storage.layer_payload(0)
    tensors = (
        (payload.k_cache, payload.v_cache)
        if isinstance(payload, ExplicitKVPayload)
        else (payload.latent_cache, payload.rope_cache)
    )
    for tensor_idx, tensor in enumerate(tensors):
        for slot in range(4):
            tensor[slot].fill_(tensor_idx * 10 + slot)

    storage.copy_slots(
        0,
        torch.tensor([3, 1], dtype=torch.long),
        torch.tensor([1, 2], dtype=torch.long),
    )

    for tensor_idx, tensor in enumerate(tensors):
        assert torch.all(tensor[1] == tensor_idx * 10 + 3)
        assert torch.all(tensor[2] == tensor_idx * 10 + 1)


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


def test_mla_storage_can_revalidate_between_graph_warmup_and_capture():
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

    with patch(
        "sparsevllm.engine.cache_manager.storage.mla_latent.copy_latent_to_cache"
    ) as copy:
        storage.validate_slot_mapping(slots)
        storage.store(0, slots, write)
        storage.store(1, slots, write)
        storage.validate_slot_mapping(slots)
        storage.store(0, slots, write)
        storage.store(1, slots, write)

    assert [call.kwargs["validate_slots"] for call in copy.call_args_list] == [
        False,
        False,
        False,
        False,
    ]


def test_mla_storage_prevalidates_nonuniform_layer_mappings():
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=2, num_slots=2, device=torch.device("cpu"))
    layer_slots = (
        torch.tensor([0], dtype=torch.int32),
        torch.tensor([1], dtype=torch.int32),
    )
    write = MlaLatentWrite(
        latent=torch.empty(1, 1, 512, dtype=torch.bfloat16),
        rope=torch.empty(1, 1, 64, dtype=torch.bfloat16),
    )
    storage.validate_slot_mappings(layer_slots)

    with patch(
        "sparsevllm.engine.cache_manager.storage.mla_latent.copy_latent_to_cache"
    ) as copy:
        storage.store(0, layer_slots[0], write)
        storage.store(1, layer_slots[1], write)
        storage.store(0, layer_slots[0], write)

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


def test_snapkv_manager_delegates_latent_store_and_compute_view():
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=2, num_slots=4, device=torch.device("cpu"))
    manager = object.__new__(SnapKVCacheManager)
    manager.attention_cache_storage = storage
    manager.runtime_layout = RuntimeLayout.dense(2)
    manager.layer_batch_states = [LayerBatchStates(), LayerBatchStates()]
    manager._pyramidkv_prefill_staging_active = False
    manager.layer_batch_states[1].slot_mapping = torch.tensor(
        [1], dtype=torch.int32
    )
    write = MlaLatentWrite(
        latent=torch.empty(1, 1, 512, dtype=torch.bfloat16),
        rope=torch.empty(1, 1, 64, dtype=torch.bfloat16),
    )

    with patch.object(storage, "store") as store:
        returned_slots = manager.store_attention_payload(1, write)
    store.assert_called_once_with(
        1,
        manager.layer_batch_states[1].slot_mapping,
        write,
    )
    assert returned_slots is manager.layer_batch_states[1].slot_mapping

    active_slots = torch.tensor([[0, 1]], dtype=torch.int32)
    req_indices = torch.tensor([0], dtype=torch.int32)
    context_lens = torch.tensor([2], dtype=torch.int32)
    payload, actual_slots, actual_rows, actual_lens = (
        manager.get_layer_compute_payload(
            1,
            active_slots,
            req_indices,
            context_lens,
        )
    )
    assert isinstance(payload, MlaLatentPayload)
    assert actual_slots is active_slots
    assert actual_rows is req_indices
    assert actual_lens is context_lens


def test_graph_capture_prevalidates_nonuniform_latent_layer_mappings():
    storage = MlaLatentStorage(
        kv_lora_rank=512,
        rope_dim=64,
        dtype=torch.bfloat16,
    )
    storage.allocate(num_layers=2, num_slots=4, device=torch.device("cpu"))
    manager = object.__new__(SnapKVCacheManager)
    manager.attention_cache_storage = storage
    manager.runtime_layout = RuntimeLayout.dense(2)
    manager.layer_batch_states = [
        LayerBatchStates(slot_mapping=torch.tensor([0], dtype=torch.int32)),
        LayerBatchStates(slot_mapping=torch.tensor([1], dtype=torch.int32)),
    ]
    write = MlaLatentWrite(
        latent=torch.empty(1, 1, 512, dtype=torch.bfloat16),
        rope=torch.empty(1, 1, 64, dtype=torch.bfloat16),
    )

    manager.validate_decode_cuda_graph_slot_mappings()
    with patch(
        "sparsevllm.engine.cache_manager.storage.mla_latent.copy_latent_to_cache"
    ) as copy:
        manager.store_attention_payload(0, write)
        manager.store_attention_payload(1, write)

    assert [call.kwargs["validate_slots"] for call in copy.call_args_list] == [
        False,
        False,
    ]


def test_snapkv_explicit_compute_view_preserves_legacy_payload():
    storage = ExplicitKVStorage(
        num_kv_heads=2,
        head_dim=8,
        dtype=torch.float16,
    )
    storage.allocate(num_layers=2, num_slots=4, device=torch.device("cpu"))
    manager = object.__new__(SnapKVCacheManager)
    manager.attention_cache_storage = storage
    manager.runtime_layout = RuntimeLayout.dense(2)
    manager.kv_cache = storage.cache
    manager.layer_batch_states = [LayerBatchStates(), LayerBatchStates()]
    manager._pyramidkv_prefill_staging_active = False
    active_slots = torch.tensor([[0, 1]], dtype=torch.int32)
    req_indices = torch.tensor([0], dtype=torch.int32)
    context_lens = torch.tensor([2], dtype=torch.int32)

    payload, actual_slots, actual_rows, actual_lens = (
        manager.get_layer_compute_payload(
            1,
            active_slots,
            req_indices,
            context_lens,
        )
    )

    assert isinstance(payload, ExplicitKVPayload)
    assert payload.k_cache.data_ptr() == storage.cache[0, 1].data_ptr()
    assert payload.v_cache.data_ptr() == storage.cache[1, 1].data_ptr()
    assert actual_slots is active_slots
    assert actual_rows is req_indices
    assert actual_lens is context_lens


@pytest.mark.parametrize(
    ("attention_tp_size", "local_heads"),
    [(1, 20), (4, 5)],
)
def test_standard_manager_accounts_storage_tensors_explicitly(
    attention_tp_size,
    local_heads,
):
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
    manager.hf_config = SimpleNamespace(
        torch_dtype=torch.bfloat16,
        num_attention_heads=20,
        qk_nope_head_dim=192,
        qk_rope_head_dim=64,
        v_head_dim=256,
    )
    manager.parallel_context = SimpleNamespace(
        attention_tp_size=attention_tp_size,
    )
    manager.num_layers = 2
    manager.num_kv_layers = 2
    manager.num_kv_heads = 4
    manager.head_dim = 64
    manager.row_seq_lens = np.array([2], dtype=np.int32)

    accounting = manager.memory_accounting()

    assert accounting["kv_or_latent_tensor_bytes"] == 2 * 3 * 576 * 2
    assert accounting["logical_live_kv_bytes"] == 2 * 2 * 576 * 2
    assert accounting["dense_baseline_bytes"] == (
        3 * 2 * local_heads * (256 + 256) * 2
    )
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
