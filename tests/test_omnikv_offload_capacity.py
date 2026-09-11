"""Pool admission must reject exhausted GPU/host budgets before allocating KV."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparsevllm.engine.cache_manager.omnikv import OmniKVCacheManager
from sparsevllm.engine.cache_manager.omnikv_capacity import (
    fit_omnikv_host_slots,
    omnikv_host_pool_bytes,
)
from sparsevllm.engine.cache_manager.storage import ExplicitKVStorage


@pytest.mark.parametrize("gpu_bytes,host_kib", [(1, 100000), (1000000, 0)])
def test_insufficient_pool_budget_does_not_publish_capacity(gpu_bytes, host_kib):
    # Existing slot-allocator tests cannot catch reserving a GPU pool while
    # silently exceeding the newly coupled host backing budget.
    manager = object.__new__(OmniKVCacheManager)
    manager.offload_enabled = True
    manager.device = torch.device("cuda")
    original = ExplicitKVStorage(num_kv_heads=2, head_dim=128, dtype=torch.bfloat16)
    manager.attention_cache_storage = original
    manager.num_kv_layers = 4
    manager.kv_layer_index = int
    manager.max_model_len = 64
    manager.max_buffer_rows = 2
    manager.world_size = 1
    manager.config = SimpleNamespace(
        max_model_len=64,
        full_attention_layers=[0],
        sink_keep_tokens=0,
        decode_keep_tokens=8,
        recent_keep_tokens=2,
        enable_prefix_cache_offload=False,
        prefix_cache_block_size=16,
        enable_prefix_caching=True,
        num_kvcache_slots=0,
    )
    manager._get_available_slots_info = lambda: (gpu_bytes, 1024)
    with (
        patch("pathlib.Path.read_text", return_value=f"MemAvailable: {host_kib} kB\n"),
        patch(
            "sparsevllm.engine.cache_manager.omnikv.OmniKVStorage.__init__"
        ) as allocate,
    ):
        with pytest.raises(MemoryError, match="pools cannot fit"):
            manager.allocate_kv_cache()
        allocate.assert_not_called()
    assert manager.config.num_kvcache_slots == 0
    assert manager.attention_cache_storage is original


@pytest.mark.parametrize("prefix_caching", [False, True])
def test_history_preallocation_is_bounded_by_request_capacity(prefix_caching):
    # Prefix retention must not multiply the live-request history allocation.
    manager = object.__new__(OmniKVCacheManager)
    manager.offload_enabled = True
    manager.device = torch.device("cuda")
    original = ExplicitKVStorage(num_kv_heads=2, head_dim=128, dtype=torch.bfloat16)
    manager.attention_cache_storage = original
    manager.num_kv_layers = 4
    manager.kv_layer_index = int
    manager.max_model_len = 64
    manager.max_buffer_rows = 2
    manager.world_size = 1
    manager.config = SimpleNamespace(
        max_model_len=64,
        full_attention_layers=[0],
        sink_keep_tokens=0,
        decode_keep_tokens=8,
        recent_keep_tokens=2,
        enable_prefix_cache_offload=False,
        prefix_cache_block_size=16,
        enable_prefix_caching=prefix_caching,
        num_kvcache_slots=0,
        prefix_cache_max_blocks=None,
    )
    manager._get_available_slots_info = lambda: (1024**3, 1024)

    class AllocationReached(Exception):
        pass

    with (
        patch("pathlib.Path.read_text", return_value="MemAvailable: 1048576 kB\n"),
        patch(
            "sparsevllm.engine.cache_manager.omnikv.OmniKVStorage.__init__",
            side_effect=AllocationReached,
        ) as allocate,
        pytest.raises(AllocationReached),
    ):
        manager.allocate_kv_cache()
    assert allocate.call_args.kwargs["num_slots"] == (
        manager.max_model_len * manager.max_buffer_rows
    )


@pytest.mark.parametrize("part_bytes", [(2048, 2048), (1024, 128)])
@pytest.mark.parametrize("prefix_slots", [0, 23])
def test_host_capacity_covers_allocator_rounding_and_split_prefix(
    part_bytes, prefix_slots
):
    # Logical bytes previously admitted a slot just beyond a pinned allocator
    # size boundary, nearly doubling real RAM use. GPU-only budgets miss this.
    limit, sparse, full = 1025, 3, 2
    budget = sum(part_bytes) * (sparse * limit + (sparse + full) * prefix_slots)
    slots = fit_omnikv_host_slots(limit, budget, part_bytes, sparse, full, prefix_slots)
    assert 0 < slots < limit
    assert (
        omnikv_host_pool_bytes(slots, part_bytes, sparse, full, prefix_slots) <= budget
    )
    assert (
        omnikv_host_pool_bytes(slots + 1, part_bytes, sparse, full, prefix_slots)
        > budget
    )


@pytest.mark.skipif(
    not torch.cuda.is_available()
    or not hasattr(torch.cuda.memory, "host_memory_stats"),
    reason="CUDA host allocator statistics required",
)
@pytest.mark.parametrize("part_bytes", [(2048, 2048), (1024, 128)])
def test_host_pool_budget_matches_real_pinned_allocations(part_bytes):
    # Compare against the allocator, not another copy of the rounding formula.
    slots, prefix, sparse, full = 1025, 23, 3, 2
    torch.cuda.synchronize()
    before = torch.cuda.memory.host_memory_stats()["active_bytes.current"]
    buffers = [
        torch.empty(count * width, dtype=torch.uint8, pin_memory=True)
        for count, layers in ((slots + prefix, sparse), (prefix, full))
        for _ in range(layers)
        for width in part_bytes
    ]
    actual = torch.cuda.memory.host_memory_stats()["active_bytes.current"] - before
    assert actual == omnikv_host_pool_bytes(slots, part_bytes, sparse, full, prefix)
    assert actual > sum(t.numel() for t in buffers)


def _reject_exhausted_peer(rank, init_file):
    import torch.distributed as dist

    torch.cuda.set_device(rank)
    dist.init_process_group(
        "nccl", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        manager = object.__new__(OmniKVCacheManager)
        manager.offload_enabled = True
        manager.device = torch.device("cuda", rank)
        manager.attention_cache_storage = ExplicitKVStorage(
            num_kv_heads=2, head_dim=128, dtype=torch.bfloat16
        )
        manager.num_kv_layers = 4
        manager.kv_layer_index = int
        manager.max_model_len = 64
        manager.max_buffer_rows = 2
        manager.world_size = 2
        manager.parallel_context = SimpleNamespace(world_all_reduce=dist.all_reduce)
        manager.config = SimpleNamespace(
            max_model_len=64,
            full_attention_layers=[0],
            sink_keep_tokens=0,
            decode_keep_tokens=8,
            recent_keep_tokens=2,
            enable_prefix_cache_offload=False,
            prefix_cache_block_size=16,
            enable_prefix_caching=True,
            num_kvcache_slots=0,
        )
        manager._get_available_slots_info = lambda: (1 if rank == 0 else 1000000, 1024)
        with (
            patch("pathlib.Path.read_text", return_value="MemAvailable: 100000 kB\n"),
            patch(
                "sparsevllm.engine.cache_manager.omnikv.OmniKVStorage.__init__"
            ) as allocate,
        ):
            with pytest.raises(MemoryError, match="pools cannot fit"):
                manager.allocate_kv_cache()
            allocate.assert_not_called()
        assert manager.config.num_kvcache_slots == 0
        # Both ranks reached the same failure boundary before any KV mutation.
        dist.barrier()
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="two idle CUDA devices required"
)
def test_pool_admission_rejects_an_exhausted_peer(tmp_path):
    torch.multiprocessing.spawn(
        _reject_exhausted_peer,
        args=(str(tmp_path / "init"),),
        nprocs=2,
        join=True,
    )
