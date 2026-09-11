"""Pool admission must reject exhausted GPU/host budgets before allocating KV."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparsevllm.engine.cache_manager.omnikv import OmniKVCacheManager
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
        patch("sparsevllm.engine.cache_manager.omnikv.OmniKVStorage") as allocate,
    ):
        with pytest.raises(MemoryError, match="pools cannot fit"):
            manager.allocate_kv_cache()
        allocate.assert_not_called()
    assert manager.config.num_kvcache_slots == 0
    assert manager.attention_cache_storage is original


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
            patch("sparsevllm.engine.cache_manager.omnikv.OmniKVStorage") as allocate,
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
