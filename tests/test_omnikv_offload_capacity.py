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
