"""Synthetic native state for allocation profiling, never for model inference."""

import torch

from sparsevllm.engine.cache_manager.methods.deepseek_v4_manager import DeepseekV4CacheManager
from sparsevllm.engine.cache_manager.storage.native_capacity import NativeCacheGeometry
from sparsevllm.engine.cache_manager.storage.native_pool import NativeLiveRow


class NativePrefillHistoryCacheManager(DeepseekV4CacheManager):
    def __init__(self, config, parallel_context):
        budget = NativeCacheGeometry.from_config(config).budget_for_tokens(
            config.num_kvcache_slots, live_rows=config.max_num_seqs_in_gpu,
        )
        super().__init__(config, parallel_context, allocation_budget_bytes=budget)
        for layer in self.pool.layers:
            for tensor in layer.accounting_tensors():
                tensor.zero_()

    def seed_history(self, seq):
        # Profiling needs valid physical indices and finite window/carry/index
        # history at maximum context; these values are not quality evidence.
        length = int(seq.num_prefilled_tokens)
        pool = self.pool
        if seq.seq_id in pool.live or not 0 < length < pool.max_model_len:
            raise ValueError("Synthetic history requires a new request within context capacity.")
        if not pool.free_live_rows or any(length // ratio > slots.free_count for ratio, slots in pool.slots.items()):
            raise RuntimeError("Synthetic native history exceeds physical capacity.")
        row = pool.free_live_rows.popleft()
        for ratio, allocator in pool.slots.items():
            slots = allocator.allocate(length // ratio)
            pool.host_tables[ratio][row, :len(slots)] = slots
            pool.tables[ratio][row, :len(slots)].copy_(torch.from_numpy(slots))
        pool.live[seq.seq_id] = NativeLiveRow(row, length)
