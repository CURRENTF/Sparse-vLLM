"""Native compressed pools, scheduler budgets and committed radix snapshots."""

from dataclasses import dataclass

import torch

from sparsevllm.engine.cache_manager.base import CacheManager, LayerBatchStates
from sparsevllm.engine.cache_manager.native_attention import NativeAttentionExecution
from sparsevllm.engine.cache_manager.storage.native_capacity import NativeCacheGeometry
from sparsevllm.engine.cache_manager.storage.native_pool import NativeCachePool, NativeDecodeState
from sparsevllm.engine.decode_graph_contract import CacheDecodeGraphState, DecodeGraphInputs
from sparsevllm.engine.prefix_cache import PrefixCacheRoutingSnapshot, build_prefix_cache_fingerprint

from .deepseek_v4_prefix import NativePrefixCache


@dataclass
class NativeCacheDecodeGraphState(CacheDecodeGraphState):
    native: NativeDecodeState
    executions: tuple[NativeAttentionExecution, ...]


class DeepseekV4CacheManager(CacheManager):
    def __init__(self, config, parallel_context, *, allocation_budget_bytes=None):
        super().__init__(config, parallel_context, allocation_budget_bytes=allocation_budget_bytes)
        self.enable_prefix_caching = bool(config.enable_prefix_caching)
        if (getattr(config, "enable_prefix_cache_offload", False)
                or getattr(config, "enable_chain_cache", False)):
            raise ValueError("Native cache does not support prefix offload or chain cache.")
        self.prefix_cache_block_size = int(config.prefix_cache_block_size)
        self._step = self._prefix_plan = None
        self._batches = ()
        self._executions = ()
        self._context_lens = None
        self._eager_decode = {}
        self.allocate_kv_cache()
        self.prefix = NativePrefixCache(self.pool, block_size=self.prefix_cache_block_size,
                                        fingerprint=build_prefix_cache_fingerprint(config, self.prefix_cache_block_size)) \
            if self.enable_prefix_caching else None
        self.prefix_cache = self.prefix.index if self.prefix is not None else None

    def allocate_kv_cache(self):
        geometry = NativeCacheGeometry.from_config(self.config)
        budget = self.allocation_budget_bytes
        if budget is None:
            free, total = self.platform.get_available_memory(self.device.index or 0)
            budget = int(free - total * (1 - self.config.gpu_memory_utilization))
        snapshot_rows = 0
        if self.enable_prefix_caching:
            # Jointly budget one state row and its prefix blocks' compressed KV.
            # The explicit max-blocks setting can leave more capacity for live KV.
            upper = max(0, (budget - geometry.temporary_bytes) // geometry.row_bytes - self.max_buffer_rows)
            requested = self.config.prefix_cache_max_blocks
            if getattr(self.config, "startup_cache_phase", None) == "profiling":
                requested = 1
            if requested is not None:
                upper = min(upper, int(requested))
            while snapshot_rows < upper:
                candidate = (snapshot_rows + upper + 1) // 2
                try:
                    plan = geometry.capacity_for_budget(budget, live_rows=self.max_buffer_rows, snapshot_rows=candidate)
                    fits = all(capacity >= (candidate * self.prefix_cache_block_size + ratio - 1) // ratio + self.max_buffer_rows
                               for ratio, capacity in plan.compressed_capacities.items())
                except MemoryError:
                    fits = False
                if fits:
                    snapshot_rows = candidate
                else:
                    upper = candidate - 1
            if snapshot_rows == 0:
                raise MemoryError("Native byte budget cannot hold a prefix snapshot and live compressed storage.")
        plan = geometry.capacity_for_budget(budget, live_rows=self.max_buffer_rows, snapshot_rows=snapshot_rows)
        self.pool = NativeCachePool(ratios=geometry.ratios, live_rows=self.max_buffer_rows, snapshot_rows=snapshot_rows,
                                    max_model_len=self.max_model_len, prefill_capacity=geometry.prefill_capacity,
                                    compressed_capacities=plan.compressed_capacities, device=self.device,
                                    head_dim=geometry.head_dim, index_dim=geometry.index_dim, window_size=geometry.window_size)
        self.config.num_kvcache_slots = min((capacity * ratio for ratio, capacity in plan.compressed_capacities.items()),
                                           default=self.max_buffer_rows * self.max_model_len)

    @property
    def num_free_slots(self):
        return min((allocator.free_count * ratio for ratio, allocator in self.pool.slots.items()),
                   default=len(self.pool.free_live_rows) * self.max_model_len)

    def prefill_step_free_slots(self):
        return self.pool.prefill_capacity

    def decode_step_free_slots(self):
        return self.config.max_decoding_seqs

    def reserved_prefill_slots(self, waiting_seqs, engine_prefill_chunk_size):
        return 0

    def prompt_admission_free_slots(self):
        return len(self.pool.free_live_rows)

    def prompt_logical_reservation_cost(self, seq):
        return int(seq.seq_id not in self.pool.live)

    def step_resource_budgets(self, *, is_prefill):
        if not is_prefill and self.prefix is not None:
            costs = {ratio: sum((row.length + 1) // ratio - row.length // ratio for row in self.pool.live.values())
                     for ratio in self.pool.slots}
            self.prefix.evict_for_capacity(costs)
        return {f"compressed_{ratio}": allocator.free_count for ratio, allocator in self.pool.slots.items()}

    def step_resource_costs(self, seq, scheduled_tokens, *, is_prefill):
        start = max(seq.num_prefilled_tokens, seq.prefix_cache_hit_len) if is_prefill else seq.decode_input_position
        return {f"compressed_{ratio}": need for ratio, need in self.pool.append_costs(start, start + scheduled_tokens).items()}

    def prompt_admission_costs(self, seq):
        start = max(seq.num_prefilled_tokens, seq.prefix_cache_hit_len)
        return {"rows": int(seq.seq_id not in self.pool.live),
                **{f"compressed_{ratio}": need for ratio, need in self.pool.append_costs(start, seq.num_prompt_tokens).items()}}

    def prompt_admission_budgets(self, waiting_seqs, engine_prefill_chunk_size):
        reserved = dict.fromkeys(self.pool.slots, 0)
        requested = dict.fromkeys(self.pool.slots, 0)
        for seq in waiting_seqs:
            if self.prefix is not None and seq.num_prefilled_tokens == 0 and seq.num_completion_tokens == 0:
                self.prefix.refresh_hit(seq)
            start = max(seq.num_prefilled_tokens, seq.prefix_cache_hit_len)
            costs = self.pool.append_costs(start, seq.num_prompt_tokens)
            for ratio, need in costs.items():
                if seq.seq_id in self.pool.live:
                    reserved[ratio] += need
                else:
                    requested[ratio] = max(requested[ratio], need)
        if self.prefix is not None:
            self.prefix.evict_for_capacity({ratio: reserved[ratio] + requested[ratio] for ratio in reserved})
        return {"rows": len(self.pool.free_live_rows),
                **{f"compressed_{ratio}": max(0, allocator.free_count - reserved[ratio])
                   for ratio, allocator in self.pool.slots.items()}}

    def refresh_prefix_cache_hit(self, seq):
        if self.prefix is None:
            seq.clear_prefix_cache_hit()
        else:
            self.prefix.refresh_hit(seq)

    def clear_prefix_cache_hit(self, seq):
        if self.prefix is None:
            seq.clear_prefix_cache_hit()
        else:
            self.prefix.clear_hit(seq)

    def _require_prefix_cache(self):
        if self.prefix_cache is None:
            raise RuntimeError("prefix cache is not enabled for this cache manager.")
        return self.prefix_cache

    def prefix_cache_match(self, token_ids):
        snapshot = (self.prefix_cache.routing_snapshot("deepseek_v4") if self.prefix_cache is not None
                    else PrefixCacheRoutingSnapshot(supported=True, enabled=False, method="deepseek_v4"))
        return snapshot.match(token_ids)

    def prefix_cache_inspect(self, token_ids, *, include_subtree=False):
        return self._require_prefix_cache().inspect_prefix(
            [int(token) for token in token_ids], include_subtree=include_subtree,
        )

    def prefix_cache_delete_subtree(self, token_ids):
        index = self._require_prefix_cache()
        normalized = [int(token) for token in token_ids]
        self.synchronize_prefix_cache_delete_plan(index.preview_delete_subtree(normalized).to_dict())
        result = index.safe_delete_subtree(normalized)
        for block in result.deleted_blocks:
            self.pool.release_snapshot(block.payload)
        return result.to_dict()

    def prefix_cache_set_eviction_priority(self, token_ids, *, priority):
        return self._require_prefix_cache().set_subtree_eviction_priority(
            [int(token) for token in token_ids], int(priority),
        )

    def reset_prefill_execution_state(self, seq_id):
        super().reset_prefill_execution_state(seq_id)
        if self.prefix is not None:
            self.prefix.release_sequence(seq_id)

    def complete_prefill_execution(self, seq):
        # Completion preserves the incremental radix path for generated tokens;
        # cancellation/preemption clears it through the separate reset hook.
        super().reset_prefill_execution_state(seq.seq_id)

    def _require_no_pending_step(self):
        if self._step is not None:
            raise RuntimeError("Native cache step must finish or abort before preparing another step.")

    def _prepare_prefill(self, seqs):
        self._require_no_pending_step()
        for seq in seqs:
            if seq.seq_id not in self.pool.live and seq.prefix_cache_hit_len:
                self.prefix.attach_hit(seq)
        requests = tuple((seq.seq_id, seq.num_prefilled_tokens, seq.num_prefilled_tokens + seq.current_chunk_size) for seq in seqs)
        plan = self.prefix.plan(seqs, requests) if self.prefix is not None else None
        try:
            self._step = self.pool.reserve_prefill(requests, snapshot_ends=plan.snapshot_ends if plan is not None else None)
        except Exception:
            if plan is not None:
                self.prefix.finish_plan(plan)
            raise
        self._prefix_plan, self._batches = plan, self._step.batches
        self._executions = tuple(NativeAttentionExecution(layer, batch) for layer, batch in zip(self.pool.layers, self._batches))
        ids = [token for seq in seqs for token in seq.token_ids[seq.num_prefilled_tokens:seq.num_prefilled_tokens + seq.current_chunk_size]]
        batch = self._batches[0]
        self._context_lens = batch.start_positions + batch.cu_seqlens[1:] - batch.cu_seqlens[:-1]
        return torch.tensor(ids, dtype=torch.int64, device=self.device), batch.window.positions.long(), batch.cu_seqlens

    def init_decode_graph_state(self, contract, inputs):
        inputs.validate(contract)
        native = self.pool.make_decode_state(contract.batch_capacity)
        executions = tuple(NativeAttentionExecution(layer, batch) for layer, batch in zip(self.pool.layers, native.batches))
        state = NativeCacheDecodeGraphState(contract, inputs, native, executions)
        self._batches = state.native.batches
        self._executions, self._context_lens = state.executions, inputs.context_lens
        return state

    def prepare_decode_graph_step(self, seqs, state):
        self._require_no_pending_step()
        requests = tuple((seq.seq_id, seq.decode_input_position) for seq in seqs)
        ranges = tuple((seq_id, position, position + 1) for seq_id, position in requests)
        plan = self.prefix.plan(seqs, ranges) if self.prefix is not None else None
        try:
            self._step = self.pool.reserve_decode(requests, state.native,
                                                  snapshot_sequence_ids=tuple(plan.snapshot_ends) if plan is not None else ())
        except Exception:
            if plan is not None:
                self.prefix.finish_plan(plan)
            raise
        self._prefix_plan, self._batches = plan, self._step.batches
        inputs, host = state.inputs, state.inputs.host
        self._executions, self._context_lens = state.executions, inputs.context_lens
        host.input_ids.fill_(-1)
        host.positions.fill_(-1)
        host.context_lens.zero_()
        host.request_indices.fill_(-1)
        host.active_mask.zero_()
        for token, seq in enumerate(seqs):
            host.input_ids[token] = seq.decode_input_token
            host.positions[token] = seq.decode_input_position
            host.context_lens[token] = seq.decode_input_position + 1
            host.request_indices[token] = self.pool.live[seq.seq_id].row
            host.active_mask[token] = True
        for destination, source in zip((inputs.input_ids, inputs.positions, inputs.context_lens,
                                        inputs.request_indices, inputs.active_mask), host.tensors()):
            destination.copy_(source, non_blocking=True)
        inputs.write_slot_mapping.fill_(-1)
        return inputs.input_ids, inputs.positions, None

    def prepare_decode_graph_in(self, state):
        self._batches = state.native.batches
        self._executions, self._context_lens = state.executions, state.inputs.context_lens
        state.native.publish()
        self.pool.layers[0].regions.commit_slots(self._batches[0].window, state.inputs.write_slot_mapping)

    def _prepare_decode(self, seqs):
        capacity = max(1, len(seqs))
        if capacity not in self._eager_decode:
            from sparsevllm.engine.decode_graph_contract import DecodeGraphContract
            contract = DecodeGraphContract("deepseek_v4", "unified", capacity, self.max_model_len)
            inputs = DecodeGraphInputs.allocate(contract, device=self.device, pin_memory=self.device.type == "cuda")
            self._eager_decode[capacity] = self.init_decode_graph_state(contract, inputs)
        state = self._eager_decode[capacity]
        result = self.prepare_decode_graph_step(seqs, state)
        self.prepare_decode_graph_in(state)
        return result

    def on_forward_end(self, seqs, is_prefill):
        if self._step is None:
            raise RuntimeError("Native forward completion requires a pending step.")
        self.pool.commit(self._step)
        if self._prefix_plan is not None:
            self.prefix.publish(self._prefix_plan, self._step)
        self._step = self._prefix_plan = None
        super().on_forward_end(seqs, is_prefill)

    def abort_step(self):
        if self._step is not None:
            self.pool.abort(self._step)
            if self._prefix_plan is not None:
                self.prefix.finish_plan(self._prefix_plan)
            self._step = self._prefix_plan = None

    def free_seq(self, seq_id):
        if self._step is not None and seq_id in self._step.sequence_ids:
            self.abort_step()
        if seq_id in self.pool.live:
            self.pool.release_sequence(seq_id)
        self.reset_prefill_execution_state(seq_id)

    def get_layer_batch_states(self, layer_idx):
        batch = self._batches[layer_idx]
        return LayerBatchStates(slot_mapping=batch.window_write_slots, req_indices=batch.request_rows,
                                context_lens=self._context_lens, max_context_len=self.max_model_len)

    def native_attention_executions(self):
        return self._executions

    def decode_graph_state_keepalive_tensors(self, state):
        return list(self.pool.accounting_tensors()) + list(state.native.keepalive_tensors())

    def free_slot_stats(self):
        return {"free_slots": self.num_free_slots, "free_rows": len(self.pool.free_live_rows),
                **{f"compressed_{ratio}": allocator.free_count for ratio, allocator in self.pool.slots.items()},
                **(self.prefix_cache.stats() if self.prefix_cache is not None else {})}

    def debug_live_seq_slots(self):
        return {seq_id: row.length for seq_id, row in self.pool.live.items()}

    def debug_state_summary(self):
        return {**super().debug_state_summary(),
                "native_rows": {seq_id: {"row": live.row, "length": live.length}
                                for seq_id, live in self.pool.live.items()},
                "native_capacity": self.free_slot_stats(),
                "native_allocated_bytes": self.pool.allocated_bytes()}

    def reset_after_warmup(self):
        self.abort_step()
        for seq_id in tuple(self.pool.live):
            self.free_seq(seq_id)
        if self.prefix is not None:
            for seq_id in tuple(self.prefix.paths):
                self.prefix.release_sequence(seq_id)
            rows = len(self.pool.free_snapshot_rows) + len(self.pool.snapshots)
            if not self.prefix.evict_for_capacity({}, snapshot_rows=rows):
                raise RuntimeError("Warmup reset could not release native prefix snapshots.")

    def get_layer_kv_cache(self, layer_idx):
        raise TypeError("Native shared KV uses NativeAttentionLayerStorage, not explicit K/V tensors.")

    def get_layer_store_view(self, layer_idx):
        raise TypeError("Native shared KV uses typed window/compression writes.")

    def get_layer_compute_tensors(self, layer_idx, selection=None):
        raise TypeError("Native shared KV uses indexed typed attention views.")

    def get_layer_buffer_req_to_token_slots(self, layer_idx):
        raise TypeError("Native shared KV has distinct window and compressed coordinates.")

    def free_part_slots(self, layer_idx, seq, keep_indices):
        raise TypeError("Native compressed history cannot be compacted as tokenwise explicit KV.")
