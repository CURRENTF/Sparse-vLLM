from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import torch

from sparsevllm.kernels.triton.indexed_host_copy import append_rows, gather_rows
from sparsevllm.utils.context import get_context

from .omnikv_storage import OmniKVStorage, payload_tensors
from .standard import StandardCacheManager
from .storage import HeterogeneousExplicitKVStorage


class OmniKVCacheManager(StandardCacheManager):
    def __init__(self, config, parallel_context, *, allocation_budget_bytes=None):
        self.offload_enabled = bool(config.enable_omnikv_offload)
        self._prefetched = set()
        self._current_writes = {}
        super().__init__(
            config, parallel_context, allocation_budget_bytes=allocation_budget_bytes
        )

    def allocate_kv_cache(self):
        if not self.offload_enabled:
            return super().allocate_kv_cache()
        original = self.attention_cache_storage
        if self.device.type != "cuda" or isinstance(
            original, HeterogeneousExplicitKVStorage
        ):
            raise ValueError(
                "OmniKV offload requires CUDA uniform explicit KV or MLA latent storage."
            )
        if original.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("OmniKV offload supports FP16/BF16 cache storage.")
        full = [self.kv_layer_index(i) for i in self.config.full_attention_layers]
        # Before the first observer, GPU-only OmniKV also consumes full history.
        if full:
            full = sorted(set(full) | set(range(min(full))))
        sparse = self.num_kv_layers - len(full)
        if not full or sparse <= 0:
            raise ValueError(
                "OmniKV offload requires both full and sparse attention layers."
            )
        available, per_layer = self._get_available_slots_info()
        self.selected_capacity = min(
            self.max_model_len,
            self.config.sink_keep_tokens
            + self.config.decode_keep_tokens
            + self.config.recent_keep_tokens,
        )
        if self.selected_capacity == 0:
            raise ValueError(
                "OmniKV offload requires a positive total selected-token budget."
            )
        staging_slots = self.max_buffer_rows * self.selected_capacity
        fixed_bytes = sparse * staging_slots * per_layer + staging_slots * 4
        metadata_bytes = self.max_buffer_rows * self.max_model_len * 4
        # One shared full-context prefill buffer, plus the full-attention layers.
        slot_bytes = (len(full) + 1) * per_layer + 8
        slots = (available - fixed_bytes - metadata_bytes) // slot_bytes
        # Host backing is bounded across worker processes, not once per GPU.
        meminfo = dict(
            line.split(":", 1)
            for line in Path("/proc/meminfo").read_text().splitlines()
        )
        host_budget = (
            int(meminfo["MemAvailable"].split()[0]) * 1024 // (2 * self.world_size)
        )
        prefix_bytes = (
            int((self.config.prefix_cache_host_size_gb or 0) * 1024**3)
            if self.config.enable_prefix_cache_offload
            else 0
        )
        self.prefix_host_blocks = prefix_bytes // (
            self.num_kv_layers * per_layer * self.config.prefix_cache_block_size
        )
        host_budget -= prefix_bytes
        slots = min(
            slots,
            host_budget // (sparse * per_layer),
            self.max_model_len
            * self.max_buffer_rows
            * (2 if self.config.enable_prefix_caching else 1),
        )
        if self.world_size > 1:
            capacity = torch.tensor(slots, dtype=torch.int64, device=self.device)
            self.parallel_context.world_all_reduce(
                capacity, op=torch.distributed.ReduceOp.MIN
            )
            slots = int(capacity.item())
        if slots <= 0 or (
            getattr(self.config, "startup_cache_phase", "production") != "profiling"
            and slots < self.max_model_len
        ):
            raise MemoryError(
                f"OmniKV offload pools cannot fit: slots={slots}, max_model_len={self.max_model_len}."
            )
        self.config.num_kvcache_slots = int(slots)
        if self.config.prefix_cache_max_blocks is not None:
            self.config.prefix_cache_max_blocks = min(
                self.config.prefix_cache_max_blocks,
                slots // self.config.prefix_cache_block_size,
            )
        storage = OmniKVStorage(
            original,
            num_layers=self.num_kv_layers,
            num_slots=slots,
            full_layers=full,
            device=self.device,
            prefix_slots=self.prefix_host_blocks * self.config.prefix_cache_block_size,
        )
        self.attention_cache_storage = storage
        self.kv_cache = None

        def allocate(count):
            return tuple(
                torch.empty(count, *shape, dtype=storage.dtype, device=self.device)
                for shape in storage.shapes
            )

        self.prefill_staging = allocate(slots)
        self.selected_staging = {
            i: allocate(staging_slots)
            for i in range(self.num_kv_layers)
            if i not in storage.full_layers
        }
        self.selected_slots = torch.arange(
            staging_slots, dtype=torch.int32, device=self.device
        ).view(self.max_buffer_rows, self.selected_capacity)
        self.selected_rows = torch.arange(
            self.max_buffer_rows, dtype=torch.int32, device=self.device
        )
        self.prefetch_stream = torch.cuda.Stream(device=self.device)
        self.selection_done = torch.cuda.Event()
        self.layer_ready = {i: torch.cuda.Event() for i in self.selected_staging}
        self._selection_pending = False

    def _init_prefix_offload(self):
        if not self.offload_enabled:
            return super()._init_prefix_offload()
        from .omnikv_prefix import OmniKVPrefixOffloadController, OmniKVPrefixPool

        if self.prefix_host_blocks <= 0:
            raise ValueError(
                "OmniKV prefix offload requires positive prefix_cache_host_size_gb."
            )
        required = self.config.num_kvcache_slots // self.prefix_cache_block_size
        if self.prefix_cache.max_blocks is not None:
            required = min(required, self.prefix_cache.max_blocks)
        if self.prefix_host_blocks < required:
            raise ValueError(
                f"Prefix host pool needs {required} blocks, has {self.prefix_host_blocks}."
            )
        pool = OmniKVPrefixPool(
            self.attention_cache_storage,
            self.prefix_host_blocks,
            self.prefix_cache_block_size,
            self.device,
        )
        self.prefix_offload_controller = OmniKVPrefixOffloadController(
            prefix_cache=self.prefix_cache,
            storage=self.attention_cache_storage,
            host_pool=pool,
            block_size=self.prefix_cache_block_size,
            device=self.device,
        )

    def _iter_accounting_tensors(self):
        yield from super()._iter_accounting_tensors()
        if self.offload_enabled and self.prefix_offload_controller is not None:
            pool = self.prefix_offload_controller.host_pool
            for layer, parts in enumerate(pool.layers):
                for component, tensor in enumerate(parts):
                    yield f"prefix_host_cache.{layer}.{component}", tensor

    def prefix_kv_payload_nbytes(self, payload):
        if not self.offload_enabled:
            return super().prefix_kv_payload_nbytes(payload)
        return (
            payload.token_slots.numel()
            * len(self.attention_cache_storage.full_layers)
            * self.attention_cache_storage.bytes_per_slot_per_layer()
        )

    def memory_accounting(self):
        result = super().memory_accounting()
        if self.offload_enabled:
            tensors = result["tensors"]
            gpu_bytes = sum(
                t["nbytes"] for t in tensors if t["device"].startswith("cuda")
            )
            host_bytes = sum(t["nbytes"] for t in tensors if t["device"] == "cpu")
            baseline = (
                self.config.num_kvcache_slots
                * self.num_kv_layers
                * self.attention_cache_storage.bytes_per_slot_per_layer()
            )
            result.update(
                allocated_device_tensor_bytes=gpu_bytes,
                allocated_host_tensor_bytes=host_bytes,
                omnikv_gpu_only_kv_bytes=baseline,
                observed_savings=1 - gpu_bytes / baseline,
            )
        return result

    def debug_state_summary(self):
        result = super().debug_state_summary()
        if self.offload_enabled:
            accounting = self.memory_accounting()
            result["omnikv_offload"] = {
                name: accounting[name]
                for name in (
                    "allocated_device_tensor_bytes",
                    "allocated_host_tensor_bytes",
                    "omnikv_gpu_only_kv_bytes",
                    "observed_savings",
                )
            }
            result["omnikv_offload"].update(
                slots=self.config.num_kvcache_slots,
                selected_capacity=self.selected_capacity,
                staging_rows=self.max_buffer_rows,
                full_layers=len(self.attention_cache_storage.full_layers),
            )
            if self.prefix_offload_controller is not None:
                result["omnikv_offload"]["prefix_transfer"] = (
                    self.prefix_offload_controller.stats()
                )
        return result

    def begin_selection_step(self):
        self._prefetched.clear()
        self._current_writes.clear()
        self._selection_pending = False

    def _prepare_prefill(self, seqs):
        self.begin_selection_step()
        return super()._prepare_prefill(seqs)

    def _prepare_decode(self, seqs):
        self.begin_selection_step()
        return super()._prepare_decode(seqs)

    def _prepare_decode_graph_buffers(self, seqs, **kwargs):
        self.begin_selection_step()
        return super()._prepare_decode_graph_buffers(seqs, **kwargs)

    @contextmanager
    def selection_stream(self):
        self.prefetch_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.prefetch_stream):
            yield

    def prefetch_selections(self, selections):
        self.selection_done.record(self.prefetch_stream)
        self._selection_pending = True
        for layer_idx, selection in selections:
            kv_idx = self.kv_layer_index(layer_idx)
            slots = self._default_active_slots_for_selection(layer_idx, selection)
            self._gather_decode(
                kv_idx, slots, selection.req_indices, selection.context_lens
            )
            self.layer_ready[kv_idx].record(self.prefetch_stream)
            self._prefetched.add(kv_idx)

    def _gather_decode(self, kv_idx, slots, rows, lengths):
        for component, destination in enumerate(self.selected_staging[kv_idx]):
            gather_rows(
                self.attention_cache_storage.pointers[kv_idx],
                destination,
                slots,
                rows,
                lengths,
                capacity=self.selected_capacity,
                component=component,
                skip_last=self.config.recent_keep_tokens > 0,
                exclude_slots=self.layer_batch_state.slot_mapping,
                # Bound the whole batch footprint to leave SMs for model work.
                max_blocks=max(1, 32 // rows.numel()),
                slot_map=self.attention_cache_storage.host_slot_map,
            )

    def store_attention_payload(self, layer_idx, payload):
        slots = super().store_attention_payload(layer_idx, payload)
        if (
            self.offload_enabled
            and not get_context().is_prefill
            and self.kv_layer_index(layer_idx) in self.selected_staging
        ):
            self._current_writes[layer_idx] = payload
        return slots

    def get_layer_kv_cache(self, layer_idx):
        if not self.offload_enabled:
            return super().get_layer_kv_cache(layer_idx)
        return payload_tensors(
            self.attention_cache_storage.layer_payload(self.kv_layer_index(layer_idx))
        )

    def get_layer_compute_payload(
        self, layer_idx, active_slots, req_indices, context_lens, selection=None
    ):
        if not self.offload_enabled:
            return super().get_layer_compute_payload(
                layer_idx, active_slots, req_indices, context_lens, selection
            )
        kv_idx = self.kv_layer_index(layer_idx)
        storage = self.attention_cache_storage
        stream = torch.cuda.current_stream(self.device)
        if kv_idx in storage.full_layers:
            # The next observer reuses the shared raw score buffer.
            if self._selection_pending:
                stream.wait_event(self.selection_done)
            return (
                storage.layer_payload(kv_idx),
                active_slots,
                req_indices,
                context_lens,
            )
        if kv_idx in self._prefetched:
            stream.wait_event(self.layer_ready[kv_idx])
        else:
            self._gather_decode(kv_idx, active_slots, req_indices, context_lens)
        parts = self.selected_staging[kv_idx]
        for source, destination in zip(
            payload_tensors(self._current_writes.pop(layer_idx)), parts
        ):
            append_rows(
                source,
                destination,
                context_lens,
                self.layer_batch_state.slot_mapping,
                self.selected_capacity,
                table=active_slots if self.config.recent_keep_tokens == 0 else None,
                rows=req_indices if self.config.recent_keep_tokens == 0 else None,
            )
        batch = req_indices.numel()
        return (
            storage.make_payload(parts),
            self.selected_slots[:batch],
            self.selected_rows[:batch],
            context_lens,
        )

    def get_prefill_compute_payload(
        self,
        layer_idx,
        k_current,
        v_current,
        selection,
        active_slots,
        req_indices,
        context_lens,
    ):
        if not self.offload_enabled:
            return super().get_prefill_compute_payload(
                layer_idx,
                k_current,
                v_current,
                selection,
                active_slots,
                req_indices,
                context_lens,
            )
        kv_idx = self.kv_layer_index(layer_idx)
        storage = self.attention_cache_storage
        if kv_idx in storage.full_layers:
            return (
                storage.layer_payload(kv_idx),
                active_slots,
                req_indices,
                context_lens,
            )
        for component, destination in enumerate(self.prefill_staging):
            gather_rows(
                storage.pointers[kv_idx],
                destination,
                active_slots,
                req_indices,
                context_lens,
                capacity=int(selection.max_context_len),
                component=component,
                scatter=True,
                slot_map=storage.host_slot_map,
            )
        return (
            storage.make_payload(self.prefill_staging),
            active_slots,
            req_indices,
            context_lens,
        )

    def decode_graph_keepalive_tensors(self):
        result = super().decode_graph_keepalive_tensors()
        if self.offload_enabled:
            result.extend(self.attention_cache_storage.accounting_tensors())
            result.extend(self.prefill_staging)
            result.extend(x for parts in self.selected_staging.values() for x in parts)
            result.extend((self.selected_slots, self.selected_rows))
        return result

    def on_forward_end(self, seqs, is_prefill):
        if self.offload_enabled:
            # Join every captured fork and protect staging before the next step.
            torch.cuda.current_stream(self.device).wait_stream(self.prefetch_stream)
        return super().on_forward_end(seqs, is_prefill)
