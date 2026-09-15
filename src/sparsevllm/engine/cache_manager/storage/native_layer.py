"""Native per-layer physical storage; admission and radix ownership stay above it."""

from dataclasses import dataclass

import torch

from ..base import SharedKVWrite
from ..methods.deepseek_v4 import CompressionPlan
from ..native_attention import CompressedIndexView, CompressionBatchView, NativeStateSnapshots, SharedKVWindowBatch
from .shared_kv import CompressionCarryStorage, SharedKVStorage
from .shared_kv_views import SharedKVRegions


@dataclass(frozen=True)
class NativeAttentionBatch:
    window: SharedKVWindowBatch
    window_write_slots: torch.Tensor
    window_commit_slots: torch.Tensor
    attention_indices: torch.Tensor
    request_rows: torch.Tensor
    cu_seqlens: torch.Tensor
    start_positions: torch.Tensor
    state_snapshots: NativeStateSnapshots | None = None
    compression: CompressionBatchView | None = None
    index_compression: CompressionBatchView | None = None
    compressed_write_slots: torch.Tensor | None = None
    index_write_slots: torch.Tensor | None = None
    compressed_lengths: torch.Tensor | None = None
    index_view: CompressedIndexView | None = None
    selected_compressed: torch.Tensor | None = None


class NativeAttentionLayerStorage:
    def __init__(self, *, regions: SharedKVRegions, head_dim, ratio, compressed_slots, device, index_dim=128):
        if ratio not in (0, 4, 128):
            raise ValueError("Native attention storage requires ratio 0, 4 or 128.")
        if ratio and (compressed_slots is None or compressed_slots.ndim != 2
                      or len(compressed_slots) != regions.num_request_rows or compressed_slots.dtype != torch.int32):
            raise ValueError("Compressed layers require an int32 per-request physical slot table.")
        if not ratio and compressed_slots is not None:
            raise ValueError("Window-only layers do not have compressed slots.")
        self.regions, self.ratio, self.compressed_slots = regions, ratio, compressed_slots
        self.kv = SharedKVStorage(head_dim=head_dim)
        self.kv.allocate(num_layers=1, num_slots=regions.num_slots, device=device)
        self.device = self.kv.cache.device
        self.carry = self.index_carry = self.index = None
        if ratio:
            self.carry = CompressionCarryStorage(num_rows=regions.num_request_rows, ratio=ratio, head_dim=head_dim, device=device)
        if ratio == 4:
            self.index_carry = CompressionCarryStorage(num_rows=regions.num_request_rows, ratio=ratio, head_dim=index_dim, device=device)
            self.index = SharedKVStorage(head_dim=index_dim)
            self.index.allocate(num_layers=1, num_slots=regions.compressed_capacity, device=device)

    def accounting_tensors(self):
        tensors = list(self.kv.accounting_tensors())
        for component in (self.carry, self.index_carry, self.index):
            if component is not None:
                tensors.extend(component.accounting_tensors())
        # The parent cache manager owns and accounts shared per-ratio slot tables.
        return tuple(tensors)

    def _ints(self, values):
        return torch.tensor(values, device=self.device, dtype=torch.int32)

    def _batch(self, window, request_rows, cu, starts, boundary_requests, boundary_ends, *, decode, snapshots=None):
        n = len(window.positions)
        commit = torch.empty(n, device=self.device, dtype=torch.int32)
        writes = commit if decode else torch.arange(self.regions.temporary_offset, self.regions.temporary_offset + n,
                                                     device=self.device, dtype=torch.int32)
        capacity = 512 if self.ratio == 4 else self.compressed_slots.shape[1] if self.ratio else 0
        indices = torch.empty((n, 1, self.regions.window_size + capacity), device=self.device, dtype=torch.int32)
        kwargs = {}
        if self.ratio:
            metadata = (request_rows, cu, starts, boundary_requests, boundary_ends, decode, snapshots)
            kwargs["compression"] = CompressionBatchView(*self.carry.accounting_tensors(), *metadata)
            events = len(request_rows) if decode else len(boundary_requests)
            kwargs["compressed_write_slots"] = torch.empty(events, device=self.device, dtype=torch.int32)
            lengths = torch.empty(n, device=self.device, dtype=torch.int32)
            kwargs["compressed_lengths"] = lengths
            if self.index is not None:
                kwargs["index_compression"] = CompressionBatchView(*self.index_carry.accounting_tensors(), *metadata)
                kwargs["index_write_slots"] = torch.empty(events, device=self.device, dtype=torch.int32)
                kwargs["index_view"] = CompressedIndexView(self.index.cache[0, :, 0], self.compressed_slots,
                                                           window.request_rows, lengths)
                kwargs["selected_compressed"] = torch.empty((n, 512), device=self.device, dtype=torch.int32)
        return NativeAttentionBatch(window, writes, commit, indices, request_rows, cu, starts, snapshots, **kwargs)

    def make_prefill_batch(self, chunks, *, snapshots=()):
        chunks = tuple(chunks)
        plan = CompressionPlan.prefill(chunks, self.ratio or 4)
        if plan.cu_seqlens[-1] > self.regions.prefill_capacity:
            raise ValueError("Packed native prefill exceeds the temporary region capacity.")
        if any(row >= self.regions.num_request_rows for row in plan.request_rows):
            raise ValueError("Native prefill references an unallocated request row.")
        if self.ratio and any((c.start + c.length) // self.ratio > self.compressed_slots.shape[1] for c in chunks):
            raise ValueError("Native prefill exceeds the compressed slot-table capacity.")
        snapshot_view = None
        if snapshots:
            sources = {c.request_row: (i, c) for i, c in enumerate(chunks)}
            destinations = set()
            requests, snapshot_rows, snapshot_ends = [], [], []
            for snapshot in snapshots:
                if (snapshot.snapshot_row in sources or snapshot.snapshot_row in destinations
                        or not 0 <= snapshot.snapshot_row < self.regions.num_request_rows):
                    raise ValueError("Snapshot targets must be distinct allocated rows outside the active batch.")
                if snapshot.request_row not in sources:
                    raise ValueError("Snapshot source must be an active prefill request.")
                batch, chunk = sources[snapshot.request_row]
                if not chunk.start < snapshot.end <= chunk.start + chunk.length:
                    raise ValueError("Snapshot end must lie within its source prefill chunk.")
                destinations.add(snapshot.snapshot_row)
                requests.append(batch)
                snapshot_rows.append(snapshot.snapshot_row)
                snapshot_ends.append(snapshot.end)
            snapshot_view = NativeStateSnapshots(*map(self._ints, (requests, snapshot_rows, snapshot_ends)))
        rows, positions, starts, ends, packed = [], [], [], [], []
        for offset, c in zip(plan.cu_seqlens, chunks):
            rows.extend([c.request_row] * c.length)
            positions.extend(range(c.start, c.start + c.length))
            starts.extend([c.start] * c.length)
            ends.extend([c.start + c.length] * c.length)
            packed.extend([offset] * c.length)
        window = SharedKVWindowBatch(*map(self._ints, (rows, positions, starts, ends, packed)))
        return self._batch(window, *map(self._ints, (plan.request_rows, plan.cu_seqlens, plan.start_positions,
                                                    plan.boundary_requests, plan.boundary_ends)), decode=False,
                           snapshots=snapshot_view)

    def make_decode_batch(self, batch_capacity, *, capture_snapshots=False):
        if batch_capacity <= 0:
            raise ValueError("Decode graph batch capacity must be positive.")
        rows = torch.full((batch_capacity,), -1, device=self.device, dtype=torch.int32)
        positions = torch.full_like(rows, -1)
        window = SharedKVWindowBatch(rows, positions)
        snapshots = NativeStateSnapshots(torch.arange(batch_capacity, device=self.device, dtype=torch.int32),
                                         torch.full_like(rows, -1), torch.empty_like(rows)) if capture_snapshots else None
        return self._batch(window, rows, torch.arange(batch_capacity + 1, device=self.device, dtype=torch.int32),
                           positions, self._ints([]), self._ints([]), decode=True, snapshots=snapshots)

    def prepare(self, batch):
        self.regions.commit_slots(batch.window, batch.window_commit_slots)
        if batch.state_snapshots is not None and batch.window.chunk_starts is None:
            torch.add(batch.start_positions, 1, out=batch.state_snapshots.ends)
        if self.ratio:
            from sparsevllm.kernels.triton.deepseek_v4.layer_metadata import prepare_layer_metadata
            prepare_layer_metadata(batch.window, batch.compression, self.compressed_slots, batch.compressed_lengths,
                                   batch.compressed_write_slots, batch.index_write_slots, ratio=self.ratio,
                                   compressed_offset=self.regions.compressed_offset)

    def store_window(self, batch, values):
        self.kv.store(0, batch.window_write_slots, SharedKVWrite(values))
        if batch.state_snapshots is not None:
            from sparsevllm.kernels.triton.deepseek_v4.window_snapshot import snapshot_window
            snapshot_window(self.kv.layer_payload(0).cache, batch.request_rows, batch.cu_seqlens, batch.start_positions,
                            batch.state_snapshots, window_size=self.regions.window_size,
                            temporary_offset=self.regions.temporary_offset, decode=batch.window.chunk_starts is None)

    def restore_state_rows(self, source_rows, destination_rows):
        """Restore private mutable state; the parent separately attaches shared compressed slots."""
        if source_rows.shape != destination_rows.shape or source_rows.ndim != 1:
            raise ValueError("Native state restore requires equal source/destination row vectors.")
        offsets = torch.arange(self.regions.window_size, device=self.device, dtype=torch.int32)
        source_slots = (source_rows[:, None] * self.regions.window_size + offsets).flatten()
        destination_slots = (destination_rows[:, None] * self.regions.window_size + offsets).flatten()
        self.kv.copy_slots(0, source_slots, destination_slots)
        for carry in (self.carry, self.index_carry):
            if carry is not None:
                carry.copy_rows(source_rows, destination_rows)

    def store_compressed(self, batch, values, positions):
        self.kv.store(0, batch.compressed_write_slots, SharedKVWrite(values[:, None], positions))

    def store_index(self, batch, values, positions):
        self.index.store(0, batch.index_write_slots, SharedKVWrite(values[:, None], positions))

    def attention_view(self, batch, selected=None):
        if self.ratio == 128:
            return self.regions.attention_view(self.kv.layer_payload(0).cache, batch.window, batch.attention_indices,
                                               compressed=self.compressed_slots, compressed_lengths=batch.compressed_lengths,
                                               compressed_per_request=True)
        return self.regions.attention_view(self.kv.layer_payload(0).cache, batch.window, batch.attention_indices,
                                           compressed=selected)

    def finish_window(self, batch, values):
        if batch.window.chunk_starts is not None:
            self.kv.store(0, batch.window_commit_slots, SharedKVWrite(values))
