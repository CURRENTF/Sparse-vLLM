"""Physical native cache allocation, prefix snapshots and shared-slot lifetime."""

from collections import deque
from dataclasses import dataclass, replace

import numpy as np
import torch

from ..methods.deepseek_v4 import CompressionChunk, StateSnapshot
from ..native_attention import SharedKVWindowBatch
from .native_layer import NativeAttentionLayerStorage
from .native_capacity import NativeCacheGeometry
from .shared_kv_views import SharedKVRegions


class _SharedSlots:
    def __init__(self, capacity):
        self.free = np.arange(capacity, dtype=np.int32)
        self.free_count = capacity
        self.refs = np.zeros(capacity, dtype=np.int32)

    def allocate(self, count):
        if not 0 <= count <= self.free_count:
            raise RuntimeError("Native compressed slot capacity exceeded.")
        self.free_count -= count
        slots = self.free[self.free_count:self.free_count+count].copy()
        self.refs[slots] = 1
        return slots

    def retain(self, slots):
        self.refs[slots] += 1

    def release(self, slots):
        self.refs[slots] -= 1
        freed = slots[self.refs[slots] == 0]
        self.free[self.free_count:self.free_count+len(freed)] = freed
        self.free_count += len(freed)


@dataclass
class NativeLiveRow:
    row: int
    length: int = 0
    pending_end: int | None = None
    pending_token: object | None = None


@dataclass(eq=False)
class NativePrefixSnapshot:
    row: int
    length: int
    ready: bool = False


@dataclass
class NativePoolStep:
    reservation: object
    sequence_ids: tuple[int, ...]
    chunks: tuple[CompressionChunk, ...]
    snapshots: tuple[NativePrefixSnapshot, ...]
    snapshot_plan: tuple[StateSnapshot, ...]
    batches: tuple
    decode: bool = False


@dataclass
class NativeDecodeState:
    pool: "NativeCachePool"
    host: torch.Tensor
    device: torch.Tensor
    batches: tuple

    @property
    def capacity(self):
        return self.device.shape[1]

    def publish(self):
        from sparsevllm.kernels.triton.deepseek_v4.publish_slots import publish_compressed_slots
        for index, (ratio, table) in enumerate(self.pool.tables.items()):
            publish_compressed_slots(table, self.device[0], self.device[1], self.device[3 + index],
                                     self.device[2], ratio=ratio)

    def keepalive_tensors(self):
        return self.host, self.device


class NativeCachePool:
    def __init__(self, *, ratios, live_rows, snapshot_rows, max_model_len, prefill_capacity,
                 compressed_capacities, device, head_dim=512, index_dim=128, window_size=128):
        if (not ratios or any(r not in (0, 4, 128) for r in ratios)
                or min(live_rows, max_model_len, prefill_capacity) <= 0 or snapshot_rows < 0):
            raise ValueError("Native pools require valid layer ratios and positive live/token capacities.")
        required = set(ratios) - {0}
        if set(compressed_capacities) != required or any(c <= 0 for c in compressed_capacities.values()):
            raise ValueError("Native compressed capacities must cover exactly the configured nonzero ratios.")
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.max_model_len, self.prefill_capacity = max_model_len, prefill_capacity
        self.geometry = NativeCacheGeometry(tuple(ratios), max_model_len, prefill_capacity, head_dim, index_dim, window_size)
        self.free_live_rows = deque(range(live_rows))
        self.free_snapshot_rows = deque(range(live_rows, live_rows + snapshot_rows))
        self.live = {}
        self.snapshots = {}
        self.slots = {ratio: _SharedSlots(capacity) for ratio, capacity in compressed_capacities.items()}
        self.host_tables = {ratio: np.full((live_rows + snapshot_rows, max(1, max_model_len // ratio)), -1, dtype=np.int32)
                            for ratio in self.slots}
        self.tables = {ratio: torch.full(table.shape, -1, dtype=torch.int32, device=self.device)
                       for ratio, table in self.host_tables.items()}
        self.layers = tuple(NativeAttentionLayerStorage(
            regions=SharedKVRegions(live_rows + snapshot_rows, window_size,
                                    compressed_capacities.get(ratio, 0), prefill_capacity),
            head_dim=head_dim, ratio=ratio, compressed_slots=self.tables.get(ratio),
            device=self.device, index_dim=index_dim,
        ) for ratio in ratios)

    def accounting_tensors(self):
        return (*self.tables.values(), *(tensor for layer in self.layers for tensor in layer.accounting_tensors()))

    def allocated_bytes(self):
        return sum(t.numel() * t.element_size() for t in self.accounting_tensors())

    def _row_slots(self, ratio, row, length):
        return self.host_tables[ratio][row, :length // ratio]

    @torch.inference_mode()
    def _clear_row(self, row):
        for ratio, table in self.tables.items():
            table[row].fill_(-1)
            self.host_tables[ratio][row].fill(-1)

    def _copy_tables(self, source, destination, length):
        for ratio, allocator in self.slots.items():
            slots = self._row_slots(ratio, source, length)
            allocator.retain(slots)
            self.host_tables[ratio][destination, :len(slots)] = slots
            self.tables[ratio][destination, :len(slots)].copy_(self.tables[ratio][source, :len(slots)])

    def append_costs(self, start, end):
        return {ratio: end // ratio - start // ratio for ratio in self.slots}

    def _validate_append(self, seq_id, start, end, *, allow_new):
        state = self.live.get(seq_id)
        if not 0 <= start < end <= self.max_model_len:
            raise ValueError("Native append range is outside the model capacity.")
        if ((state is None and (start != 0 or not allow_new))
                or (state is not None and (state.length != start or state.pending_end is not None))):
            raise ValueError("Native append must extend committed request state.")
        return state

    def _layer_batches(self, make_template):
        templates, batches = {}, []
        for layer in self.layers:
            template = templates.get(layer.ratio)
            if template is None:
                template = make_template(layer)
                templates[layer.ratio] = template
                batches.append(template)
            else:
                changes = {}
                for name, carry in (("compression", layer.carry), ("index_compression", layer.index_carry)):
                    if carry is not None:
                        changes[name] = replace(getattr(template, name), state_kv=carry.kv, state_gate=carry.gate)
                if layer.index is not None:
                    changes["index_view"] = replace(template.index_view, keys=layer.index.cache[0, :, 0])
                batches.append(replace(template, **changes))
        return tuple(batches)

    def reserve_prefill(self, requests, *, snapshot_ends=None):
        """Reserve a complete mixed step atomically before any model state is changed.

        Each request is (sequence_id, absolute_start, exclusive_end). Snapshot
        publication is delayed until all layers complete the step.
        """
        requests = tuple(requests)
        snapshot_ends = {} if snapshot_ends is None else snapshot_ends
        ids = tuple(seq_id for seq_id, _, _ in requests)
        if len(set(ids)) != len(ids) or set(snapshot_ends) - set(ids):
            raise ValueError("Native steps require unique sequence IDs and snapshot sources in the batch.")
        costs = dict.fromkeys(self.slots, 0)
        for seq_id, start, end in requests:
            self._validate_append(seq_id, start, end, allow_new=True)
            for ratio, cost in self.append_costs(start, end).items():
                costs[ratio] += cost
            if any(not start < boundary <= end for boundary in snapshot_ends.get(seq_id, ())):
                raise ValueError("Native snapshot boundaries must lie within their source step.")
        if sum(end-start for _, start, end in requests) > self.prefill_capacity:
            raise ValueError("Native packed prefill exceeds temporary storage capacity.")
        if (sum(seq_id not in self.live for seq_id in ids) > len(self.free_live_rows)
                or sum(map(len, snapshot_ends.values())) > len(self.free_snapshot_rows)
                or any(cost > self.slots[ratio].free_count for ratio, cost in costs.items())):
            raise RuntimeError("Insufficient native cache capacity for the complete step.")
        reservation = object()
        chunks, snapshots, plan, new_rows = [], [], [], []
        for seq_id, start, end in requests:
            state = self.live.get(seq_id)
            if state is None:
                state = NativeLiveRow(self.free_live_rows.popleft())
                self.live[seq_id] = state
                new_rows.append(state.row)
            for ratio, allocator in self.slots.items():
                begin, stop = start // ratio, end // ratio
                allocated = allocator.allocate(stop-begin)
                self.host_tables[ratio][state.row, begin:stop] = allocated
                self.tables[ratio][state.row, begin:stop].copy_(torch.from_numpy(allocated).to(self.device))
            state.pending_end = end
            state.pending_token = reservation
            chunks.append(CompressionChunk(state.row, start, end-start))
            for boundary in snapshot_ends.get(seq_id, ()):
                snapshot = NativePrefixSnapshot(self.free_snapshot_rows.popleft(), boundary)
                self._copy_tables(state.row, snapshot.row, boundary)
                self.snapshots[snapshot.row] = snapshot
                snapshots.append(snapshot)
                plan.append(StateSnapshot(state.row, boundary, snapshot.row))
        if new_rows:
            rows = torch.tensor(new_rows, dtype=torch.int32, device=self.device)
            for layer in self.layers:
                for carry in (layer.carry, layer.index_carry):
                    if carry is not None:
                        carry.clear_rows(rows)
        chunks, plan = tuple(chunks), tuple(plan)
        batches = self._layer_batches(lambda layer: layer.make_prefill_batch(chunks, snapshots=plan))
        return NativePoolStep(reservation, ids, chunks, tuple(snapshots), plan, batches)

    def make_decode_state(self, batch_capacity):
        if batch_capacity <= 0:
            raise ValueError("Native decode state requires a positive batch capacity.")
        # Rows, absolute positions, optional snapshot destinations and one
        # newly reserved slot per ratio travel in a single bounded H2D copy.
        shape = (3 + len(self.slots), batch_capacity)
        host = torch.full(shape, -1, dtype=torch.int32, device="cpu", pin_memory=self.device.type == "cuda")
        device = torch.full(shape, -1, dtype=torch.int32, device=self.device)
        window = SharedKVWindowBatch(device[0], device[1])
        cu = torch.arange(batch_capacity + 1, dtype=torch.int32, device=self.device)
        empty = torch.empty(0, dtype=torch.int32, device=self.device)
        batches = self._layer_batches(lambda layer: layer._batch(
            window, window.request_rows, cu, window.positions, empty, empty, decode=True,
        ))
        return NativeDecodeState(self, host, device, batches)

    def reserve_decode(self, requests, state, *, snapshot_sequence_ids=()):
        """Reserve on CPU; the captured publish hook updates device slot tables."""
        requests = tuple(requests)
        ids = tuple(seq_id for seq_id, _ in requests)
        snapshot_ids = set(snapshot_sequence_ids)
        if state.pool is not self or len(requests) > state.capacity:
            raise ValueError("Native decode state belongs to another pool or has insufficient batch capacity.")
        if len(set(ids)) != len(ids) or snapshot_ids - set(ids):
            raise ValueError("Native decode requires unique sequence IDs and active snapshot sources.")
        costs = dict.fromkeys(self.slots, 0)
        for seq_id, position in requests:
            self._validate_append(seq_id, position, position + 1, allow_new=False)
            for ratio, cost in self.append_costs(position, position + 1).items():
                costs[ratio] += cost
        if (len(snapshot_ids) > len(self.free_snapshot_rows)
                or any(cost > self.slots[ratio].free_count for ratio, cost in costs.items())):
            raise RuntimeError("Insufficient native cache capacity for the complete decode step.")
        reservation = object()
        chunks, snapshots, plan = [], [], []
        staged = state.host.numpy()
        staged.fill(-1)
        for token, (seq_id, position) in enumerate(requests):
            live = self.live[seq_id]
            for index, (ratio, allocator) in enumerate(self.slots.items()):
                if (position + 1) % ratio == 0:
                    slot = allocator.allocate(1)[0]
                    self.host_tables[ratio][live.row, position // ratio] = slot
                    staged[3 + index, token] = slot
            live.pending_end, live.pending_token = position + 1, reservation
            staged[0, token], staged[1, token] = live.row, position
            chunks.append(CompressionChunk(live.row, position, 1))
            if seq_id in snapshot_ids:
                snapshot = NativePrefixSnapshot(self.free_snapshot_rows.popleft(), position + 1)
                self._copy_tables(live.row, snapshot.row, snapshot.length)
                self.snapshots[snapshot.row] = snapshot
                snapshots.append(snapshot)
                plan.append(StateSnapshot(live.row, snapshot.length, snapshot.row))
                staged[2, token] = snapshot.row
        state.device.copy_(state.host, non_blocking=True)
        return NativePoolStep(reservation, ids, tuple(chunks), tuple(snapshots), tuple(plan), state.batches, decode=True)

    def _validate_pending_step(self, step):
        for seq_id, chunk in zip(step.sequence_ids, step.chunks):
            state = self.live.get(seq_id)
            if (state is None or state.row != chunk.request_row or state.pending_token is not step.reservation
                    or state.pending_end != chunk.start + chunk.length):
                raise RuntimeError("Native step does not own the pending request reservation.")
        for snapshot in step.snapshots:
            if self.snapshots.get(snapshot.row) is not snapshot or snapshot.ready:
                raise RuntimeError("Native step has a stale snapshot reservation.")

    def commit(self, step):
        self._validate_pending_step(step)
        if step.decode and step.snapshots:
            # A decode prefix ends at the current token. Copy after the graph
            # only on snapshot steps; prefill must capture intermediate state
            # in the forward kernels before its remaining tokens overwrite it.
            self._copy_state_rows([p.request_row for p in step.snapshot_plan],
                                  [p.snapshot_row for p in step.snapshot_plan])
        for seq_id in step.sequence_ids:
            state = self.live[seq_id]
            state.length, state.pending_end = state.pending_end, None
            state.pending_token = None
        for snapshot in step.snapshots:
            snapshot.ready = True

    def abort(self, step):
        """Discard failed requests and unpublished snapshots; no numerical rollback is promised."""
        self._validate_pending_step(step)
        for snapshot in step.snapshots:
            self.release_snapshot(snapshot)
        for seq_id in step.sequence_ids:
            self.release_sequence(seq_id)

    def attach(self, seq_id, snapshot):
        if self.snapshots.get(snapshot.row) is not snapshot or not snapshot.ready:
            raise ValueError("Cannot attach an unpublished or stale native prefix snapshot.")
        if seq_id in self.live:
            raise ValueError("Native prefix attachment requires a new sequence.")
        if not self.free_live_rows:
            raise RuntimeError("No private native request row is available.")
        row = self.free_live_rows.popleft()
        self._copy_tables(snapshot.row, row, snapshot.length)
        self._copy_state_rows([snapshot.row], [row])
        self.live[seq_id] = NativeLiveRow(row, snapshot.length)

    def _copy_state_rows(self, source_rows, destination_rows):
        source = torch.tensor(source_rows, device=self.device, dtype=torch.int32)
        destination = torch.tensor(destination_rows, device=self.device, dtype=torch.int32)
        for layer in self.layers:
            layer.restore_state_rows(source, destination)

    def release_snapshot(self, snapshot):
        if self.snapshots.get(snapshot.row) is not snapshot:
            raise ValueError("Native prefix snapshot was already released or recycled.")
        for ratio, allocator in self.slots.items():
            allocator.release(self._row_slots(ratio, snapshot.row, snapshot.length))
        del self.snapshots[snapshot.row]
        self._clear_row(snapshot.row)
        self.free_snapshot_rows.append(snapshot.row)

    def release_sequence(self, seq_id):
        state = self.live.pop(seq_id)
        length = state.pending_end if state.pending_end is not None else state.length
        for ratio, allocator in self.slots.items():
            allocator.release(self._row_slots(ratio, state.row, length))
        self._clear_row(state.row)
        self.free_live_rows.append(state.row)
