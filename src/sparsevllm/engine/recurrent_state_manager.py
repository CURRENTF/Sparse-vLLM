from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from sparsevllm.config import Config
from sparsevllm.constant import REDUNDANCY_BATCH_SIZE_FACTOR
from sparsevllm.engine.sequence import Sequence


@dataclass
class RecurrentPrefixPayload:
    token_count: int
    layer_states: dict[int, dict[str, torch.Tensor]] = field(default_factory=dict)


@dataclass(frozen=True)
class RecurrentStateSpec:
    """Model-provided schema for one recurrent layer's persistent tensors."""

    name: str
    state_names: tuple[str, ...]

    def __post_init__(self) -> None:
        names = tuple(str(name) for name in self.state_names)
        if not self.name or not names or any(not name for name in names) or len(set(names)) != len(names):
            raise ValueError(
                f"Invalid recurrent state spec name={self.name!r} state_names={names!r}."
            )
        object.__setattr__(self, "state_names", names)


class RecurrentStateManager:
    """Preallocated per-sequence rows for a model-declared recurrent schema."""

    def __init__(
        self,
        config: Config,
        rank: int,
        world_size: int,
        *,
        state_spec: RecurrentStateSpec,
    ):
        self.config = config
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.runtime_layout = config.runtime_layout
        self.state_spec = state_spec
        self.row_capacity = max(
            int(config.max_num_seqs_in_batch) * REDUNDANCY_BATCH_SIZE_FACTOR,
            int(config.max_decoding_seqs),
        )
        if self.row_capacity <= 0:
            raise ValueError(f"Recurrent state row capacity must be positive, got {self.row_capacity}.")
        self.scratch_row = self.row_capacity
        self.seq_id_to_row: dict[int, int] = {}
        self.row_to_seq_id: list[int | None] = [None] * self.row_capacity
        self.free_rows: deque[int] = deque(range(self.row_capacity))
        self.layer_buffers: dict[int, dict[str, torch.Tensor]] = {}
        self.decode_state_indices: dict[int, torch.Tensor] = {}

    def _allocate_row(self, seq_id: int) -> int:
        seq_id = int(seq_id)
        existing = self.seq_id_to_row.get(seq_id)
        if existing is not None:
            return int(existing)
        if not self.free_rows:
            raise RuntimeError(
                "Recurrent state row capacity exhausted: "
                f"capacity={self.row_capacity} active_sequences={len(self.seq_id_to_row)}."
            )
        row = int(self.free_rows.popleft())
        self.seq_id_to_row[seq_id] = row
        self.row_to_seq_id[row] = seq_id
        for buffers in self.layer_buffers.values():
            for buffer in buffers.values():
                buffer[row].zero_()
        return row

    def _ensure_buffer(self, layer_idx: int, name: str, value: torch.Tensor) -> torch.Tensor:
        layer_idx = int(layer_idx)
        name = str(name)
        buffers = self.layer_buffers.setdefault(layer_idx, {})
        buffer = buffers.get(name)
        expected_shape = (self.row_capacity + 1, *value.shape)
        if buffer is None:
            buffer = torch.zeros(expected_shape, dtype=value.dtype, device=value.device)
            buffers[name] = buffer
            return buffer
        if tuple(buffer.shape) != tuple(expected_shape):
            raise RuntimeError(
                "Recurrent state shape changed after row pool allocation: "
                f"layer_idx={layer_idx} name={name!r} "
                f"expected={tuple(buffer.shape[1:])} got={tuple(value.shape)}."
            )
        if buffer.dtype != value.dtype or buffer.device != value.device:
            raise RuntimeError(
                "Recurrent state dtype/device changed after row pool allocation: "
                f"layer_idx={layer_idx} name={name!r} "
                f"expected={buffer.dtype}/{buffer.device} got={value.dtype}/{value.device}."
            )
        return buffer

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool) -> None:
        del is_prefill
        for seq in seqs:
            self._allocate_row(int(seq.seq_id))

    def prepare_decode_static(
        self,
        seqs: list[Sequence],
        *,
        token_batch: int,
        device: torch.device,
    ) -> None:
        self.prepare_step(seqs, is_prefill=False)
        token_batch = int(token_batch)
        if token_batch < len(seqs):
            raise RuntimeError(
                f"{self.state_spec.name} static decode has fewer token rows than real sequences: "
                f"token_batch={token_batch} real_batch={len(seqs)}."
            )
        rows = [self.seq_id_to_row[int(seq.seq_id)] for seq in seqs]
        rows.extend([self.scratch_row] * (token_batch - len(seqs)))
        state_indices = self.decode_state_indices.get(token_batch)
        if state_indices is None:
            state_indices = torch.empty(token_batch, dtype=torch.int32, device=device)
            self.decode_state_indices[token_batch] = state_indices
        elif state_indices.device != device:
            raise RuntimeError(
                f"{self.state_spec.name} decode state-index buffer changed device: "
                f"expected={state_indices.device} got={device}."
            )
        state_indices.copy_(torch.tensor(rows, dtype=torch.int32, device=device))

    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool) -> None:
        del seqs, is_prefill

    def free_seq(self, seq_id: int) -> None:
        seq_id = int(seq_id)
        row = self.seq_id_to_row.pop(seq_id, None)
        if row is None:
            return
        if self.row_to_seq_id[row] != seq_id:
            raise RuntimeError(
                "Recurrent state row ownership is inconsistent: "
                f"seq_id={seq_id} row={row} owner={self.row_to_seq_id[row]}."
            )
        self.row_to_seq_id[row] = None
        self.free_rows.append(int(row))

    def reset_after_warmup(self) -> None:
        self.seq_id_to_row.clear()
        self.row_to_seq_id = [None] * self.row_capacity
        self.free_rows = deque(range(self.row_capacity))
        for buffers in self.layer_buffers.values():
            for buffer in buffers.values():
                buffer.zero_()

    def get_layer_state(self, seq_id: int, layer_idx: int) -> dict[str, torch.Tensor] | None:
        row = self.seq_id_to_row.get(int(seq_id))
        buffers = self.layer_buffers.get(int(layer_idx))
        if row is None or not buffers:
            return None
        return {name: buffer[row] for name, buffer in buffers.items()}

    def set_layer_state(
        self,
        seq_id: int,
        layer_idx: int,
        states: dict[str, torch.Tensor],
    ) -> None:
        layer_idx = int(layer_idx)
        if self.runtime_layout is not None and not self.runtime_layout.is_linear_attention(layer_idx):
            raise RuntimeError(f"layer_idx={layer_idx} is full_attention and has no recurrent state")
        missing = [name for name in self.state_spec.state_names if name not in states]
        unexpected = [name for name in states if name not in self.state_spec.state_names]
        if missing or unexpected:
            raise RuntimeError(
                f"{self.state_spec.name} recurrent state schema mismatch: "
                f"missing={missing} unexpected={unexpected}."
            )
        for name, value in states.items():
            if not torch.is_tensor(value):
                raise TypeError(
                    f"Recurrent state {name!r} must be a tensor, got {type(value).__name__}."
                )
        row = self._allocate_row(int(seq_id))
        for name, value in states.items():
            buffer = self._ensure_buffer(layer_idx, str(name), value)
            buffer[row].copy_(value.detach())

    def get_decode_layer_state(
        self,
        seqs: list[Sequence],
        *,
        layer_idx: int,
        token_batch: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        layer_idx = int(layer_idx)
        token_batch = int(token_batch)
        real_batch = len(seqs)
        if token_batch < real_batch:
            raise RuntimeError(
                f"{self.state_spec.name} decode has fewer token rows than real sequences: "
                f"token_batch={token_batch} real_batch={real_batch}."
            )
        buffers = self.layer_buffers.get(layer_idx)
        missing = [name for name in self.state_spec.state_names if not buffers or name not in buffers]
        if missing:
            raise RuntimeError(
                f"{self.state_spec.name} decode requires initialized recurrent state for "
                f"layer_idx={layer_idx}; missing={missing}."
            )
        state_buffers = {name: buffers[name] for name in self.state_spec.state_names}
        for name, buffer in state_buffers.items():
            if buffer.dtype != dtype or buffer.device != device:
                raise RuntimeError(
                    f"{self.state_spec.name} decode recurrent pool dtype/device does not match activations: "
                    f"layer_idx={layer_idx} name={name} pool={buffer.dtype}/{buffer.device} "
                    f"expected={dtype}/{device}."
                )
        expected_rows: list[int] = []
        for seq in seqs:
            row = self.seq_id_to_row.get(int(seq.seq_id))
            if row is None:
                raise RuntimeError(
                    f"{self.state_spec.name} decode requires recurrent state for every sequence: "
                    f"missing_seq_id={seq.seq_id}."
                )
            expected_rows.append(int(row))
        for buffer in state_buffers.values():
            buffer[self.scratch_row].zero_()
        state_indices = self.decode_state_indices.get(token_batch)
        if state_indices is None:
            raise RuntimeError(
                f"{self.state_spec.name} decode state-index buffer was not prepared for the static batch: "
                f"token_batch={token_batch}."
            )
        expected_rows.extend([self.scratch_row] * (token_batch - real_batch))
        if state_indices.device.type == "cpu" and state_indices.tolist() != expected_rows:
            raise RuntimeError(
                f"{self.state_spec.name} decode state-index buffer is stale: "
                f"expected={expected_rows} got={state_indices.tolist()}."
            )
        return state_buffers, state_indices

    def build_prefix_recurrent_payload(self, seq: Sequence, token_count: int) -> RecurrentPrefixPayload:
        token_count = int(token_count)
        if token_count <= 0:
            raise ValueError(f"token_count must be > 0, got {token_count}.")
        block_size = int(getattr(self.config, "prefix_cache_block_size", 0) or 0)
        if block_size > 0 and token_count % block_size != 0:
            raise ValueError(
                "Recurrent prefix snapshots are only valid at prefix block boundaries: "
                f"seq_id={seq.seq_id} token_count={token_count} block_size={block_size}."
            )
        row = self.seq_id_to_row.get(int(seq.seq_id))
        if row is None:
            raise RuntimeError(f"Cannot snapshot recurrent state for unknown seq_id={seq.seq_id}.")
        payload_states: dict[int, dict[str, torch.Tensor]] = {}
        for layer_idx, buffers in self.layer_buffers.items():
            if self.runtime_layout is not None and not self.runtime_layout.is_linear_attention(layer_idx):
                continue
            payload_states[layer_idx] = {
                name: buffer[row].detach().clone()
                for name, buffer in buffers.items()
            }
        return RecurrentPrefixPayload(token_count=token_count, layer_states=payload_states)

    def attach_prefix_recurrent_payload(self, seq: Sequence, payload: RecurrentPrefixPayload | None) -> None:
        if payload is None:
            return
        if int(payload.token_count) <= 0:
            raise RuntimeError(f"Invalid recurrent prefix payload token_count={payload.token_count}.")
        block_size = int(getattr(self.config, "prefix_cache_block_size", 0) or 0)
        if block_size > 0 and int(payload.token_count) % block_size != 0:
            raise RuntimeError(
                "Cannot attach recurrent prefix payload from a non-boundary snapshot: "
                f"seq_id={seq.seq_id} token_count={payload.token_count} block_size={block_size}."
            )
        for layer_idx, state in payload.layer_states.items():
            self.set_layer_state(int(seq.seq_id), int(layer_idx), state)

    def free_prefix_recurrent_payload(self, payload: RecurrentPrefixPayload | None) -> None:
        del payload

    def prefix_recurrent_payload_nbytes(self, payload: RecurrentPrefixPayload | None) -> int:
        if payload is None:
            return 0
        total = 0
        for state in payload.layer_states.values():
            for tensor in state.values():
                if torch.is_tensor(tensor):
                    total += int(tensor.numel() * tensor.element_size())
        return int(total)
