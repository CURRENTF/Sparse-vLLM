from __future__ import annotations

from dataclasses import dataclass
import os

import torch


@dataclass
class _RawKVEntry:
    k: torch.Tensor | None
    v: torch.Tensor | None
    filled_until: int = 0
    capacity: int = 0
    k_shape_tail: tuple[int, ...] = ()
    v_shape_tail: tuple[int, ...] = ()
    dtype: torch.dtype | None = None
    chunks: dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None


class RawKVOffloadBuffer:
    """CPU backing store for chunked prefill raw KV.

    The buffer is intentionally shape-explicit: callers allocate one row/layer/kind
    with a known total prompt length, then write and restore contiguous ranges.
    """

    def __init__(self, *, pin_memory: bool = True, mode: str | None = None):
        self.pin_memory = bool(pin_memory)
        raw_mode = mode if mode is not None else os.getenv("SPARSEVLLM_RAWKV_BUFFER_MODE", "chunked")
        self.mode = str(raw_mode).strip().lower().replace("-", "_")
        if self.mode not in {"contiguous", "chunked"}:
            raise ValueError(
                "SPARSEVLLM_RAWKV_BUFFER_MODE must be 'contiguous' or 'chunked', "
                f"got {raw_mode!r}."
            )
        self._entries: dict[tuple[int, int, str], _RawKVEntry] = {}

    def ensure_entry(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        kind: str,
        total_len: int,
        k_shape_tail: tuple[int, ...],
        v_shape_tail: tuple[int, ...],
        dtype: torch.dtype,
    ) -> None:
        key = (int(layer_idx), int(row_idx), str(kind))
        total_len = int(total_len)
        if total_len < 0:
            raise ValueError(f"RawKVOffloadBuffer total_len must be non-negative, got {total_len}.")
        entry = self._entries.get(key)
        if entry is not None:
            if int(entry.capacity) < total_len:
                raise RuntimeError(
                    "RawKVOffloadBuffer cannot grow an existing entry: "
                    f"key={key} existing={int(entry.capacity)} requested={total_len}."
                )
            if tuple(entry.k_shape_tail) != tuple(k_shape_tail) or tuple(entry.v_shape_tail) != tuple(v_shape_tail):
                raise RuntimeError(
                    "RawKVOffloadBuffer existing entry shape tail mismatch: "
                    f"key={key} existing_k_tail={entry.k_shape_tail} requested_k_tail={k_shape_tail} "
                    f"existing_v_tail={entry.v_shape_tail} requested_v_tail={v_shape_tail}."
                )
            return
        if self.mode == "chunked":
            self._entries[key] = _RawKVEntry(
                k=None,
                v=None,
                capacity=total_len,
                k_shape_tail=tuple(k_shape_tail),
                v_shape_tail=tuple(v_shape_tail),
                dtype=dtype,
                chunks={},
            )
        else:
            self._entries[key] = _RawKVEntry(
                k=torch.empty(
                    (total_len, *k_shape_tail),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=self.pin_memory,
                ),
                v=torch.empty(
                    (total_len, *v_shape_tail),
                    dtype=dtype,
                    device="cpu",
                    pin_memory=self.pin_memory,
                ),
                capacity=total_len,
                k_shape_tail=tuple(k_shape_tail),
                v_shape_tail=tuple(v_shape_tail),
                dtype=dtype,
            )

    @torch.no_grad()
    def put_range(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        kind: str,
        start: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> None:
        key = (int(layer_idx), int(row_idx), str(kind))
        entry = self._entries.get(key)
        if entry is None:
            raise RuntimeError(f"RawKVOffloadBuffer entry is missing for key={key}.")
        start = int(start)
        end = start + int(k.shape[0])
        if start < 0 or end > int(entry.capacity) or int(v.shape[0]) != int(k.shape[0]):
            raise RuntimeError(
                "RawKVOffloadBuffer put_range shape mismatch: "
                f"key={key} start={start} end={end} capacity={int(entry.capacity)} "
                f"k={tuple(k.shape)} v={tuple(v.shape)}."
            )
        if tuple(k.shape[1:]) != tuple(entry.k_shape_tail) or tuple(v.shape[1:]) != tuple(entry.v_shape_tail):
            raise RuntimeError(
                "RawKVOffloadBuffer put_range shape tail mismatch: "
                f"key={key} k={tuple(k.shape)} expected_tail={entry.k_shape_tail} "
                f"v={tuple(v.shape)} expected_tail={entry.v_shape_tail}."
            )
        if entry.chunks is not None:
            if start > int(entry.filled_until):
                raise RuntimeError(
                    "RawKVOffloadBuffer chunked put_range cannot leave a gap: "
                    f"key={key} start={start} filled_until={int(entry.filled_until)}."
                )
            k_cpu = torch.empty(tuple(k.shape), dtype=entry.dtype, device="cpu", pin_memory=self.pin_memory)
            v_cpu = torch.empty(tuple(v.shape), dtype=entry.dtype, device="cpu", pin_memory=self.pin_memory)
            k_cpu.copy_(k.detach().to(dtype=entry.dtype), non_blocking=True)
            v_cpu.copy_(v.detach().to(dtype=entry.dtype), non_blocking=True)
            entry.chunks[start] = (k_cpu, v_cpu)
        else:
            if entry.k is None or entry.v is None:
                raise RuntimeError(f"RawKVOffloadBuffer contiguous entry is missing tensors for key={key}.")
            entry.k[start:end].copy_(k.detach().to(device="cpu", dtype=entry.k.dtype), non_blocking=True)
            entry.v[start:end].copy_(v.detach().to(device="cpu", dtype=entry.v.dtype), non_blocking=True)
        entry.filled_until = max(int(entry.filled_until), end)

    @torch.no_grad()
    def restore_prefix(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        kind: str,
        end: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(layer_idx), int(row_idx), str(kind))
        entry = self._entries.get(key)
        if entry is None:
            raise RuntimeError(f"RawKVOffloadBuffer entry is missing for key={key}.")
        end = int(end)
        if end < 0 or end > int(entry.filled_until):
            raise RuntimeError(
                "RawKVOffloadBuffer restore_prefix reads an unwritten range: "
                f"key={key} end={end} filled_until={int(entry.filled_until)}."
            )
        if entry.chunks is not None:
            k_out = torch.empty((end, *entry.k_shape_tail), device=device, dtype=dtype)
            v_out = torch.empty((end, *entry.v_shape_tail), device=device, dtype=dtype)
            cursor = 0
            for chunk_start in sorted(entry.chunks):
                k_chunk, v_chunk = entry.chunks[chunk_start]
                chunk_start = int(chunk_start)
                chunk_end = chunk_start + int(k_chunk.shape[0])
                if chunk_start > cursor:
                    raise RuntimeError(
                        "RawKVOffloadBuffer chunked restore_prefix found a gap: "
                        f"key={key} cursor={cursor} next_start={chunk_start}."
                    )
                if chunk_end <= cursor:
                    continue
                copy_start = max(cursor, chunk_start)
                copy_end = min(end, chunk_end)
                if copy_start < copy_end:
                    src_start = copy_start - chunk_start
                    src_end = copy_end - chunk_start
                    k_out[copy_start:copy_end].copy_(k_chunk[src_start:src_end], non_blocking=True)
                    v_out[copy_start:copy_end].copy_(v_chunk[src_start:src_end], non_blocking=True)
                    cursor = copy_end
                if cursor >= end:
                    break
            if cursor != end:
                raise RuntimeError(
                    "RawKVOffloadBuffer chunked restore_prefix did not fill requested prefix: "
                    f"key={key} cursor={cursor} end={end}."
                )
            return k_out, v_out
        if entry.k is None or entry.v is None:
            raise RuntimeError(f"RawKVOffloadBuffer contiguous entry is missing tensors for key={key}.")
        return (
            entry.k[:end].to(device=device, dtype=dtype, non_blocking=True),
            entry.v[:end].to(device=device, dtype=dtype, non_blocking=True),
        )

    def get_prefix_cpu(
        self,
        *,
        layer_idx: int,
        row_idx: int,
        kind: str,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(layer_idx), int(row_idx), str(kind))
        entry = self._entries.get(key)
        if entry is None:
            raise RuntimeError(f"RawKVOffloadBuffer entry is missing for key={key}.")
        end = int(end)
        if end < 0 or end > int(entry.filled_until):
            raise RuntimeError(
                "RawKVOffloadBuffer get_prefix_cpu reads an unwritten range: "
                f"key={key} end={end} filled_until={int(entry.filled_until)}."
            )
        if entry.chunks is not None:
            k_out = torch.empty((end, *entry.k_shape_tail), dtype=entry.dtype, device="cpu", pin_memory=self.pin_memory)
            v_out = torch.empty((end, *entry.v_shape_tail), dtype=entry.dtype, device="cpu", pin_memory=self.pin_memory)
            cursor = 0
            for chunk_start in sorted(entry.chunks):
                k_chunk, v_chunk = entry.chunks[chunk_start]
                chunk_start = int(chunk_start)
                chunk_end = chunk_start + int(k_chunk.shape[0])
                if chunk_start > cursor:
                    raise RuntimeError(
                        "RawKVOffloadBuffer chunked get_prefix_cpu found a gap: "
                        f"key={key} cursor={cursor} next_start={chunk_start}."
                    )
                copy_end = min(end, chunk_end)
                if cursor < copy_end:
                    src_start = cursor - chunk_start
                    src_end = copy_end - chunk_start
                    k_out[cursor:copy_end].copy_(k_chunk[src_start:src_end], non_blocking=True)
                    v_out[cursor:copy_end].copy_(v_chunk[src_start:src_end], non_blocking=True)
                    cursor = copy_end
                if cursor >= end:
                    break
            if cursor != end:
                raise RuntimeError(
                    "RawKVOffloadBuffer chunked get_prefix_cpu did not fill requested prefix: "
                    f"key={key} cursor={cursor} end={end}."
                )
            return k_out, v_out
        if entry.k is None or entry.v is None:
            raise RuntimeError(f"RawKVOffloadBuffer contiguous entry is missing tensors for key={key}.")
        return entry.k[:end], entry.v[:end]

    def release_layer(self, *, layer_idx: int, row_idx: int, kind: str | None = None) -> None:
        prefix = (int(layer_idx), int(row_idx))
        for key in list(self._entries):
            if key[:2] == prefix and (kind is None or key[2] == str(kind)):
                del self._entries[key]

    def release_row(self, row_idx: int) -> None:
        row_idx = int(row_idx)
        for key in list(self._entries):
            if key[1] == row_idx:
                del self._entries[key]
