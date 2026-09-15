"""Typed physical views for native attention cache representations."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .storage.native_layer import NativeAttentionBatch, NativeAttentionLayerStorage


@dataclass(frozen=True)
class NativeAttentionExecution:
    storage: "NativeAttentionLayerStorage"
    batch: "NativeAttentionBatch"


@dataclass(frozen=True)
class IndexedSharedKVView:
    """One vector serves as both key and value; indices are physical slots.

    kv: [slots, 1, head_dim]. indices: [queries, 1, selection_capacity],
    with -1 padding. The cache manager owns both tensors and all slot lifetimes.
    Causality and request isolation are encoded by the selected physical slots.
    """

    kv: torch.Tensor
    indices: torch.Tensor


@dataclass(frozen=True)
class PackedSharedKVPayload:
    """Opaque DSv4 FP8 pages in the FlashMLA view, with logical slot capacity.

    cache is uint8 [pages, 64, 1, 584]. Its page stride includes padding;
    data and scales occupy separate regions within each page. It must not be
    flattened or treated as independently contiguous 584-byte token rows.
    """

    cache: torch.Tensor
    slot_capacity: int


@dataclass(frozen=True)
class IndexedPackedSharedKVView:
    """Decode reads packed physical pages directly with token-slot indices."""

    payload: PackedSharedKVPayload
    indices: torch.Tensor


@dataclass(frozen=True)
class NativeStateSnapshots:
    """Copy exact prefix-end state into cache-owned, disjoint snapshot rows.

    source_requests indexes the packed request batch; ends are exclusive
    absolute positions within those chunks. Negative destination rows are
    inactive graph slots. The cache owner validates disjoint row ownership.
    """

    source_requests: torch.Tensor
    destination_rows: torch.Tensor
    ends: torch.Tensor


@dataclass(frozen=True)
class CompressionBatchView:
    """Cache-owned raw carry and absolute packed-chunk coordinates.

    Request rows are unique among active chunks. Decode uses one token per
    row and negative row IDs for graph padding. Prefill boundary requests
    index the packed batch, while boundary ends are exclusive token positions.
    """

    state_kv: torch.Tensor
    state_gate: torch.Tensor
    request_rows: torch.Tensor
    cu_seqlens: torch.Tensor
    start_positions: torch.Tensor
    boundary_requests: torch.Tensor
    boundary_ends: torch.Tensor
    decode: bool = False
    snapshots: NativeStateSnapshots | None = None


@dataclass(frozen=True)
class CompressedIndexView:
    """Cache-owned index keys and logical compressed-position mapping.

    keys: BF16 [slots, head_dim]. slots: int32 [request_rows, capacity].
    query_rows and visible_lengths are int32 token vectors. Each visible prefix
    is allocated, contains no holes, and includes only completed causal blocks.
    Padding uses row -1 and length 0. Selected slots address this compressed
    pool; the cache owner translates them into the attention pool's region.
    """

    keys: torch.Tensor
    slots: torch.Tensor
    query_rows: torch.Tensor
    visible_lengths: torch.Tensor


@dataclass(frozen=True)
class SharedKVWindowBatch:
    """Physical window coordinates per packed query token.

    Decode uses only rows/positions after writing the current token to its ring.
    Prefill also supplies absolute chunk starts/ends and packed chunk offsets;
    new tokens live in a temporary region until all chunk queries have attended.
    Negative request rows and positions denote graph padding.
    """

    request_rows: torch.Tensor
    positions: torch.Tensor
    chunk_starts: torch.Tensor | None = None
    chunk_ends: torch.Tensor | None = None
    packed_starts: torch.Tensor | None = None
