"""Byte-exact native pool geometry and bounded compressed-capacity planning."""

from collections import Counter
from dataclasses import dataclass

from ..methods.deepseek_v4 import CompressionStateShape
from .packed_shared_kv import HEAD_DIM, PAGE_BYTES, PAGE_SIZE, packed_shared_kv_bytes


@dataclass(frozen=True)
class NativePoolCapacity:
    compressed_capacities: dict[int, int]
    allocated_bytes: int
    unused_bytes: int


@dataclass(frozen=True)
class NativeCacheGeometry:
    ratios: tuple[int, ...]
    max_model_len: int
    prefill_capacity: int
    head_dim: int = 512
    index_dim: int = 128
    window_size: int = 128
    packed_kv: bool = False

    def __post_init__(self):
        if self.packed_kv and self.head_dim != HEAD_DIM:
            raise ValueError("Packed native cache requires head dimension 512.")

    @property
    def kv_slot_bytes(self):
        # A lower bound for packed storage; allocation_bytes rounds each layer
        # to whole physical pages before admission.
        return PAGE_BYTES // PAGE_SIZE if self.packed_kv else self.head_dim * 2

    @classmethod
    def from_config(cls, config):
        hf = config.hf_config
        return cls(tuple(hf.compress_ratios[:hf.num_hidden_layers]), config.max_model_len,
                   config.max_num_batched_tokens, hf.head_dim, hf.index_head_dim, hf.sliding_window)

    def budget_for_tokens(self, num_tokens, *, live_rows, snapshot_rows=0):
        """Include per-request rounding and the planner's coarse-slot allowance."""
        rows = live_rows + snapshot_rows
        sizes = self.compressed_slot_bytes
        if not sizes:
            return self.allocation_bytes(live_rows, snapshot_rows, {})
        primary = min(sizes)
        fine = min((num_tokens + primary - 1) // primary + rows,
                   rows * max(1, self.max_model_len // primary))
        capacities = {ratio: fine if ratio == primary else min(
            (fine * primary + ratio - 1) // ratio + rows,
            rows * max(1, self.max_model_len // ratio),
        ) for ratio in sizes}
        return self.allocation_bytes(live_rows, snapshot_rows, capacities)

    @property
    def compressed_slot_bytes(self):
        return {ratio: count * (self.kv_slot_bytes + (2 * self.index_dim if ratio == 4 else 0))
                for ratio, count in sorted(Counter(self.ratios).items()) if ratio}

    @property
    def row_bytes(self):
        """Linear row cost, a lower bound when physical pages need rounding."""
        window = len(self.ratios) * self.window_size * self.kv_slot_bytes
        return window + self.state_row_bytes

    @property
    def state_row_bytes(self):
        carry = sum(CompressionStateShape(ratio, self.head_dim).row_bytes for ratio in self.ratios if ratio)
        carry += self.ratios.count(4) * CompressionStateShape(4, self.index_dim).row_bytes
        tables = sum(4 * max(1, self.max_model_len // ratio) for ratio in self.compressed_slot_bytes)
        return carry + tables

    @property
    def temporary_bytes(self):
        return len(self.ratios) * self.prefill_capacity * self.kv_slot_bytes

    def gather_page_capacities(self, live_rows, snapshot_rows, compressed_capacities):
        """Per-ratio page domains sharing one largest-domain BF16 gather buffer."""
        if not self.packed_kv:
            return {}
        window_slots = (live_rows + snapshot_rows) * self.window_size
        return {ratio: (window_slots + compressed_capacities.get(ratio, 0)
                        + self.prefill_capacity + PAGE_SIZE - 1) // PAGE_SIZE
                for ratio in sorted(set(self.ratios))}

    def allocation_bytes(self, live_rows, snapshot_rows, compressed_capacities):
        if self.packed_kv:
            rows = live_rows + snapshot_rows
            main = sum(packed_shared_kv_bytes(rows * self.window_size + self.prefill_capacity
                                             + compressed_capacities.get(ratio, 0))
                       for ratio in self.ratios)
            index = self.ratios.count(4) * compressed_capacities.get(4, 0) * self.index_dim * 2
            pages = self.gather_page_capacities(live_rows, snapshot_rows, compressed_capacities)
            # One shared materialization buffer; each ratio owns a page table,
            # physical-to-gather map, and gather length. Storage owns FP32 RoPE.
            gather = max(pages.values(), default=0) * PAGE_SIZE * self.head_dim * 2
            metadata = sum(2 * count * 4 + 4 for count in pages.values())
            rotary = len(self.ratios) * 64 * 4
            return rows * self.state_row_bytes + main + index + gather + metadata + rotary
        return ((live_rows + snapshot_rows) * self.row_bytes + self.temporary_bytes
                + sum(compressed_capacities[ratio] * size for ratio, size in self.compressed_slot_bytes.items()))

    def capacity_for_budget(self, available_bytes, *, live_rows, snapshot_rows):
        if available_bytes <= 0 or live_rows <= 0 or snapshot_rows < 0:
            raise ValueError("Native cache planning requires a positive byte/live-row budget and nonnegative snapshot rows.")
        sizes = self.compressed_slot_bytes
        rows = live_rows + snapshot_rows
        if not sizes:
            used = self.allocation_bytes(live_rows, snapshot_rows, {})
            if used > available_bytes:
                raise MemoryError("Native window/state storage exceeds its byte budget.")
            return NativePoolCapacity({}, used, available_bytes-used)
        primary = min(sizes)

        def capacities(count):
            # A shared-prefix branch can complete a coarse block while adding
            # only one fine block. One coarse-slot allowance per live/snapshot
            # endpoint bounds that fragmentation; exact admission still uses
            # the pool's per-ratio slot counts.
            return {ratio: count if ratio == primary else min(
                (count * primary + ratio - 1) // ratio + rows,
                rows * max(1, self.max_model_len // ratio),
            ) for ratio in sizes}

        lower, upper = 0, min(available_bytes // sizes[primary], rows * max(1, self.max_model_len // primary))
        while lower < upper:
            candidate = (lower + upper + 1) // 2
            if self.allocation_bytes(live_rows, snapshot_rows, capacities(candidate)) <= available_bytes:
                lower = candidate
            else:
                upper = candidate - 1
        if lower == 0:
            raise MemoryError("Native window/carry/table storage and one compressed slot exceed the byte budget.")
        selected = capacities(lower)
        used = self.allocation_bytes(live_rows, snapshot_rows, selected)
        return NativePoolCapacity(selected, used, available_bytes-used)
