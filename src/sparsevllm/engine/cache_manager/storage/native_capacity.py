"""Byte-exact native pool geometry and bounded compressed-capacity planning."""

from collections import Counter
from dataclasses import dataclass

from ..methods.deepseek_v4 import CompressionStateShape


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
        return {ratio: count * 2 * (self.head_dim + (self.index_dim if ratio == 4 else 0))
                for ratio, count in sorted(Counter(self.ratios).items()) if ratio}

    @property
    def row_bytes(self):
        window = len(self.ratios) * self.window_size * self.head_dim * 2
        carry = sum(CompressionStateShape(ratio, self.head_dim).row_bytes for ratio in self.ratios if ratio)
        carry += self.ratios.count(4) * CompressionStateShape(4, self.index_dim).row_bytes
        tables = sum(4 * max(1, self.max_model_len // ratio) for ratio in self.compressed_slot_bytes)
        return window + carry + tables

    @property
    def temporary_bytes(self):
        return len(self.ratios) * self.prefill_capacity * self.head_dim * 2

    def allocation_bytes(self, live_rows, snapshot_rows, compressed_capacities):
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
