"""Physical compression metadata for the native DeepSeek V4 cache.

These CPU plans are prepared before execution/capture. The eventual cache
manager owns their device buffers and all compressed/window/state pools.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CompressionChunk:
    request_row: int
    start: int
    length: int


@dataclass(frozen=True)
class StateSnapshot:
    request_row: int
    end: int
    snapshot_row: int


@dataclass(frozen=True)
class CompressionPlan:
    request_rows: tuple[int, ...]
    cu_seqlens: tuple[int, ...]
    start_positions: tuple[int, ...]
    boundary_requests: tuple[int, ...]
    boundary_ends: tuple[int, ...]

    @classmethod
    def prefill(cls, chunks: tuple[CompressionChunk, ...], ratio: int) -> "CompressionPlan":
        if ratio not in (4, 128):
            raise ValueError(f"DeepSeek V4 compression requires ratio 4 or 128, got {ratio}.")
        rows, cu, starts, requests, ends = [], [0], [], [], []
        seen = set()
        for batch, chunk in enumerate(chunks):
            if chunk.request_row < 0 or chunk.start < 0 or chunk.length <= 0:
                raise ValueError("Prefill chunks require a valid row, nonnegative start and positive length.")
            if chunk.start + chunk.length >= 2**31 or chunk.request_row >= 2**31 or cu[-1] + chunk.length >= 2**31:
                raise ValueError("Compression metadata exceeds int32 indexing capacity.")
            if chunk.request_row in seen:
                raise ValueError("A request may occur only once in a packed compression batch.")
            seen.add(chunk.request_row)
            rows.append(chunk.request_row)
            starts.append(chunk.start)
            cu.append(cu[-1] + chunk.length)
            # Absolute boundaries, never chunk-relative multiples of ratio.
            first_end = (chunk.start // ratio + 1) * ratio
            chunk_ends = range(first_end, chunk.start + chunk.length + 1, ratio)
            ends.extend(chunk_ends)
            requests.extend([batch] * len(chunk_ends))
        return cls(tuple(rows), tuple(cu), tuple(starts), tuple(requests), tuple(ends))


@dataclass(frozen=True)
class CompressionStateShape:
    ratio: int
    head_dim: int

    def __post_init__(self):
        if self.ratio not in (4, 128) or self.head_dim <= 0:
            raise ValueError("Compression state requires ratio 4 or 128 and a positive head dimension.")

    @property
    def row_shape(self) -> tuple[int, int]:
        overlap = self.ratio == 4
        return (self.ratio * (1 + overlap), self.head_dim * (1 + overlap))

    @property
    def row_bytes(self) -> int:
        # Both unpooled values and gates remain FP32 across chunk boundaries.
        window, width = self.row_shape
        return 2 * window * width * 4
