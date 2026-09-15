"""Gated pooling over packed chunks and request-owned carry state.

The carry is a ring of *unpooled* FP32 projections. Pool before updating the
ring: an early boundary in a long chunk can still need the previous chunk.
Normalization, rotary embedding and QAT rounding follow this primitive.
"""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _pool(
    KV, Gate, Ape, StateKV, StateGate, Rows, Cu, Starts,
    BoundaryRequests, BoundaryEnds, Out, RopePositions,
    D: tl.constexpr, RATIO: tl.constexpr, DECODE: tl.constexpr,
    TILE_D: tl.constexpr,
):
    event = tl.program_id(0)
    if DECODE:
        batch = event
        end = tl.load(Starts + batch) + 1
    else:
        batch = tl.load(BoundaryRequests + event)
        end = tl.load(BoundaryEnds + event)
    row = tl.load(Rows + batch)
    active = row >= 0
    if DECODE:
        active = active & (end % RATIO == 0)
    d = tl.program_id(1) * TILE_D + tl.arange(0, TILE_D)
    if active:
        OVERLAP: tl.constexpr = RATIO == 4
        WIDTH: tl.constexpr = D * (1 + OVERLAP)
        WINDOW: tl.constexpr = RATIO * (1 + OVERLAP)
        p = end - WINDOW + tl.arange(0, WINDOW)
        start = tl.load(Starts + batch)
        packed_start = tl.load(Cu + batch)
        channel = d[None, :]
        if OVERLAP:
            channel = channel + tl.where(p[:, None] >= end - RATIO, D, 0)
        valid = (p[:, None] >= 0) & (d[None, :] < D)
        current = p[:, None] >= start
        src = (packed_start + p[:, None] - start) * WIDTH + channel
        old = (row * WINDOW + p[:, None] % WINDOW) * WIDTH + channel
        kv = tl.load(KV + src, valid & current, other=0)
        kv += tl.load(StateKV + old, valid & ~current, other=0)
        gate = tl.load(Gate + src, valid & current, other=0)
        gate += tl.load(StateGate + old, valid & ~current, other=0)
        ape = tl.load(Ape + (p[:, None] % RATIO) * WIDTH + channel,
                      valid, other=0)
        score = tl.where(valid, gate + ape, -float("inf"))
        m = tl.max(score, axis=0)
        # Padded dimensions have no valid entries.
        weight = libdevice.exp(score - tl.where(d < D, m, 0)[None, :])
        # Native softmax rounds its probabilities before the weighted sum.
        # Moving division after reduction can cross a later MXFP4 midpoint.
        weight = tl.div_rn(weight, tl.sum(weight, axis=0)[None, :])
        value = tl.sum(weight * kv, axis=0)
        tl.store(Out + event * D + d, value, d < D)
    else:
        tl.store(Out + event * D + d, 0, d < D)
    if tl.program_id(1) == 0:
        tl.store(RopePositions + event, tl.where(active, end - RATIO, -1))


@tr.jit
def _snapshot_carry(
    KV, Gate, StateKV, StateGate, Rows, Cu, Starts,
    SourceRequests, DestinationRows, Ends,
    WIDTH: tl.constexpr, WINDOW: tl.constexpr, TILE_D: tl.constexpr,
):
    snapshot = tl.program_id(0)
    destination = tl.load(DestinationRows + snapshot)
    if destination >= 0:
        batch = tl.load(SourceRequests + snapshot)
        row = tl.load(Rows + batch)
        start = tl.load(Starts + batch)
        end = tl.load(Ends + snapshot)
        packed_start = tl.load(Cu + batch)
        ring_slot = tl.program_id(1)
        distance = ((end - 1) % WINDOW + WINDOW - ring_slot) % WINDOW
        position = end - 1 - distance
        d = tl.program_id(2) * TILE_D + tl.arange(0, TILE_D)
        valid = (position >= 0) & (d < WIDTH)
        current = position >= start
        src = (packed_start + position - start) * WIDTH + d
        old = (row * WINDOW + ring_slot) * WIDTH + d
        value = tl.load(KV + src, valid & current, other=0)
        value += tl.load(StateKV + old, valid & ~current, other=0)
        gate = tl.load(Gate + src, valid & current, other=0)
        gate += tl.load(StateGate + old, valid & ~current, other=0)
        dst = (destination * WINDOW + ring_slot) * WIDTH + d
        tl.store(StateKV + dst, value, d < WIDTH)
        tl.store(StateGate + dst, gate, d < WIDTH)


@tr.jit
def _carry(
    KV, Gate, StateKV, StateGate, Rows, Cu, Starts,
    WIDTH: tl.constexpr, WINDOW: tl.constexpr, TILE_D: tl.constexpr,
    DECODE: tl.constexpr,
):
    batch = tl.program_id(0)
    ring_slot = tl.program_id(1)
    row = tl.load(Rows + batch)
    if row >= 0:
        start = tl.load(Starts + batch)
        if DECODE:
            ring_slot = start % WINDOW
        packed_start = tl.load(Cu + batch)
        length = tl.load(Cu + batch + 1) - packed_start
        end = start + length
        # Only the most recent token for each ring slot writes it, even when
        # a prefill chunk spans many complete compression blocks.
        # Triton signed remainder follows C, so keep its dividend nonnegative
        # when the first short chunk has not yet visited every ring slot.
        distance = ((end - 1) % WINDOW + WINDOW - ring_slot) % WINDOW
        position = end - 1 - distance
        if (position >= start) & (position < end) & (length > 0):
            d = tl.program_id(2) * TILE_D + tl.arange(0, TILE_D)
            src = (packed_start + position - start) * WIDTH + d
            dst = (row * WINDOW + ring_slot) * WIDTH + d
            tl.store(StateKV + dst, tl.load(KV + src, d < WIDTH, other=0), d < WIDTH)
            tl.store(StateGate + dst, tl.load(Gate + src, d < WIDTH, other=0), d < WIDTH)


def compress_projected(
    kv: torch.Tensor,
    gate: torch.Tensor,
    ape: torch.Tensor,
    state_kv: torch.Tensor,
    state_gate: torch.Tensor,
    request_rows: torch.Tensor,
    cu_seqlens: torch.Tensor,
    start_positions: torch.Tensor,
    boundary_requests: torch.Tensor,
    boundary_ends: torch.Tensor,
    out: torch.Tensor,
    rope_positions: torch.Tensor,
    *,
    ratio: int,
    decode: bool = False,
    snapshots=None,
) -> None:
    """Pool then commit carry, with no allocation or device-to-host reads.

    All tensors are contiguous. KV/gate and carry are FP32; output is BF16.
    Metadata is int32. Each active row occurs once in a packed batch; starts
    are absolute token positions. Prefill boundaries are (batch index,
    exclusive absolute block end), built by the cache owner. Decode has one
    input and output row per batch row, and computes the boundary on device.
    An output with rope_positions=-1 is inactive and must not enter the cache.
    Padded request rows are -1; they cannot mutate or read persistent state.
    """
    if ratio not in (4, 128):
        raise ValueError(f"DeepSeek V4 compression requires ratio 4 or 128, got {ratio}.")
    if out.ndim != 2 or out.shape[1] <= 0:
        raise ValueError("Compression output must be [events, positive head_dim].")
    d = out.shape[1]
    width, window = d * (1 + (ratio == 4)), ratio * (1 + (ratio == 4))
    batch = request_rows.numel()
    if kv.ndim != 2 or kv.shape[1] != width or gate.shape != kv.shape:
        raise ValueError("Compression projections must be [tokens, (1+overlap)*head_dim].")
    if state_kv.ndim != 3 or state_kv.shape[1:] != (window, width) or state_gate.shape != state_kv.shape:
        raise ValueError("Compression carry must be [requests, window, projection_dim].")
    if ape.shape != (ratio, width):
        raise ValueError("Compression APE must match ratio and projection width.")
    if cu_seqlens.shape != (batch + 1,) or start_positions.shape != (batch,):
        raise ValueError("Compression batch metadata has inconsistent capacity.")
    events = batch if decode else boundary_requests.numel()
    if out.shape != (events, d) or rope_positions.shape != (events,):
        raise ValueError("Compression output capacity must match boundary metadata.")
    if not decode and boundary_ends.shape != boundary_requests.shape:
        raise ValueError("Compression boundaries require request indices and ends.")
    if decode and kv.shape[0] != batch:
        raise ValueError("Compression decode requires one token per padded batch row.")
    floats = (kv, gate, ape, state_kv, state_gate)
    ints = (request_rows, cu_seqlens, start_positions, boundary_requests, boundary_ends, rope_positions)
    tensors = (*floats, *ints, out)
    if any(t.dtype != torch.float32 for t in floats) or out.dtype != torch.bfloat16:
        raise TypeError("Compression requires FP32 projections/state/APE and BF16 output.")
    if any(t.dtype != torch.int32 for t in ints):
        raise TypeError("Compression metadata must be int32.")
    if any(t.device != kv.device or not t.is_contiguous() for t in tensors):
        raise ValueError("Compression tensors must be contiguous and on one device.")
    if snapshots is not None:
        metadata = (snapshots.source_requests, snapshots.destination_rows, snapshots.ends)
        if any(t.ndim != 1 or t.shape != metadata[0].shape or t.dtype != torch.int32
               or t.device != kv.device or not t.is_contiguous() for t in metadata):
            raise ValueError("Compression snapshots require equally sized contiguous int32 metadata on the cache device.")
        if metadata[0].numel():
            # Source carry is still the previous chunk here. Snapshot targets
            # are disjoint from active rows, so concurrent snapshots cannot
            # overwrite each other's source state.
            _snapshot_carry[(metadata[0].numel(), window, tr.cdiv(width, 128))](
                kv, gate, state_kv, state_gate, request_rows, cu_seqlens,
                start_positions, *metadata, width, window, 128,
            )
    if events:
        _pool[(events, tr.cdiv(d, 32))](
            kv, gate, ape, state_kv, state_gate, request_rows, cu_seqlens,
            start_positions, boundary_requests, boundary_ends, out, rope_positions,
            d, ratio, decode, 32, num_warps=4, enable_fp_fusion=False,
        )
    if batch:
        _carry[(batch, 1 if decode else window, tr.cdiv(width, 128))](
            kv, gate, state_kv, state_gate, request_rows, cu_seqlens,
            start_positions, width, window, 128, decode,
        )
