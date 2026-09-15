"""CPU admission metadata contracts; these do not validate numerical kernels."""

import pytest

from sparsevllm.engine.cache_manager.methods.deepseek_v4 import (
    CompressionChunk,
    CompressionPlan,
    CompressionStateShape,
)


@pytest.mark.parametrize("ratio", [4, 128])
def test_arbitrary_chunks_emit_each_absolute_boundary_once(ratio):
    # Chunk-relative rounding loses/duplicates blocks after nonaligned chunks.
    cuts = [0, 1, ratio - 1, ratio + 1, 2 * ratio + 3, 5 * ratio + 1]
    boundaries = []
    for start, end in zip(cuts, cuts[1:]):
        plan = CompressionPlan.prefill((CompressionChunk(7, start, end - start),), ratio)
        boundaries.extend(plan.boundary_ends)
    expected = [p + 1 for p in range(cuts[-1]) if (p + 1) % ratio == 0]
    assert boundaries == expected


def test_mixed_requests_use_independent_positions_and_physical_rows():
    # Packed offsets and request rows are different coordinate systems.
    chunks = (CompressionChunk(9, 127, 3), CompressionChunk(2, 0, 129), CompressionChunk(4, 259, 2))
    plan = CompressionPlan.prefill(chunks, 128)
    actual = list(zip(plan.boundary_requests, plan.boundary_ends))
    expected = [(i, p + 1) for i, c in enumerate(chunks)
                for p in range(c.start, c.start + c.length) if (p + 1) % 128 == 0]
    assert actual == expected
    assert plan.request_rows == tuple(c.request_row for c in chunks)
    assert all(b - a == c.length for a, b, c in zip(plan.cu_seqlens, plan.cu_seqlens[1:], chunks))


def test_duplicate_rows_and_invalid_ranges_rejected_before_device_mutation():
    with pytest.raises(ValueError, match="only once"):
        CompressionPlan.prefill((CompressionChunk(0, 0, 1), CompressionChunk(0, 4, 2)), 4)
    with pytest.raises(ValueError, match="nonnegative"):
        CompressionPlan.prefill((CompressionChunk(0, -1, 4),), 4)
    with pytest.raises(ValueError, match="int32"):
        CompressionPlan.prefill((CompressionChunk(0, 2**31 - 1, 1),), 4)


def test_state_accounting_includes_both_projections_and_overlap():
    # Independent scalar enumeration catches forgetting the gate or overlap bank.
    for ratio, dim in ((4, 128), (128, 512)):
        shape = CompressionStateShape(ratio, dim)
        blocks = 2 if ratio == 4 else 1
        scalars = sum(1 for _ in range(2) for _ in range(blocks)
                      for _ in range(ratio) for _ in range(blocks * dim))
        assert shape.row_bytes == scalars * 4
