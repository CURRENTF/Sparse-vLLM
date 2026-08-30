from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _prepare_quest_decode_geometry_kernel(
    context_lens,
    num_pages,
    previous_page_counts,
    NUM_ROWS: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < NUM_ROWS
    context_len = tl.load(context_lens + offsets, mask=mask)
    row_num_pages = (context_len + PAGE_SIZE - 1) // PAGE_SIZE
    tl.store(num_pages + offsets, row_num_pages, mask=mask)
    tl.store(
        previous_page_counts + offsets,
        tl.maximum(row_num_pages - 1, 0),
        mask=mask,
    )


def prepare_quest_decode_geometry(
    context_lens: torch.Tensor,
    *,
    page_size: int,
    num_pages: torch.Tensor,
    previous_page_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare layer-invariant QuEST page counts into stable graph buffers."""

    num_rows = int(context_lens.numel())
    if (
        context_lens.ndim != 1
        or context_lens.dtype != torch.int32
        or not context_lens.is_contiguous()
        or not context_lens.is_cuda
    ):
        raise TypeError("QuEST decode context lengths must be contiguous CUDA int32.")
    for name, output in {
        "num_pages": num_pages,
        "previous_page_counts": previous_page_counts,
    }.items():
        if (
            output.shape != context_lens.shape
            or output.dtype != torch.int32
            or not output.is_contiguous()
            or output.device != context_lens.device
        ):
            raise TypeError(
                f"QuEST {name} must be contiguous int32 and match context_lens."
            )
    if num_rows <= 0:
        raise ValueError("QuEST decode geometry requires a non-empty batch.")
    if int(page_size) <= 0:
        raise ValueError("QuEST decode geometry requires a positive page size.")

    block_size = triton.next_power_of_2(num_rows)
    _prepare_quest_decode_geometry_kernel[(1,)](
        context_lens,
        num_pages,
        previous_page_counts,
        NUM_ROWS=num_rows,
        PAGE_SIZE=int(page_size),
        BLOCK_SIZE=block_size,
        num_warps=1,
        num_stages=1,
    )
    return num_pages, previous_page_counts


@triton.jit
def _score_quest_pages_kernel(
    query,
    page_max,
    page_min,
    row_page_slots,
    output,
    query_stride_row,
    query_stride_head,
    metadata_stride_page,
    metadata_stride_head,
    page_table_stride,
    output_stride,
    NUM_QUERY_HEADS: tl.constexpr,
    NUM_METADATA_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    logical_page = tl.program_id(1)
    physical_page = tl.maximum(
        tl.load(row_page_slots + row * page_table_stride + logical_page),
        0,
    )
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    group_size: tl.constexpr = NUM_QUERY_HEADS // NUM_METADATA_HEADS
    best_score = -float("inf")
    for query_head in tl.static_range(0, NUM_QUERY_HEADS):
        query_values = tl.load(
            query
            + row * query_stride_row
            + query_head * query_stride_head
            + dim_offsets,
            mask=dim_mask,
            other=0.0,
        )
        metadata_offsets = (
            physical_page * metadata_stride_page
            + (query_head // group_size) * metadata_stride_head
            + dim_offsets
        )
        max_values = tl.load(
            page_max + metadata_offsets,
            mask=dim_mask,
            other=0.0,
        )
        min_values = tl.load(
            page_min + metadata_offsets,
            mask=dim_mask,
            other=0.0,
        )
        positive_bound = tl.sum(
            tl.where(
                query_values >= 0,
                query_values.to(tl.float32) * max_values.to(tl.float32),
                0.0,
            ),
            axis=0,
        ).to(query_values.dtype)
        negative_bound = tl.sum(
            tl.where(
                query_values < 0,
                query_values.to(tl.float32) * min_values.to(tl.float32),
                0.0,
            ),
            axis=0,
        ).to(query_values.dtype)
        bound = (positive_bound + negative_bound).to(query_values.dtype)
        best_score = tl.maximum(best_score, bound)
    tl.store(output + row * output_stride + logical_page, best_score)


def score_quest_pages(
    query: torch.Tensor,
    page_max: torch.Tensor,
    page_min: torch.Tensor,
    row_page_slots: torch.Tensor,
) -> torch.Tensor:
    """Score logical pages directly from physical QuEST metadata."""

    if query.ndim != 3 or not query.is_contiguous():
        raise ValueError("QuEST score query must be contiguous and rank 3")
    if page_max.shape != page_min.shape or page_max.ndim != 3:
        raise ValueError("QuEST max/min metadata must have matching rank-3 shapes")
    if not page_max.is_contiguous() or not page_min.is_contiguous():
        raise ValueError("QuEST page metadata must be contiguous")
    batch_size, num_query_heads, head_dim = map(int, query.shape)
    _, num_metadata_heads, metadata_head_dim = map(int, page_max.shape)
    if metadata_head_dim != head_dim or num_query_heads % num_metadata_heads:
        raise ValueError(
            "QuEST query heads/dim must be divisible by metadata heads and "
            "share the metadata dimension"
        )
    if (
        row_page_slots.ndim != 2
        or int(row_page_slots.shape[0]) != batch_size
        or row_page_slots.dtype != torch.int32
        or not row_page_slots.is_contiguous()
    ):
        raise ValueError(
            "QuEST row page slots must be contiguous int32 [batch, pages]"
        )
    tensors = (page_max, page_min, row_page_slots)
    if any(tensor.device != query.device for tensor in tensors):
        raise ValueError("QuEST scoring inputs must share one device")
    if page_max.dtype != query.dtype or page_min.dtype != query.dtype:
        raise TypeError("QuEST score query and metadata must share a dtype")
    if query.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(
            "Fused QuEST page scoring requires FP16, BF16, or FP32 tensors"
        )
    if not query.is_cuda:
        raise ValueError("Fused QuEST page scoring requires CUDA tensors")
    if head_dim > 65536:
        raise ValueError(f"QuEST metadata head_dim is too large: {head_dim}")

    num_logical_pages = int(row_page_slots.shape[1])
    output = torch.empty(
        (batch_size, num_logical_pages),
        dtype=query.dtype,
        device=query.device,
    )
    block_d = triton.next_power_of_2(head_dim)
    _score_quest_pages_kernel[(batch_size, num_logical_pages)](
        query,
        page_max,
        page_min,
        row_page_slots,
        output,
        query.stride(0),
        query.stride(1),
        page_max.stride(0),
        page_max.stride(1),
        row_page_slots.stride(0),
        output.stride(0),
        NUM_QUERY_HEADS=num_query_heads,
        NUM_METADATA_HEADS=num_metadata_heads,
        HEAD_DIM=head_dim,
        BLOCK_D=block_d,
        num_warps=min(max(block_d // 256, 1), 8),
        num_stages=1,
    )
    return output


@triton.jit
def _finalize_quest_decode_view_kernel(
    selected_prev_page_slots,
    row_page_slots,
    num_pages,
    context_lens,
    dense_slots,
    packed_slots,
    local_req_indices,
    local_context_lens,
    selected_stride,
    page_table_stride,
    dense_stride,
    output_stride,
    WIDTH: tl.constexpr,
    PREV_BUDGET: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOKEN_BUDGET: tl.constexpr,
    USE_DENSE_FALLBACK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.program_id(1)
    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    output_mask = offsets < WIDTH

    row_num_pages = tl.maximum(tl.load(num_pages + row), 1)
    context_len = tl.load(context_lens + row)
    sparse_keep = (PREV_BUDGET + 1) * PAGE_SIZE
    page_indices = offsets // PAGE_SIZE
    page_offsets = offsets % PAGE_SIZE
    previous_mask = page_indices < PREV_BUDGET
    previous_indices = tl.minimum(page_indices, PREV_BUDGET - 1)
    previous_page_slots = tl.load(
        selected_prev_page_slots
        + row * selected_stride
        + previous_indices,
        mask=output_mask & previous_mask,
        other=0,
    )
    last_page_slot = tl.load(
        row_page_slots
        + row * page_table_stride
        + row_num_pages
        - 1
    )
    page_slots = tl.where(previous_mask, previous_page_slots, last_page_slot)
    sparse_slots = page_slots * PAGE_SIZE + page_offsets

    last_page_len = context_len - (row_num_pages - 1) * PAGE_SIZE
    sparse_len = PREV_BUDGET * PAGE_SIZE + last_page_len
    if USE_DENSE_FALLBACK:
        use_dense = (context_len <= TOKEN_BUDGET) | (
            row_num_pages <= PREV_BUDGET + 1
        )
        dense_values = tl.load(
            dense_slots + row * dense_stride + offsets,
            mask=output_mask,
            other=0,
        )
        sparse_values = tl.where(offsets < sparse_keep, sparse_slots, dense_values)
        output_values = tl.where(use_dense, dense_values, sparse_values)
        output_len = tl.where(use_dense, context_len, sparse_len)
    else:
        output_values = sparse_slots
        output_len = sparse_len

    tl.store(
        packed_slots + row * output_stride + offsets,
        output_values,
        mask=output_mask,
    )
    if block == 0:
        tl.store(local_req_indices + row, row)
        tl.store(local_context_lens + row, output_len)


def finalize_quest_decode_view(
    selected_prev_page_slots: torch.Tensor,
    row_page_slots: torch.Tensor,
    num_pages: torch.Tensor,
    context_lens: torch.Tensor,
    dense_slots: torch.Tensor | None,
    *,
    page_size: int,
    token_budget: int,
    output_width: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Expand selected pages into token slots in one graph-safe kernel."""

    tensors = {
        "selected_prev_page_slots": selected_prev_page_slots,
        "row_page_slots": row_page_slots,
        "num_pages": num_pages,
        "context_lens": context_lens,
    }
    batch_size = int(selected_prev_page_slots.shape[0])
    if selected_prev_page_slots.ndim != 2:
        raise ValueError("selected_prev_page_slots must be rank 2")
    if row_page_slots.ndim != 2 or int(row_page_slots.shape[0]) != batch_size:
        raise ValueError("row_page_slots must be rank 2 with the same batch size")
    if num_pages.shape != (batch_size,) or context_lens.shape != (batch_size,):
        raise ValueError("num_pages and context_lens must have one value per row")
    for name, tensor in tensors.items():
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must use torch.int32, got {tensor.dtype}")
        if tensor.device != selected_prev_page_slots.device:
            raise ValueError(f"{name} must share one device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    page_size = int(page_size)
    token_budget = int(token_budget)
    output_width = int(output_width)
    prev_budget = int(selected_prev_page_slots.shape[1])
    if page_size <= 0 or token_budget <= 0 or output_width <= 0:
        raise ValueError("page_size, token_budget, and output_width must be positive")
    sparse_width = (prev_budget + 1) * page_size
    use_dense_fallback = dense_slots is not None
    if use_dense_fallback:
        if (
            dense_slots.shape != (batch_size, output_width)
            or dense_slots.dtype != torch.int32
            or dense_slots.device != selected_prev_page_slots.device
            or not dense_slots.is_contiguous()
        ):
            raise ValueError(
                "dense_slots must be contiguous int32 [batch, output_width] "
                "on the selection device"
            )
    elif output_width != sparse_width:
        raise ValueError(
            f"sparse-only output width must be {sparse_width}, got {output_width}"
        )

    if not selected_prev_page_slots.is_cuda:
        safe_num_pages = num_pages.to(torch.long).clamp_min(1)
        last_page_slots = row_page_slots.gather(
            1,
            (safe_num_pages - 1)[:, None],
        )
        selected_page_slots = torch.cat(
            (selected_prev_page_slots, last_page_slots),
            dim=1,
        )
        page_offsets = torch.arange(
            page_size,
            dtype=torch.int32,
            device=selected_prev_page_slots.device,
        )
        sparse_slots = (
            selected_page_slots[:, :, None] * page_size
            + page_offsets[None, None, :]
        ).reshape(batch_size, -1)
        last_page_len = context_lens - (num_pages - 1) * page_size
        sparse_lens = prev_budget * page_size + last_page_len
        if dense_slots is None:
            packed_slots = sparse_slots
            local_context_lens = sparse_lens
        else:
            use_dense = (context_lens <= token_budget) | (
                num_pages <= prev_budget + 1
            )
            packed_slots = dense_slots.clone()
            packed_slots[:, :sparse_width] = torch.where(
                use_dense[:, None],
                dense_slots[:, :sparse_width],
                sparse_slots,
            )
            local_context_lens = torch.where(
                use_dense,
                context_lens,
                sparse_lens,
            )
        local_req_indices = torch.arange(
            batch_size,
            dtype=torch.int32,
            device=selected_prev_page_slots.device,
        )
        return packed_slots, local_req_indices, local_context_lens.to(
            torch.int32
        )

    packed_slots = torch.empty(
        (batch_size, output_width),
        dtype=torch.int32,
        device=selected_prev_page_slots.device,
    )
    local_req_indices = torch.empty(
        (batch_size,),
        dtype=torch.int32,
        device=selected_prev_page_slots.device,
    )
    local_context_lens = torch.empty_like(local_req_indices)
    block_size = 256
    _finalize_quest_decode_view_kernel[
        (batch_size, triton.cdiv(output_width, block_size))
    ](
        selected_prev_page_slots,
        row_page_slots,
        num_pages,
        context_lens,
        dense_slots if dense_slots is not None else row_page_slots,
        packed_slots,
        local_req_indices,
        local_context_lens,
        selected_prev_page_slots.stride(0),
        row_page_slots.stride(0),
        dense_slots.stride(0) if dense_slots is not None else 0,
        packed_slots.stride(0),
        WIDTH=output_width,
        PREV_BUDGET=prev_budget,
        PAGE_SIZE=page_size,
        TOKEN_BUDGET=token_budget,
        USE_DENSE_FALLBACK=use_dense_fallback,
        BLOCK_SIZE=block_size,
        num_warps=4,
        num_stages=1,
    )
    return packed_slots, local_req_indices, local_context_lens


@triton.jit
def _finalize_quest_paged_decode_view_kernel(
    selected_prev_page_slots,
    row_page_slots,
    num_pages,
    context_lens,
    output_page_table,
    output_req_indices,
    output_context_lens,
    output_page_counts,
    output_last_page_lens,
    selected_stride,
    row_page_stride,
    output_stride,
    WIDTH: tl.constexpr,
    PREV_BUDGET: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    TOKEN_BUDGET: tl.constexpr,
    USE_DENSE_FALLBACK: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, BLOCK_SIZE)
    output_mask = offsets < WIDTH
    row_num_pages = tl.maximum(tl.load(num_pages + row), 1)
    context_len = tl.load(context_lens + row)
    last_page_len = context_len - (row_num_pages - 1) * PAGE_SIZE
    use_dense = False
    if USE_DENSE_FALLBACK:
        use_dense = (context_len <= TOKEN_BUDGET) | (
            row_num_pages <= PREV_BUDGET + 1
        )

    previous_indices = tl.minimum(offsets, PREV_BUDGET - 1)
    previous_pages = tl.load(
        selected_prev_page_slots + row * selected_stride + previous_indices,
        mask=output_mask & (offsets < PREV_BUDGET),
        other=0,
    )
    last_page = tl.load(
        row_page_slots + row * row_page_stride + row_num_pages - 1
    )
    sparse_pages = tl.where(offsets < PREV_BUDGET, previous_pages, last_page)
    sparse_valid = offsets < PREV_BUDGET + 1
    if USE_DENSE_FALLBACK:
        dense_pages = tl.load(
            row_page_slots + row * row_page_stride + offsets,
            mask=output_mask & (offsets < row_num_pages),
            other=0,
        )
        output_pages = tl.where(use_dense, dense_pages, sparse_pages)
        valid_pages = tl.where(use_dense, offsets < row_num_pages, sparse_valid)
        output_len = tl.where(
            use_dense,
            context_len,
            PREV_BUDGET * PAGE_SIZE + last_page_len,
        )
        output_page_count = tl.where(
            use_dense, row_num_pages, PREV_BUDGET + 1
        )
    else:
        output_pages = sparse_pages
        valid_pages = sparse_valid
        output_len = PREV_BUDGET * PAGE_SIZE + last_page_len
        output_page_count = PREV_BUDGET + 1

    tl.store(
        output_page_table + row * output_stride + offsets,
        tl.where(valid_pages, output_pages, 0),
        mask=output_mask,
    )
    tl.store(output_req_indices + row, row)
    tl.store(output_context_lens + row, output_len)
    tl.store(output_page_counts + row, output_page_count)
    tl.store(output_last_page_lens + row, last_page_len)


def finalize_quest_paged_decode_view(
    selected_prev_page_slots: torch.Tensor,
    row_page_slots: torch.Tensor,
    num_pages: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    page_size: int,
    token_budget: int,
    output_page_table: torch.Tensor,
    output_req_indices: torch.Tensor,
    output_context_lens: torch.Tensor,
    output_page_counts: torch.Tensor,
    output_last_page_lens: torch.Tensor,
    use_dense_fallback: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Finalize selected QuEST pages directly into a caller-owned paged view."""

    batch_size, prev_budget = map(int, selected_prev_page_slots.shape)
    width = int(output_page_table.shape[1])
    tensors = {
        "selected_prev_page_slots": selected_prev_page_slots,
        "row_page_slots": row_page_slots,
        "num_pages": num_pages,
        "context_lens": context_lens,
        "output_page_table": output_page_table,
        "output_req_indices": output_req_indices,
        "output_context_lens": output_context_lens,
        "output_page_counts": output_page_counts,
        "output_last_page_lens": output_last_page_lens,
    }
    if selected_prev_page_slots.ndim != 2 or row_page_slots.ndim != 2:
        raise ValueError("QuEST paged finalizer requires rank-2 page tables")
    if int(row_page_slots.shape[0]) != batch_size:
        raise ValueError("QuEST paged finalizer page tables must share a batch size")
    if output_page_table.shape != (batch_size, width) or width < prev_budget + 1:
        raise ValueError("QuEST paged output table has insufficient capacity")
    for name, tensor in tensors.items():
        if tensor.dtype != torch.int32:
            raise TypeError(f"{name} must use torch.int32, got {tensor.dtype}")
        if tensor.device != selected_prev_page_slots.device:
            raise ValueError(f"{name} must share the selection device")
        if not tensor.is_contiguous():
            raise ValueError(f"{name} must be contiguous")
    for name, tensor in {
        "num_pages": num_pages,
        "context_lens": context_lens,
        "output_req_indices": output_req_indices,
        "output_context_lens": output_context_lens,
        "output_page_counts": output_page_counts,
        "output_last_page_lens": output_last_page_lens,
    }.items():
        if tensor.shape != (batch_size,):
            raise ValueError(f"{name} must have shape [{batch_size}]")
    page_size = int(page_size)
    token_budget = int(token_budget)
    if page_size <= 0 or token_budget <= 0:
        raise ValueError("page_size and token_budget must be positive")

    if not selected_prev_page_slots.is_cuda:
        safe_num_pages = num_pages.to(torch.long).clamp_min(1)
        last_pages = row_page_slots.gather(1, (safe_num_pages - 1)[:, None])
        sparse_pages = torch.cat((selected_prev_page_slots, last_pages), dim=1)
        last_page_lens = context_lens - (num_pages - 1) * page_size
        sparse_lens = prev_budget * page_size + last_page_lens
        if use_dense_fallback:
            use_dense = (context_lens <= token_budget) | (
                num_pages <= prev_budget + 1
            )
            positions = torch.arange(
                width, dtype=torch.int32, device=context_lens.device
            )
            dense_pages = torch.where(
                positions[None, :] < num_pages[:, None],
                row_page_slots[:, :width],
                0,
            )
            sparse_padded = torch.zeros_like(output_page_table)
            sparse_padded[:, : prev_budget + 1].copy_(sparse_pages)
            output_page_table.copy_(
                torch.where(use_dense[:, None], dense_pages, sparse_padded)
            )
            output_context_lens.copy_(torch.where(use_dense, context_lens, sparse_lens))
            output_page_counts.copy_(
                torch.where(use_dense, num_pages, prev_budget + 1)
            )
        else:
            output_page_table.zero_()
            output_page_table[:, : prev_budget + 1].copy_(sparse_pages)
            output_context_lens.copy_(sparse_lens)
            output_page_counts.fill_(prev_budget + 1)
        output_req_indices.copy_(
            torch.arange(batch_size, dtype=torch.int32, device=context_lens.device)
        )
        output_last_page_lens.copy_(last_page_lens)
        return (
            output_page_table,
            output_req_indices,
            output_context_lens,
            output_page_counts,
            output_last_page_lens,
        )

    block_size = triton.next_power_of_2(width)
    _finalize_quest_paged_decode_view_kernel[(batch_size,)](
        selected_prev_page_slots,
        row_page_slots,
        num_pages,
        context_lens,
        output_page_table,
        output_req_indices,
        output_context_lens,
        output_page_counts,
        output_last_page_lens,
        selected_prev_page_slots.stride(0),
        row_page_slots.stride(0),
        output_page_table.stride(0),
        WIDTH=width,
        PREV_BUDGET=prev_budget,
        PAGE_SIZE=page_size,
        TOKEN_BUDGET=token_budget,
        USE_DENSE_FALLBACK=bool(use_dense_fallback),
        BLOCK_SIZE=block_size,
        num_warps=min(max(block_size // 32, 1), 8),
        num_stages=1,
    )
    return (
        output_page_table,
        output_req_indices,
        output_context_lens,
        output_page_counts,
        output_last_page_lens,
    )


__all__ = [
    "finalize_quest_decode_view",
    "finalize_quest_paged_decode_view",
    "prepare_quest_decode_geometry",
    "score_quest_pages",
]
