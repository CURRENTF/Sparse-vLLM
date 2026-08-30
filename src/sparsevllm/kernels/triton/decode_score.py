from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _decode_score_lse_kernel(
    Scores,
    Candidate_Lens,
    Lse,
    stride_sb,
    stride_sh,
    stride_sl,
    stride_lseb,
    stride_lseh,
    softmax_scale: tl.constexpr,
    CANDIDATE_START: tl.constexpr,
    MAX_CANDIDATES: tl.constexpr,
    HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_head = tl.program_id(0)
    batch = batch_head // HEADS
    head = batch_head % HEADS
    candidate_len = tl.maximum(
        0,
        tl.minimum(tl.load(Candidate_Lens + batch), MAX_CANDIDATES),
    )
    running_max = -float("inf")
    running_sum = 0.0
    offsets = tl.arange(0, BLOCK_N)
    block_count = tl.cdiv(candidate_len, BLOCK_N)
    for block_index in range(0, block_count):
        candidate_offsets = block_index * BLOCK_N + offsets
        valid = candidate_offsets < candidate_len
        scores = tl.load(
            Scores
            + batch * stride_sb
            + head * stride_sh
            + (CANDIDATE_START + candidate_offsets) * stride_sl,
            mask=valid,
            other=-float("inf"),
        ).to(tl.float32)
        logits = scores * softmax_scale
        block_max = tl.max(logits, axis=0)
        next_max = tl.maximum(running_max, block_max)
        old_scale = tl.exp(running_max - next_max)
        block_sum = tl.sum(tl.exp(logits - next_max), axis=0)
        running_sum = running_sum * old_scale + block_sum
        running_max = next_max
    lse = running_max + tl.log(running_sum)
    tl.store(Lse + batch * stride_lseb + head * stride_lseh, lse)


@triton.jit
def _decode_score_reduce_kernel(
    Scores,
    Candidate_Lens,
    Lse,
    Output,
    stride_sb,
    stride_sh,
    stride_sl,
    stride_lseb,
    stride_lseh,
    stride_ob,
    stride_ol,
    softmax_scale: tl.constexpr,
    CANDIDATE_START: tl.constexpr,
    SCORE_WIDTH: tl.constexpr,
    HEADS: tl.constexpr,
    REDUCE_HEADS: tl.constexpr,
    BLOCK_N: tl.constexpr,
    MIN_OUTPUT: tl.constexpr,
):
    batch = tl.program_id(0)
    positions = tl.program_id(1) * BLOCK_N + tl.arange(0, BLOCK_N)
    heads = tl.arange(0, REDUCE_HEADS)
    candidate_len = tl.maximum(
        0,
        tl.minimum(
            tl.load(Candidate_Lens + batch),
            SCORE_WIDTH - CANDIDATE_START,
        ),
    )
    candidate_offsets = positions - CANDIDATE_START
    valid_positions = (
        (positions >= CANDIDATE_START)
        & (candidate_offsets < candidate_len)
        & (positions < SCORE_WIDTH)
    )
    valid_heads = heads < HEADS
    scores = tl.load(
        Scores
        + batch * stride_sb
        + heads[:, None] * stride_sh
        + positions[None, :] * stride_sl,
        mask=valid_heads[:, None] & valid_positions[None, :],
        other=-float("inf"),
    ).to(tl.float32)
    lse = tl.load(
        Lse + batch * stride_lseb + heads * stride_lseh,
        mask=valid_heads,
        other=float("inf"),
    )
    probabilities = tl.exp(scores * softmax_scale - lse[:, None])
    reduced = tl.max(
        tl.where(valid_heads[:, None], probabilities, 0.0),
        axis=0,
    )
    tl.store(
        Output + batch * stride_ob + positions * stride_ol,
        tl.where(valid_positions, reduced, MIN_OUTPUT),
        mask=positions < SCORE_WIDTH,
    )


@torch.no_grad()
def decode_softmax_token_scores(
    scores: torch.Tensor,
    candidate_lens: torch.Tensor,
    *,
    candidate_start: int,
    softmax_scale: float,
    output_dtype: torch.dtype,
    lse_workspace: torch.Tensor | None = None,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Normalize per-head raw decode scores and reduce the head dimension."""

    if scores.ndim != 3:
        raise ValueError(
            "Decode scores must have shape [batch, heads, width], got "
            f"{tuple(scores.shape)}."
        )
    if scores.device.type != "cuda":
        raise ValueError(f"Decode scores must be CUDA tensors, got {scores.device}.")
    if scores.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"Unsupported decode score dtype {scores.dtype}.")
    if scores.stride(-1) != 1:
        raise ValueError(
            "Decode score width must be contiguous, got stride "
            f"{tuple(scores.stride())}."
        )
    batch, heads, score_width = map(int, scores.shape)
    candidate_start = int(candidate_start)
    if not 0 <= candidate_start <= score_width:
        raise ValueError(
            "candidate_start must be within the decode score width, got "
            f"start={candidate_start} width={score_width}."
        )
    if candidate_lens.shape != (batch,):
        raise ValueError(
            "candidate_lens must have one entry per batch row, got "
            f"{tuple(candidate_lens.shape)} for batch={batch}."
        )
    if (
        candidate_lens.device != scores.device
        or candidate_lens.dtype not in {torch.int32, torch.int64}
        or candidate_lens.stride(0) != 1
    ):
        raise TypeError(
            "candidate_lens must be contiguous int32/int64 on the score device."
        )
    if output_dtype not in {torch.float16, torch.bfloat16, torch.float32}:
        raise TypeError(f"Unsupported decode score output dtype {output_dtype}.")
    if not float(softmax_scale) > 0.0:
        raise ValueError("Decode score softmax_scale must be positive.")

    if lse_workspace is None:
        lse_workspace = torch.empty(
            (batch, heads), dtype=torch.float32, device=scores.device
        )
    elif (
        tuple(lse_workspace.shape) != (batch, heads)
        or lse_workspace.dtype != torch.float32
        or lse_workspace.device != scores.device
        or lse_workspace.stride(-1) != 1
    ):
        raise ValueError(
            "Decode score LSE workspace must be contiguous FP32 [batch, heads] "
            f"on {scores.device}, got shape={tuple(lse_workspace.shape)} "
            f"dtype={lse_workspace.dtype} device={lse_workspace.device}."
        )
    if output is None:
        output = torch.empty(
            (batch, score_width), dtype=output_dtype, device=scores.device
        )
    elif (
        tuple(output.shape) != (batch, score_width)
        or output.dtype != output_dtype
        or output.device != scores.device
        or output.stride(-1) != 1
    ):
        raise ValueError(
            "Decode score output must be contiguous with shape [batch, width], "
            f"dtype={output_dtype}, and device={scores.device}."
        )

    max_candidates = score_width - candidate_start
    if max_candidates == 0:
        output.fill_(torch.finfo(output_dtype).min)
        return output
    block_n = 256
    _decode_score_lse_kernel[(batch * heads,)](
        scores,
        candidate_lens,
        lse_workspace,
        *scores.stride(),
        *lse_workspace.stride(),
        softmax_scale=float(softmax_scale),
        CANDIDATE_START=candidate_start,
        MAX_CANDIDATES=max_candidates,
        HEADS=heads,
        BLOCK_N=block_n,
        num_warps=4,
        num_stages=2,
    )
    reduce_heads = triton.next_power_of_2(heads)
    _decode_score_reduce_kernel[(batch, triton.cdiv(score_width, block_n))](
        scores,
        candidate_lens,
        lse_workspace,
        output,
        *scores.stride(),
        *lse_workspace.stride(),
        *output.stride(),
        softmax_scale=float(softmax_scale),
        CANDIDATE_START=candidate_start,
        SCORE_WIDTH=score_width,
        HEADS=heads,
        REDUCE_HEADS=reduce_heads,
        BLOCK_N=block_n,
        MIN_OUTPUT=float(torch.finfo(output_dtype).min),
        num_warps=8 if reduce_heads >= 64 else 4,
        num_stages=3,
    )
    return output


__all__ = ["decode_softmax_token_scores"]
