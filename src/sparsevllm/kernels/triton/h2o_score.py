from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _h2o_softmax_accumulate_kernel(
    logits,
    cumulative,
    stride_ll,
    stride_lb,
    stride_lw,
    stride_cl,
    stride_cb,
    stride_cw,
    batch_size,
    width,
    previous_width,
    softmax_scale: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    layer = row // batch_size
    batch = row - layer * batch_size
    offsets = tl.arange(0, BLOCK)
    valid = offsets < width
    values = tl.load(
        logits
        + layer * stride_ll
        + batch * stride_lb
        + offsets * stride_lw,
        mask=valid,
        other=-float("inf"),
    )
    values = values * softmax_scale
    values = values - tl.max(values, axis=0)
    probabilities = tl.exp(values)
    probabilities = probabilities / tl.sum(probabilities, axis=0)
    previous = tl.load(
        cumulative
        + layer * stride_cl
        + batch * stride_cb
        + offsets * stride_cw,
        mask=offsets < previous_width,
        other=0.0,
    )
    tl.store(
        cumulative
        + layer * stride_cl
        + batch * stride_cb
        + offsets * stride_cw,
        previous + probabilities,
        mask=valid,
    )


@torch.no_grad()
def h2o_softmax_accumulate(
    raw_logits: torch.Tensor,
    cumulative_scores: torch.Tensor,
    *,
    width: int,
    previous_width: int,
    softmax_scale: float,
) -> None:
    """Normalize and accumulate every H2O layer/batch row in one launch."""

    if raw_logits.ndim != 3 or cumulative_scores.ndim != 3:
        raise ValueError(
            "H2O fused score update requires [layers, batch, width] tensors, "
            f"got {tuple(raw_logits.shape)} and {tuple(cumulative_scores.shape)}."
        )
    if tuple(raw_logits.shape[:2]) != tuple(cumulative_scores.shape[:2]):
        raise ValueError(
            "H2O fused score tensors disagree on layers or batch: "
            f"logits={tuple(raw_logits.shape)} cumulative={tuple(cumulative_scores.shape)}."
        )
    if raw_logits.dtype != torch.float32 or cumulative_scores.dtype != torch.float32:
        raise TypeError(
            "H2O fused score update requires FP32 logits and cumulative scores, "
            f"got {raw_logits.dtype} and {cumulative_scores.dtype}."
        )
    if not raw_logits.is_cuda or not cumulative_scores.is_cuda:
        raise TypeError("H2O fused score update requires CUDA tensors.")
    if raw_logits.device != cumulative_scores.device:
        raise ValueError(
            "H2O fused score tensors must share a device, got "
            f"{raw_logits.device} and {cumulative_scores.device}."
        )
    width = int(width)
    previous_width = int(previous_width)
    if not 0 <= previous_width <= width <= int(raw_logits.shape[2]):
        raise ValueError(
            "H2O fused score widths are invalid: "
            f"previous={previous_width} width={width} logits={int(raw_logits.shape[2])}."
        )
    if width > int(cumulative_scores.shape[2]):
        raise ValueError(
            "H2O cumulative score capacity is too small: "
            f"width={width} capacity={int(cumulative_scores.shape[2])}."
        )
    if width == 0:
        return

    rows = int(raw_logits.shape[0]) * int(raw_logits.shape[1])
    block = triton.next_power_of_2(int(raw_logits.shape[2]))
    _h2o_softmax_accumulate_kernel[(rows,)](
        raw_logits,
        cumulative_scores,
        *raw_logits.stride(),
        *cumulative_scores.stride(),
        int(raw_logits.shape[1]),
        width,
        previous_width,
        softmax_scale=float(softmax_scale),
        BLOCK=block,
        num_warps=8,
        num_stages=1,
    )


__all__ = ["h2o_softmax_accumulate"]
