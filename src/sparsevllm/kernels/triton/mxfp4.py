"""Packed MXFP4 GEMM with per-128 FP8 activation and per-32 weight scales."""

import torch
import triton
import triton.language as tl

from sparsevllm.kernels.moe import MoeAlignment


@triton.jit
def _gemm(A, AS, B, BS, C, Sorted, Experts, Count,
          ROWS: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          INPUT_TOP_K: tl.constexpr, ROUTED: tl.constexpr, NAIVE: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr):
    block = tl.program_id(0)
    offsets_m = tl.arange(0, BM)
    if ROUTED:
        if block * BM >= tl.load(Count):
            return
        expert = tl.load(Experts + block)
        if expert < 0:
            return
        if NAIVE:
            rows = tl.where(offsets_m == 0, block, ROWS)
        else:
            rows = tl.load(Sorted + block * BM + offsets_m)
    else:
        expert = 0
        rows = block * BM + offsets_m
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, 32)
    a_rows = rows // INPUT_TOP_K
    accumulator = tl.full((BM, BN), 0, tl.float32)
    for group in range(K // 32):
        a = tl.load(A + a_rows[:, None] * K + group * 32 + k[None, :],
                    rows[:, None] < ROWS, 0.0)
        a_scale = tl.load(AS + a_rows * (K // 128) + group // 4, rows < ROWS, 0.0)
        packed = tl.load(B + (expert * N + n[None, :]) * (K // 2)
                         + group * 16 + k[:, None] // 2, n[None, :] < N, 0)
        code = (packed >> ((k[:, None] % 2) * 4)) & 15
        magnitude = code & 7
        fp8_bits = tl.where(magnitude < 2, magnitude * 48, (magnitude + 12) * 4) | ((code & 8) << 4)
        b = fp8_bits.to(tl.uint8).to(tl.float8e4nv, bitcast=True)
        scale_byte = tl.load(BS + (expert * N + n) * (K // 32) + group, n < N, 127).to(tl.int32)
        scale_bits = tl.where(scale_byte == 0, 0x00400000, scale_byte << 23)
        b_scale = tl.where(scale_byte == 255, float("nan"), scale_bits.to(tl.float32, bitcast=True))
        # Apply scales after each 32-element dot, matching the reference's
        # reduction order before BF16 rounding and subsequent FP8 quantization.
        accumulator += tl.dot(a, b) * a_scale[:, None] * b_scale[None, :]
    tl.store(C + rows[:, None] * N + n[None, :], accumulator,
             (rows[:, None] < ROWS) & (n[None, :] < N))


def mxfp4_gemm(a, a_scale, weight, weight_scale, out, *,
                alignment: MoeAlignment | None = None, input_top_k=1):
    """Write BF16 projections without expanding persistent packed weights.

    Routed output rows use flattened token/route order. Alignment contains
    local expert IDs; negative expert blocks are skipped, leaving their output
    rows untouched. Consumers must mask those routes.
    """
    if a.ndim != 2 or a.shape[1] == 0 or a.shape[1] % 128:
        raise ValueError("MXFP4 GEMM requires rank-2 activations and K aligned to 128.")
    if a.dtype != torch.float8_e4m3fn or a_scale.dtype != torch.float32:
        raise TypeError("MXFP4 GEMM requires E4M3 activations and FP32 activation scales.")
    if weight.dtype != torch.uint8 or weight_scale.dtype != torch.uint8 or out.dtype != torch.bfloat16:
        raise TypeError("MXFP4 GEMM requires packed uint8 weights/E8M0 scales and BF16 output.")
    if input_top_k <= 0 or out.ndim != 2:
        raise ValueError("MXFP4 GEMM requires positive input_top_k and rank-2 output.")
    k, n = a.shape[1], out.shape[1]
    if out.shape[0] != a.shape[0] * input_top_k or a_scale.shape != (a.shape[0], k // 128):
        raise ValueError("MXFP4 GEMM activation/output shapes disagree.")
    if weight.ndim != (3 if alignment is not None else 2):
        raise ValueError("MXFP4 GEMM weight rank does not match routing mode.")
    if weight.shape[-2:] != (n, k // 2) or weight_scale.shape != (*weight.shape[:-1], k // 32):
        raise ValueError("MXFP4 GEMM weight/scales do not match N and K.")
    if not a.is_cuda or any(t.device != a.device or not t.is_contiguous()
                            for t in (a, a_scale, weight, weight_scale, out)):
        raise ValueError("MXFP4 GEMM tensors must be contiguous and share a CUDA device.")
    if out.shape[0] == 0:
        return
    bm = alignment.block_size if alignment is not None else 16
    blocks = alignment.expert_ids.numel() if alignment is not None else triton.cdiv(out.shape[0], bm)
    _gemm[(blocks, triton.cdiv(n, 64))](
        a, a_scale, weight, weight_scale, out,
        alignment.sorted_token_ids if alignment else None,
        alignment.expert_ids if alignment else None,
        alignment.num_tokens_post_padded if alignment else None,
        out.shape[0], n, k, input_top_k, alignment is not None,
        alignment.naive if alignment else False, bm, 64, num_warps=4,
    )
