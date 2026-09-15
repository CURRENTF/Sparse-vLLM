"""Adjacent-pair rotary transforms with explicit normalization rounding."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from sparsevllm.kernels.triton.fp8_ue8m0 import round_ue8m0_scale


@triton.jit
def _transform(X, Out, Positions, Freqs, Weight,
               D: tl.constexpr, RD: tl.constexpr, HEADS: tl.constexpr,
               EPS: tl.constexpr, NORM: tl.constexpr, INVERSE: tl.constexpr,
               QUANTIZE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    position = tl.load(Positions + row // HEADS)
    col = tl.arange(0, BLOCK)
    value = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    if NORM == "weighted_fp32":
        variance = tl.sum(value * value, 0) / D
        weight = tl.load(Weight + col, col < D, 0).to(tl.float32)
        value = ((value * tl.rsqrt(variance + EPS)) * weight).to(tl.bfloat16).to(tl.float32)
    elif NORM == "query_bf16":
        # The native query expression evaluates square, mean, epsilon add,
        # rsqrt and multiplication as separate BF16 operations.
        square = (value * value).to(tl.bfloat16).to(tl.float32)
        variance = (tl.sum(square, 0) / D).to(tl.bfloat16).to(tl.float32)
        shifted = (variance + EPS).to(tl.bfloat16).to(tl.float32)
        inv = tl.rsqrt(shifted).to(tl.bfloat16).to(tl.float32)
        value = (value * inv).to(tl.bfloat16).to(tl.float32)
    rotary = (col >= D - RD) & (col < D)
    freq = tl.load(Freqs + (col - (D - RD)) // 2, rotary, 0)
    phase = tl.maximum(position, 0).to(tl.float32) * freq
    cosine, sine = libdevice.cos(phase), libdevice.sin(phase)
    if INVERSE:
        sine = -sine
    partner = tl.gather(value, col ^ 1, 0)
    # Match the contraction in Torch's complex multiplication. A different
    # FMA order can cross a BF16 midpoint before index quantization.
    rotated = libdevice.fma(tl.where(col % 2 == 0, -partner, partner), sine, value * cosine)
    value = tl.where(rotary, rotated, value).to(tl.bfloat16).to(tl.float32)
    if QUANTIZE and D > RD:
        groups = tl.reshape(value, (BLOCK // 64, 64))
        scale = round_ue8m0_scale(tl.max(tl.abs(groups), 1))
        quantized = (groups / scale[:, None]).to(tl.float8e4nv).to(tl.float32) * scale[:, None]
        value = tl.where(col < D - RD, tl.reshape(quantized, (BLOCK,)), value)
    tl.store(Out + row * D + col, tl.where(position >= 0, value, 0.), col < D)


def native_rotary_transform(x, positions, inv_freq, weight, out, *, norm, eps, inverse, quantize_nope):
    if x.numel():
        _transform[(x.numel() // x.shape[-1],)](
            x, out, positions, inv_freq, weight, x.shape[-1], inv_freq.numel() * 2,
            x.shape[1], eps, norm, inverse, quantize_nope,
            triton.next_power_of_2(x.shape[-1]), enable_fp_fusion=False,
        )
