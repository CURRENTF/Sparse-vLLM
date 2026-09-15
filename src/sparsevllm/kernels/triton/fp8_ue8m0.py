"""FP8 activation quantization with reference-compatible power-of-two scales."""

import torch
import triton
import triton.language as tl


@triton.jit
def round_ue8m0_scale(amax):
    scaled = tl.maximum(amax, 1.0e-4) * (1.0 / 448.0)
    bits = scaled.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return (exponent << 23).to(tl.float32, bitcast=True)


@triton.jit
def _quantize(X, Out, Scales, XS: tl.constexpr, OS: tl.constexpr,
              SS0: tl.constexpr, SS1: tl.constexpr,
              GROUP: tl.constexpr, SIMULATE: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    col = group * GROUP + tl.arange(0, GROUP)
    x = tl.load(X + row * XS + col).to(tl.float32)
    scale = round_ue8m0_scale(tl.max(tl.abs(x), 0))
    q = tl.minimum(tl.maximum(x / scale, -448.0), 448.0).to(tl.float8e4nv)
    if SIMULATE:
        value = q.to(tl.float32) * scale
    else:
        value = q
    tl.store(Out + row * OS + col, value)
    tl.store(Scales + row * SS0 + group * SS1, scale)


def quantize_fp8_ue8m0(x, out, scales, *, group_size=128):
    """Write FP8 or BF16 quantize/dequantize outputs into caller-owned storage.

    Supports strided rows (e.g. the non-RoPE slice of a KV vector); columns
    are contiguous. BF16 output may alias x for QAT cache simulation.
    """
    if group_size not in (64, 128):
        raise ValueError("UE8M0 FP8 requires groups of 64 or 128 elements.")
    if x.ndim != 2 or x.shape[1] % group_size or x.shape[1] == 0:
        raise ValueError("UE8M0 FP8 input must have a positive group-aligned width.")
    if x.dtype != torch.bfloat16 or out.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("UE8M0 FP8 requires BF16 inputs and BF16 or E4M3 outputs.")
    if out.shape != x.shape or scales.shape != (x.shape[0], x.shape[1] // group_size):
        raise ValueError("UE8M0 FP8 output or scale shape does not match input groups.")
    if scales.dtype != torch.float32:
        raise TypeError("UE8M0 FP8 scales require FP32 storage.")
    if x.stride(1) != 1 or out.stride(1) != 1:
        raise ValueError("UE8M0 FP8 input/output columns must be contiguous.")
    if any(t.device != x.device for t in (out, scales)):
        raise ValueError("UE8M0 FP8 tensors must share a device.")
    if x.shape[0]:
        _quantize[(x.shape[0], scales.shape[1])](
            x, out, scales, x.stride(0), out.stride(0), scales.stride(0), scales.stride(1),
            group_size, out.dtype == torch.bfloat16, num_warps=4,
        )
