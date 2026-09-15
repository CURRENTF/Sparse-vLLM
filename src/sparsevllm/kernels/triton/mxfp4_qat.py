"""MXFP4 activation simulation retaining the BF16 cache representation."""

import triton
import triton.language as tl


@triton.jit
def _qat(X, Out, D: tl.constexpr):
    row, group = tl.program_id(0), tl.program_id(1)
    col = group * 32 + tl.arange(0, 32)
    value = tl.load(X + row * D + col).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), 0), 6.0 * 2.0 ** -126)
    scaled = amax * (1.0 / 6.0)
    bits = scaled.to(tl.int32, bitcast=True)
    exponent = (bits >> 23) + ((bits & 0x7fffff) != 0).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    magnitude = tl.minimum(tl.abs(tl.div_rn(value, scale)), 6.0)
    # E2M1 round-to-nearest-even midpoints. Codes alternate parity even
    # across exponent transitions; exact midpoint ties retain the even code.
    quantized = tl.where(magnitude <= .25, 0.0,
                tl.where(magnitude < .75, .5,
                tl.where(magnitude <= 1.25, 1.,
                tl.where(magnitude < 1.75, 1.5,
                tl.where(magnitude <= 2.5, 2.,
                tl.where(magnitude < 3.5, 3.,
                tl.where(magnitude <= 5., 4., 6.)))))))
    result = quantized * scale
    result = tl.where(value < 0, -result, result)
    tl.store(Out + row * D + col, result)


def simulate_mxfp4(x, out):
    if x.numel():
        _qat[(x.numel() // x.shape[-1], x.shape[-1] // 32)](
            x, out, x.shape[-1], enable_fp_fusion=False, enable_reflect_ftz=False,
        )
