"""SGL packed UE8M0 scales adapted to the FlashInfer SM90 FP32 scale layout."""

import torch

from .moe import _sgl_fp8_group_quant_op


def sgl_quantize_fp8_ue8m0(x, output, packed_scales, scales):
    """Write caller-owned FP8 values and column-major FP32 scales.

    SGL v2 packs four exponent bytes per int32. FlashInfer expects unpacked
    floats with the token axis padded to four, even for a one-token decode.
    """
    quantize, _ = _sgl_fp8_group_quant_op()
    rows, groups = len(x), x.shape[1] // 128
    quantize(x, output, packed_scales.T[:rows], 128, 1e-10, -448., 448.,
             scale_ue8m0=True, enable_v2=True)
    exponents = packed_scales.view(torch.uint8).view(
        len(packed_scales), packed_scales.shape[1], 4,
    ).permute(1, 0, 2)
    # copy_ performs the E8M0-to-FP32 conversion directly into prepared storage.
    scales[:rows].view(rows, len(packed_scales), 4).copy_(
        exponents[:rows].view(torch.float8_e8m0fnu),
    )
    return output, scales[:rows, :groups]
