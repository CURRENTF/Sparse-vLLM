"""Packed external scale interoperability and reusable output storage."""

import pytest
import torch

from sparsevllm.kernels.external.sgl.fp8_ue8m0 import sgl_quantize_fp8_ue8m0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows,groups", [(1, 1), (7, 7), (129, 32)])
@torch.inference_mode()
def test_sgl_packed_scales_match_independent_oracle_and_graph(rows, groups):
    # Wrong scale strides or interpreting packed exponent bytes as floats can
    # produce finite but incorrect GEMMs; ordinary random GEMM tests miss this.
    torch.manual_seed(731)
    x = torch.randn((rows, groups * 128), device="cuda", dtype=torch.bfloat16)
    x[0].zero_()
    if rows > 1:
        x[1].mul_(1e-7)
    padded = (rows + 3) // 4 * 4
    packs = (groups + 3) // 4
    output = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    packed = torch.empty((packs, padded), device=x.device, dtype=torch.int32)
    scales = torch.empty((packs * 4, padded), device=x.device).T

    def oracle():
        values = x.float().view(rows, groups, 128)
        sf = torch.exp2(torch.ceil(torch.log2(values.abs().amax(-1).clamp_min(1e-10) / 448)))
        quantized = (values / sf[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
        return quantized.view_as(x), sf

    sgl_quantize_fp8_ue8m0(x, output, packed, scales)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        sgl_quantize_fp8_ue8m0(x, output, packed, scales)
    for factor in (1., .5):
        x.mul_(factor)
        graph.replay()
        expected, expected_scales = oracle()
        torch.testing.assert_close(output.float(), expected.float(), rtol=0, atol=0)
        torch.testing.assert_close(scales[:rows, :groups], expected_scales, rtol=0, atol=0)
    graph.reset()
