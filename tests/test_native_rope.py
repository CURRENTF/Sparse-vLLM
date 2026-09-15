"""Native BF16 rounding, long positions and inactive graph rows."""

import pytest
import torch

from sparsevllm.operators.native_rope import NativeRotarySpec, prepare_native_rotary


def _reference(x, positions, inv_freq, spec, weight):
    if spec.norm == "weighted_fp32":
        value = x.float()
        value = ((value * torch.rsqrt(value.square().mean(-1, keepdim=True) + spec.eps)) * weight).bfloat16()
    elif spec.norm == "query_bf16":
        value = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + spec.eps)
    else:
        value = x.clone()
    phases = positions.clamp_min(0).float()[:, None] * inv_freq
    freqs = torch.polar(torch.ones_like(phases), phases)
    if spec.inverse:
        freqs = freqs.conj()
    rotary = torch.view_as_complex(value[..., -spec.rotary_dim:].float().unflatten(-1, (-1, 2)))
    value[..., -spec.rotary_dim:] = torch.view_as_real(rotary * freqs[:, None]).flatten(-2).bfloat16()
    if spec.quantize_nope:
        groups = value[..., :-spec.rotary_dim].float().unflatten(-1, (-1, 64))
        scales = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448)))
        rounded = (groups / scales[..., None]).to(torch.float8_e4m3fn).float() * scales[..., None]
        value[..., :-spec.rotary_dim] = rounded.flatten(-2).bfloat16()
    value[positions < 0] = 0
    return value


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dim,heads,norm,quantize,inverse", [
    (512, 1, "weighted_fp32", True, False),
    (512, 64, "query_bf16", False, False),
    (512, 16, "none", False, True),
    (128, 1, "weighted_fp32", False, False),
    (32, 1, "none", True, False),
])
def test_native_rotary_rounding_positions_and_graph(dim, heads, norm, quantize, inverse):
    torch.manual_seed(731)
    rotary_dim = min(dim, 64)
    spec = NativeRotarySpec(dim, rotary_dim, norm, inverse=inverse, quantize_nope=quantize)
    op = prepare_native_rotary(spec, device_index=0)
    x = torch.randn((7, heads, dim), device="cuda", dtype=torch.bfloat16)
    x[0].mul_(1e-6)
    positions = torch.tensor([0, 3, 127, 65535, 100003, 1048575, -1], device="cuda", dtype=torch.int32)
    inv_freq = 1 / (160000 ** (torch.arange(0, rotary_dim, 2, device="cuda", dtype=torch.float32) / rotary_dim))
    weight = torch.randn(dim, device="cuda") if norm == "weighted_fp32" else None
    output = torch.empty_like(x)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.run(x, positions, inv_freq, weight=weight, out=output)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        op.run(x, positions, inv_freq, weight=weight, out=output)
    for _ in range(2):
        graph.replay()
        expected = _reference(x, positions, inv_freq, spec, weight)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        alias = x.clone()
        op.run(alias, positions, inv_freq, weight=weight, out=alias)
        torch.testing.assert_close(alias, output, rtol=0, atol=0)
        positions.copy_(positions.roll(1))
        x.mul_(.75)
