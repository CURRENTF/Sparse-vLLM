"""Protect grouped scale layout, vLLM quantization semantics and live graph padding."""

import sys

import pytest
import torch

from sparsevllm.layers.inverse_rotary_grouped_linear import InverseRotaryGroupedLinear
from sparsevllm.operators.inverse_rotary_grouped_linear import (
    InverseRotaryGroupedLinearSpec, RotaryTorchGroupedProjectionProvider,
    VllmDeepGemmInverseProjectionProvider,
)


def _reference(x, positions, frequencies, weight, scale, *, quantize):
    values = x.float().clone()
    angles = positions.clamp_min(0).float()[:, None] * frequencies[None]
    even, odd = values[..., -64::2].clone(), values[..., -63::2].clone()
    values[..., -64::2] = even * angles.cos()[:, None] + odd * angles.sin()[:, None]
    values[..., -63::2] = odd * angles.cos()[:, None] - even * angles.sin()[:, None]
    if quantize:
        blocks = values.reshape(*values.shape[:-1], -1, 128)
        scales = torch.exp2(torch.ceil(torch.log2(blocks.abs().amax(-1, keepdim=True).clamp_min(1e-10) / 448)))
        values = ((blocks / scales).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scales).reshape(values.shape)
    else:
        values = values.bfloat16().float()
    groups, outputs, inputs = weight.shape
    weights = weight.float() * scale.repeat_interleave(128, 1).repeat_interleave(128, 2)
    if not quantize:
        weights = weights.bfloat16().float()
    result = torch.stack([
        torch.nn.functional.linear(values.reshape(len(x), groups, inputs)[:, g], weights[g])
        for g in range(groups)
    ], 1)
    return result.masked_fill((positions < 0)[:, None, None], 0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rows", [1, 7, 128, 2048])
def test_external_inverse_projection_live_graph(rows):
    from sparsevllm.kernels.external.vllm_projection import inverse_projection_ops
    if torch.cuda.get_device_capability() != (9, 0) or inverse_projection_ops() is None:
        pytest.skip("requires SM90, native-Torch DeepGEMM and configured vLLM source")
    root_before = sys.modules.get("vllm")
    torch.manual_seed(731)
    groups, heads, inputs, outputs = 8, 64, 4096, 1024
    frequencies = 10000 ** (-torch.arange(32, device="cuda", dtype=torch.float32) / 32)
    layer = InverseRotaryGroupedLinear(groups, heads, 512, outputs, inv_freq=frequencies,
                                      max_num_tokens=rows, max_positions=131200, device_index=0)
    assert isinstance(layer.provider, VllmDeepGemmInverseProjectionProvider)
    weight = torch.randn((groups, outputs, inputs), device="cuda").to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-6, -2, (groups, outputs // 128, inputs // 128), device="cuda").float())
    layer.load_quantized_weight(weight.flatten(0, 1), scale.flatten(0, 1).to(torch.float8_e8m0fnu))
    torch.testing.assert_close(layer.weight.float(), weight.float(), rtol=0, atol=0)
    torch.testing.assert_close(layer.weight_scale, scale, rtol=0, atol=0)
    x = torch.randn((rows, heads, 512), device="cuda", dtype=torch.bfloat16)
    positions = torch.randint(0, 131200, (rows,), device="cuda", dtype=torch.int32)
    positions[0] = 131199
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            layer(x, positions)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(x, positions)
    pointer = actual.data_ptr()
    for step in range(3):
        original = x.clone()
        graph.replay()
        expected = _reference(x, positions, frequencies, weight, scale, quantize=True)
        active = positions >= 0
        if active.any():
            error = (actual[active].float() - expected[active]).norm() / expected[active].norm()
            assert error < .006, error.item()
        assert torch.count_nonzero(actual[~active]) == 0
        torch.testing.assert_close(actual, layer(x, positions), rtol=0, atol=0)
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        assert actual.data_ptr() == pointer
        x.normal_()
        positions.copy_(torch.randint(0, 131200, positions.shape, device="cuda", dtype=positions.dtype))
        positions[::2] = -1 if step == 0 else 0
    assert layer(x[:0], positions[:0]).shape == (0, groups, outputs)
    assert sys.modules.get("vllm") is root_before
    graph.reset()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_composed_projection_preserves_fallback_and_shared_table():
    from sparsevllm.operators.inverse_rotary_grouped_linear import _cos_sin_table
    frequencies = 10000 ** (-torch.arange(32, device="cuda", dtype=torch.float32) / 32)
    spec = InverseRotaryGroupedLinearSpec(2, 2, 512, 128, 7, 1024, tuple(frequencies.cpu().tolist()))
    device = torch.device("cuda", 0)
    table = _cos_sin_table(spec, device)
    assert _cos_sin_table(spec, device) is table
    op = RotaryTorchGroupedProjectionProvider(spec, device)
    weight = torch.randn((2, 128, 512), device=device).to(torch.float8_e4m3fn)
    scale = torch.full((2, 1, 4), .125, device=device)
    stored, stored_scale = op.allocate_weights()
    op.load_fp8_weights(stored, stored_scale, weight.flatten(0, 1), scale.flatten(0, 1))
    x = torch.randn((7, 2, 512), device=device, dtype=torch.bfloat16)
    positions = torch.tensor([-1, 0, 1, 127, 256, 511, 1023], device=device)
    expected = _reference(x, positions, frequencies, weight, scale, quantize=False)
    actual = op.run(x, positions, stored, stored_scale)
    assert (actual.float() - expected).norm() / expected.norm() < .004
    assert torch.count_nonzero(actual[0]) == 0
    assert op.run(x[:0], positions[:0], stored, stored_scale).shape == (0, 2, 128)


def test_missing_deepgemm_does_not_load_vllm(monkeypatch):
    from sparsevllm.kernels.external import vllm_projection
    vllm_projection.inverse_projection_ops.cache_clear()
    monkeypatch.setattr(vllm_projection.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(vllm_projection, "vllm_library", lambda *args: pytest.fail("loaded unselected dependency"))
    try:
        assert vllm_projection.inverse_projection_ops() is None
    finally:
        vllm_projection.inverse_projection_ops.cache_clear()


def test_incompatible_deepgemm_is_an_actionable_failure(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from sparsevllm.kernels.external import vllm_projection
    from sparsevllm.kernels.external.support import ExternalKernelContractError
    vllm_projection.inverse_projection_ops.cache_clear()
    original_find = vllm_projection.importlib.util.find_spec
    original_import = vllm_projection.importlib.import_module
    monkeypatch.setattr(vllm_projection.importlib.util, "find_spec",
                        lambda name, *args: object() if name == "deep_gemm" else original_find(name, *args))
    monkeypatch.setattr(vllm_projection.importlib, "import_module",
                        lambda name, *args: SimpleNamespace(__version__="0.0.0") if name == "deep_gemm" else original_import(name, *args))
    monkeypatch.setattr(vllm_projection, "vllm_library", lambda *args: tmp_path / "library.so")
    try:
        with pytest.raises(ExternalKernelContractError, match="built against the active Torch"):
            vllm_projection.inverse_projection_ops()
    finally:
        vllm_projection.inverse_projection_ops.cache_clear()
