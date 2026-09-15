"""Reference QAT rounding and prepared Linear workspace contracts."""

from dataclasses import replace

import pytest
import torch

from sparsevllm.kernels.triton.fp8_ue8m0 import quantize_fp8_ue8m0
from sparsevllm.kernels.triton.fp8_blockwise import fp8_blockwise_matmul
from sparsevllm.operators.fp8_linear import Fp8LinearSpec, resolve_fp8_linear_provider
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager
from sparsevllm.quantization.config import QuantizationConfig


def _reference_quantize(x, group_size):
    groups = x.float().reshape(x.shape[0], -1, group_size)
    scales = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(1e-4) / 448.)))
    quantized = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return quantized.reshape_as(x), scales


def test_scale_format_survives_checkpoint_config_roundtrip():
    # Dropping this field silently selects different activation quantization.
    raw = {"quant_method": "fp8", "fmt": "e4m3", "activation_scheme": "dynamic",
           "weight_block_size": [128, 128], "scale_fmt": "ue8m0"}
    config = QuantizationConfig.from_hf_config(raw, max_num_tokens=17)
    assert config.to_dict() == raw
    restored = QuantizationConfig.from_hf_config(config.to_dict())
    assert replace(config, max_num_tokens=None) == restored
    with pytest.raises(ValueError, match="scale_fmt"):
        QuantizationConfig.from_hf_config({**raw, "scale_fmt": "unknown"})
    with pytest.raises(ValueError, match="workspace bound"):
        Fp8LinearSpec((128, 128), 128, 128, scale_fmt="ue8m0")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("group_size,dim", [(64, 448), (128, 4096)])
def test_fp8_quantization_and_strided_inplace_simulation(group_size, dim):
    # Protect tiny-input clamping, exact power-of-two boundaries, strided
    # non-RoPE KV slices and round-to-nearest FP8 values.
    torch.manual_seed(731)
    backing = torch.randn((7, dim+64), device="cuda", dtype=torch.bfloat16)
    x = backing[:, :dim]
    x[0].zero_()
    x[1].mul_(1e-7)
    x[2].fill_(448 * 2**-8)
    x[3].fill_(torch.nextafter(x[2, 0], x.new_tensor(float("inf"))))
    rotary = backing[:, dim:].clone()
    out = torch.empty(x.shape, device="cuda", dtype=torch.float8_e4m3fn)
    scales = torch.empty((7, dim//group_size), device="cuda")
    quantize_fp8_ue8m0(x, out, scales, group_size=group_size)
    expected_q, expected_s = _reference_quantize(x, group_size)
    torch.testing.assert_close(scales, expected_s, rtol=0, atol=0)
    torch.testing.assert_close(out.float(), expected_q.float(), rtol=0, atol=0)
    expected = (expected_q.float().reshape(7, -1, group_size) * expected_s[..., None]).reshape_as(x).bfloat16()
    quantize_fp8_ue8m0(x, x, scales, group_size=group_size)
    torch.testing.assert_close(x, expected, rtol=0, atol=0)
    torch.testing.assert_close(backing[:, dim:], rotary, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m,n,k", [(7, 512, 4096), (33, 1024, 1024)])
def test_prepared_ue8m0_linear_matches_dequantized_oracle_and_graph(m, n, k):
    close_workspace_manager()
    try:
        torch.manual_seed(19)
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
        weight = (torch.randn((n, k), device="cuda") * 32).to(torch.float8_e4m3fn)
        scale = torch.exp2(torch.randint(-11, -6, (n//128, k//128), device="cuda").float())
        provider = resolve_fp8_linear_provider(
            (128, 128), input_features=k, output_features=n,
            scale_fmt="ue8m0", max_num_tokens=m + 9, device_index=0,
        )
        lock_workspace_manager()
        # An idle DP rank may execute a step without local tokens.
        assert provider(x[:0], weight, scale).shape == (0, n)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            provider(x, weight, scale)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = provider(x, weight, scale)
        for amplitude in (1., 1e-5):
            x.mul_(amplitude)
            graph.replay()
            eager = provider(x, weight, scale)
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)
            q, s = _reference_quantize(x, 128)
            a = (q.float().reshape(m, -1, 128) * s[..., None]).reshape(m, k)
            b = weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
            expected = (a @ b.T).bfloat16()
            torch.testing.assert_close(captured, expected, rtol=1e-2, atol=3e-3)
            for dtype in (torch.bfloat16, torch.float16):
                portable_x = x.to(dtype)
                portable_q, portable_s = _reference_quantize(portable_x, 128)
                portable_a = (portable_q.float().reshape(m, -1, 128) * portable_s[..., None]).reshape(m, k)
                portable_expected = (portable_a @ b.T).to(dtype)
                portable = fp8_blockwise_matmul(portable_x, weight, scale, scale_fmt="ue8m0")
                torch.testing.assert_close(portable, portable_expected, rtol=1e-2, atol=3e-3)
            # A smaller eager batch uses the same storage with a different
            # padded scale stride; replay must restore the captured layout.
            smaller = provider(x[:1], weight, scale)
            torch.testing.assert_close(smaller, expected[:1], rtol=1e-2, atol=3e-3)
            graph.replay()
            torch.testing.assert_close(captured, expected, rtol=1e-2, atol=3e-3)
        with pytest.raises(ValueError, match="workspace capacity"):
            provider(x.repeat(4, 1), weight, scale)
    finally:
        close_workspace_manager()
