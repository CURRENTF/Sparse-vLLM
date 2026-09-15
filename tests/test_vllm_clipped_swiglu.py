import pytest
import torch

from sparsevllm.kernels.external.vllm_moe import clipped_swiglu_op
from sparsevllm.operators.activation import resolve_silu_and_mul_provider
from sparsevllm.kernels.external.support import ExternalKernelContractError


def test_configured_wheel_missing_activation_library_fails(monkeypatch, tmp_path):
    # A partial configured wheel must not silently use the portable provider.
    moe_library = tmp_path / "_moe_C_stable_libtorch.abi3.so"
    moe_library.touch()
    monkeypatch.setenv("SPARSEVLLM_VLLM_MOE_LIBRARY", str(moe_library))
    clipped_swiglu_op.cache_clear()
    try:
        with pytest.raises(ExternalKernelContractError, match="required vLLM library"):
            clipped_swiglu_op()
    finally:
        clipped_swiglu_op.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_clipped_swiglu_matches_independent_oracle_and_live_graph():
    # Detect reversed gate/up halves, symmetric gate clipping, premature BF16
    # rounding, input mutation, and stale graph inputs at the external boundary.
    if clipped_swiglu_op() is None:
        pytest.skip("optional vLLM library is unavailable")
    torch.manual_seed(731)
    provider = resolve_silu_and_mul_provider(activation_dtype=torch.bfloat16, swiglu_limit=10., device_index=0)
    assert provider.name == "vllm_clipped"
    for rows, width in ((1, 3), (7, 128), (33, 2048)):
        x = (torch.randn(rows, 2 * width, device="cuda") * 20).bfloat16()
        def check(actual):
            gate, up = x.double().chunk(2, -1)
            expected = (torch.nn.functional.silu(gate.clamp_max(10.)) * up.clamp(-10., 10.)).bfloat16()
            torch.testing.assert_close(actual, expected, rtol=.008, atol=1e-6)
        original = x.clone()
        check(provider(x))
        torch.testing.assert_close(x, original, rtol=0, atol=0)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3): provider(x)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph): out = provider(x)
        for _ in range(2):
            x.normal_(0, 20)
            graph.replay()
            check(out)
        graph.reset()
    assert provider(x[:0]).shape == (0, x.shape[1] // 2)
    with pytest.raises(ValueError, match="contiguous"):
        provider(x[:, ::2])
    with pytest.raises(TypeError, match="dtype"):
        provider(x.float())
