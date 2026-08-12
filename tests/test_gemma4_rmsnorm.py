import pytest
import torch

from sparsevllm.layers.gemma4_rmsnorm import Gemma4RMSNorm


def _reference(x: torch.Tensor, weight: torch.Tensor | None, eps: float) -> torch.Tensor:
    output = x.float()
    output *= torch.pow(output.square().mean(-1, keepdim=True) + eps, -0.5)
    if weight is not None:
        output *= weight.float()
    return output.to(x.dtype)


@pytest.mark.parametrize("with_scale", [False, True])
def test_gemma4_rmsnorm_matches_torch(with_scale):
    torch.manual_seed(7)
    layer = Gemma4RMSNorm(32, with_scale=with_scale)
    x = torch.randn(5, 32)
    torch.testing.assert_close(layer(x), _reference(x, layer.weight, layer.eps))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_gemma4_rmsnorm_cuda_graph_matches_torch():
    torch.manual_seed(11)
    layer = Gemma4RMSNorm(2816).cuda().to(torch.bfloat16)
    x = torch.randn(4, 2816, device="cuda", dtype=torch.bfloat16)
    for _ in range(2):
        layer(x)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(x)
    graph.replay()
    torch.testing.assert_close(output, _reference(x, layer.weight, layer.eps), rtol=0, atol=0)
