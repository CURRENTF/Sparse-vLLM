"""MXFP4 ties and Hadamard QAT against a direct matrix/codebook oracle."""

import pytest
import torch

from sparsevllm.kernels.triton.mxfp4_qat import simulate_mxfp4
from sparsevllm.operators.rotated_mxfp4 import RotatedMxfp4Spec, prepare_rotated_mxfp4


def _reference_qat(x):
    groups = x.float().unflatten(-1, (-1, 32))
    scales = torch.exp2(torch.ceil(torch.log2(groups.abs().amax(-1).clamp_min(6 * 2. ** -126) / 6)))
    values = groups / scales[..., None]
    codebook = x.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], dtype=torch.float32)
    distance = (values.abs()[..., None] - codebook).abs()
    closest = distance == distance.amin(-1, keepdim=True)
    # Prefer even code indices at equal distance, independently of threshold
    # comparisons used by the production simulation kernel.
    priority = torch.tensor([0, 8, 2, 10, 4, 12, 6, 14], device=x.device)
    code = torch.where(closest, priority, 99).argmin(-1)
    return (codebook[code] * values.sign() * scales[..., None]).flatten(-2).bfloat16()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mxfp4_simulation_midpoint_ties_and_tiny_groups():
    ties = torch.tensor([0, .25, .75, 1.25, 1.75, 2.5, 3.5, 5., 6.], device="cuda", dtype=torch.bfloat16)
    x = torch.zeros((4, 128), device="cuda", dtype=torch.bfloat16)
    x[0, :len(ties)] = ties
    x[1, :len(ties)] = -ties
    x[2].fill_(2. ** -126)
    x[3].normal_()
    expected = _reference_qat(x)
    simulate_mxfp4(x, x)
    torch.testing.assert_close(x, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_rotated_mxfp4_matches_hadamard_matrix_and_replays():
    pytest.importorskip("fast_hadamard_transform")
    torch.manual_seed(731)
    x = torch.randn((7, 4, 128), device="cuda", dtype=torch.bfloat16)
    h = torch.ones((1, 1), device="cuda", dtype=torch.float32)
    while h.shape[0] < x.shape[-1]:
        h = torch.cat((torch.cat((h, h), 1), torch.cat((h, -h), 1)), 0)
    op = prepare_rotated_mxfp4(RotatedMxfp4Spec(128), device_index=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.run(x)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = op.run(x)
    for _ in range(2):
        x.mul_(.75)
        graph.replay()
        expected = _reference_qat(((x.float() @ h) * 128 ** -.5).bfloat16())
        torch.testing.assert_close(out, expected, rtol=0, atol=0)
        torch.testing.assert_close(out, op.run(x), rtol=0, atol=0)
