"""Checkpoint block scales and grouped projection compared with separate GEMMs."""

import pytest
import torch

from sparsevllm.operators.grouped_linear import GroupedLinearSpec, prepare_grouped_linear


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_linear_fp8_loading_and_graph():
    torch.manual_seed(731)
    groups, inputs, outputs, rows = 2, 256, 256, 7
    op = prepare_grouped_linear(GroupedLinearSpec(groups, inputs, outputs, rows), device_index=0)
    weight = torch.randn((groups * outputs, inputs), device="cuda").to(torch.float8_e4m3fn)
    scale = torch.exp2(torch.randint(-3, 2, (groups * outputs // 128, inputs // 128), device="cuda").float())
    stored = op.allocate_weights()
    op.load_fp8_weights(stored, weight, scale.to(torch.float8_e8m0fnu))
    expected_weight = (weight.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)).bfloat16()
    torch.testing.assert_close(stored.flatten(0, 1), expected_weight, rtol=0, atol=0)
    x = torch.randn((rows, groups, inputs), device="cuda", dtype=torch.bfloat16)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        op.run(x, stored)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = op.run(x, stored)
    for _ in range(2):
        graph.replay()
        expected = torch.stack([torch.nn.functional.linear(x[:, group].float(), stored[group].float())
                                for group in range(groups)], 1).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=8e-3)
        torch.testing.assert_close(actual, op.run(x, stored), rtol=0, atol=0)
        x.mul_(.5)
    assert op.run(x[:0], stored).shape == (0, groups, outputs)
    graph.reset()
