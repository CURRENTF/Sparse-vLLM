"""Independent Sinkhorn and residual-axis oracle, including changing graph inputs."""

import pytest
import torch

from sparsevllm.operators.hyper_connection import (
    HyperConnectionHeadSpec, HyperConnectionSpec, prepare_hyper_connection,
    prepare_hyper_connection_head,
)
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager


def _reference_pre(x, weight, scale, base, spec):
    flat = x.flatten(1).float()
    mix = torch.nn.functional.linear(flat, weight) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + spec.norm_eps)
    pre = torch.sigmoid(mix[:, :4] * scale[0] + base[:4]) + spec.mixing_eps
    post = 2 * torch.sigmoid(mix[:, 4:8] * scale[1] + base[4:8])
    comb = (mix[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4).softmax(-1) + spec.mixing_eps
    comb = comb / (comb.sum(-2, keepdim=True) + spec.mixing_eps)
    for _ in range(spec.sinkhorn_iterations - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + spec.mixing_eps)
        comb = comb / (comb.sum(-2, keepdim=True) + spec.mixing_eps)
    return (pre[..., None] * x.float()).sum(1).bfloat16(), post, comb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hyper_connection_reference_and_graph():
    torch.manual_seed(731)
    spec = HyperConnectionSpec(4096, 4, 7)
    op = prepare_hyper_connection(spec, device_index=0)
    lock_workspace_manager()
    x = torch.randn((7, 4, 4096), device="cuda", dtype=torch.bfloat16)
    x[0].mul_(1e-6)
    weight = torch.randn((24, 4 * 4096), device="cuda") * .01
    scale = torch.tensor([.6, .3, .2], device="cuda")
    base = torch.randn(24, device="cuda")
    layer_output = torch.randn((7, 4096), device="cuda", dtype=torch.bfloat16)
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            _, post, comb = op.pre(x, weight, scale, base)
            op.post(layer_output, x, post, comb)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = op.pre(x, weight, scale, base)
            combined = op.post(layer_output, x, actual[1], actual[2])
        for _ in range(2):
            graph.replay()
            reference = _reference_pre(x, weight, scale, base, spec)
            torch.testing.assert_close(actual[0], reference[0], rtol=1e-2, atol=8e-3)
            for observed, expected in zip(actual[1:], reference[1:]):
                torch.testing.assert_close(observed, expected, rtol=1e-5, atol=1e-6)
            # Explicit contraction uses old-stream -> new-stream orientation.
            expected_post = (reference[1][..., None] * layer_output[:, None].float()
                             + torch.einsum("tij,tid->tjd", reference[2], x.float())).bfloat16()
            torch.testing.assert_close(combined, expected_post, rtol=1e-2, atol=8e-3)
            eager = op.pre(x, weight, scale, base)
            for captured, direct in zip(actual, eager):
                torch.testing.assert_close(captured, direct, rtol=0, atol=0)
            torch.testing.assert_close(combined, op.post(layer_output, x, eager[1], eager[2]), rtol=0, atol=0)
            x.copy_(x.roll(1, 0) * .75)
            base.add_(.125)
            layer_output.neg_()
        empty = op.pre(x[:0], weight, scale, base)
        assert empty[0].shape == (0, 4096)
        assert op.post(layer_output[:0], x[:0], empty[1], empty[2]).shape == (0, 4, 4096)
        graph.reset()
    finally:
        close_workspace_manager()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hyper_connection_same_request_in_single_and_mixed_batches():
    # Changing batch composition must preserve numerical accuracy; cuBLAS
    # reduction order and BF16 midpoint rounding may differ across shapes.
    torch.manual_seed(731)
    single_spec = HyperConnectionSpec(4096, 4, 1)
    mixed_spec = HyperConnectionSpec(4096, 4, 3)
    single = prepare_hyper_connection(single_spec, device_index=0)
    mixed = prepare_hyper_connection(mixed_spec, device_index=0)
    lock_workspace_manager()
    x = torch.randn((3, 4, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((24, 4 * 4096), device="cuda") * .01
    scale = torch.tensor([.6, .3, .2], device="cuda")
    base = torch.randn(24, device="cuda")
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            single.pre(x[:1], weight, scale, base)
            mixed.pre(x, weight, scale, base)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = single.pre(x[:1], weight, scale, base)
        for _ in range(2):
            graph.replay()
            actual = mixed.pre(x, weight, scale, base)
            expected = _reference_pre(x, weight, scale, base, mixed_spec)
            for alone, together in zip(captured, actual):
                tolerance = dict(rtol=1e-2, atol=8e-3) if alone.dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(alone, together[:1], **tolerance)
            torch.testing.assert_close(actual[0], expected[0], rtol=1e-2, atol=8e-3)
            for observed, reference in zip(actual[1:], expected[1:]):
                torch.testing.assert_close(observed, reference, rtol=1e-5, atol=1e-6)
            x.mul_(.75)
        graph.reset()
    finally:
        close_workspace_manager()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hyper_connection_chunked_prefill_matches_whole_prompt():
    # Chunk boundaries must not introduce cross-request mixing or stale scratch.
    torch.manual_seed(731)
    spec = HyperConnectionSpec(4096, 4, 137)
    op = prepare_hyper_connection(spec, device_index=0)
    lock_workspace_manager()
    residual = torch.randn((137, 4, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((24, 4 * 4096), device="cuda") * .01
    scale = torch.tensor([.6, .3, .2], device="cuda")
    base = torch.randn(24, device="cuda")
    layer_output = torch.randn((137, 4096), device="cuda", dtype=torch.bfloat16)
    try:
        whole = op.pre(residual, weight, scale, base)
        chunks = [op.pre(x, weight, scale, base) for x in residual.split(16)]
        for i, expected in enumerate(whole):
            tolerance = dict(rtol=1e-2, atol=8e-3) if expected.dtype == torch.bfloat16 else dict(rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(torch.cat([chunk[i] for chunk in chunks]), expected, **tolerance)
        combined = op.post(layer_output, residual, whole[1], whole[2])
        partial = [op.post(y, x, mixed[1], mixed[2])
                   for y, x, mixed in zip(layer_output.split(16), residual.split(16), chunks)]
        torch.testing.assert_close(torch.cat(partial), combined, rtol=1e-2, atol=8e-3)
        reference = _reference_pre(residual, weight, scale, base, spec)
        torch.testing.assert_close(whole[0], reference[0], rtol=1e-2, atol=8e-3)
        for actual, expected in zip(whole[1:], reference[1:]):
            torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    finally:
        close_workspace_manager()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_hyper_connection_head_chunked_prefill_matches_whole_prompt():
    # Head projection and normalization must remain numerically consistent
    # when the same prompt is processed in different chunks.
    torch.manual_seed(731)
    spec = HyperConnectionHeadSpec(4096, 4, 137)
    op = prepare_hyper_connection_head(spec, device_index=0)
    lock_workspace_manager()
    residual = torch.randn((137, 4, 4096), device="cuda", dtype=torch.bfloat16)
    weight = torch.randn((4, 4 * 4096), device="cuda") * .01
    scale = torch.tensor([.6], device="cuda")
    base = torch.randn(4, device="cuda")
    try:
        whole = op.run(residual, weight, scale, base)
        chunks = [op.run(x, weight, scale, base) for x in residual.split(16)]
        torch.testing.assert_close(torch.cat(chunks), whole, rtol=1e-2, atol=8e-3)
    finally:
        close_workspace_manager()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_hyper_connection_head_fp32_projection_and_graph():
    # Distinct streams and a tiny residual row catch contraction-axis errors
    # and applying RMS normalization after a BF16 intermediate projection.
    torch.manual_seed(731)
    spec = HyperConnectionHeadSpec(4096, 4, 7)
    op = prepare_hyper_connection_head(spec, device_index=0)
    lock_workspace_manager()
    residual = torch.randn((7, 4, 4096), device="cuda", dtype=torch.bfloat16)
    residual[0].mul_(1e-6)
    weight = torch.randn((4, 4 * 4096), device="cuda") * .01
    scale = torch.tensor([.6], device="cuda")
    base = torch.randn(4, device="cuda")
    try:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op.run(residual, weight, scale, base)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = op.run(residual, weight, scale, base)
        for _ in range(2):
            graph.replay()
            x = residual.double()
            mixing = torch.nn.functional.linear(x.flatten(1), weight.double())
            mixing *= torch.rsqrt(x.square().mean((1, 2), keepdim=False)[:, None] + spec.norm_eps)
            gates = torch.sigmoid(mixing * scale.double() + base.double()) + spec.mixing_eps
            expected = (gates[..., None] * x).sum(1).bfloat16()
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=8e-3)
            torch.testing.assert_close(actual, op.run(residual, weight, scale, base), rtol=0, atol=0)
            residual.copy_(residual.roll(1, 1) * .75)
            base.add_(.125)
        assert op.run(residual[:0], weight, scale, base).shape == (0, 4096)
        graph.reset()
    finally:
        close_workspace_manager()
