"""Projection precision and request isolation before low-bit expert quantization."""

import pytest
import torch

from sparsevllm.operators.float32_linear import Float32LinearSpec, prepare_float32_linear
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_float32_projection_is_batch_invariant_and_replays_updated_inputs(dtype):
    # Observed real-model failure: M=16/32 changed route weights by one FP32
    # ULP, crossed expert activation rounding, and changed the generated token.
    close_workspace_manager()
    try:
        torch.manual_seed(731)
        spec = Float32LinearSpec(4096, 256, 128, dtype)
        op = prepare_float32_linear(spec, device_index=0)
        lock_workspace_manager()
        x = torch.randn((128, 4096), device="cuda", dtype=dtype)
        weight = torch.randn((256, 4096), device="cuda", dtype=torch.float32) / 64
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op.run(x[:16], weight)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op.run(x[:16], weight)
        for _ in range(2):
            expected = (x.double() @ weight.double().t()).float()
            graph.replay()
            result = captured.clone()
            torch.testing.assert_close(result, expected[:16], rtol=1e-5, atol=3e-6)
            for rows in (1, 32, 128):
                actual = op.run(x[:rows], weight).clone()
                torch.testing.assert_close(actual, expected[:rows], rtol=1e-5, atol=3e-6)
                torch.testing.assert_close(actual[:min(rows, 16)], result[:min(rows, 16)], rtol=0, atol=0)
            x.mul_(.75)
            weight.mul_(1.25)
        assert op.run(x[:0], weight).shape == (0, 256)
        before = op.run(x[:1], weight).clone()
        with pytest.raises(ValueError, match="capacity"):
            op.run(x.repeat(2, 1), weight)
        torch.testing.assert_close(op.run(x[:1], weight), before, rtol=0, atol=0)
        graph.reset()
    finally:
        close_workspace_manager()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_paired_projections_keep_both_outputs_live_and_replay():
    # A second projection on the ordinary shared lane overwrites the first.
    # Gated compression needs both projections alive when it updates carry.
    close_workspace_manager()
    try:
        torch.manual_seed(731)
        spec = Float32LinearSpec(4096, 1024, 137, num_projections=2)
        op = prepare_float32_linear(spec, device_index=0)
        lock_workspace_manager()
        x = torch.randn((137, 4096), device="cuda", dtype=torch.bfloat16)
        weights = tuple(torch.randn((1024, 4096), device="cuda") / 64 for _ in range(2))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            op.run_many(x[:16], weights)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = op.run_many(x[:16], weights)
        for _ in range(2):
            graph.replay()
            saved = tuple(value.clone() for value in captured)
            whole = tuple(value.clone() for value in op.run_many(x, weights))
            for small, full, weight in zip(saved, whole, weights):
                expected = (x.double() @ weight.double().T).float()
                torch.testing.assert_close(full, expected, rtol=1e-5, atol=4e-6)
                torch.testing.assert_close(small, full[:16], rtol=0, atol=0)
            x.mul_(.75)
            weights[0].mul_(1.25)
            weights[1].neg_()
        assert all(value.shape == (0, 1024) for value in op.run_many(x[:0], weights))
        with pytest.raises(ValueError, match="weights"):
            op.run_many(x, weights[:1])
        graph.reset()
    finally:
        close_workspace_manager()
