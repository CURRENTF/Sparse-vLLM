import pytest
import torch
import torch.nn.functional as F

from sparsevllm.kernels.external.vllm_moe import clipped_swiglu_op, sqrt_softplus_op
from sparsevllm.kernels.external.support import ExternalKernelContractError
from sparsevllm.operators.moe_router import MoeRouterOpSpec, VllmSqrtSoftplusRouterProvider
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager
from sparsevllm.platforms import current_platform


@pytest.mark.parametrize("load_op", [sqrt_softplus_op, clipped_swiglu_op])
def test_absent_optional_library_does_not_import_engine(monkeypatch, load_op):
    import sparsevllm.kernels.external.vllm_moe as adapter
    load_op.cache_clear()
    try:
        monkeypatch.delenv("SPARSEVLLM_VLLM_MOE_LIBRARY", raising=False)
        monkeypatch.setattr(adapter.importlib.util, "find_spec", lambda name: None)
        assert load_op() is None
    finally:
        load_op.cache_clear()


@pytest.mark.parametrize("load_op", [sqrt_softplus_op, clipped_swiglu_op])
def test_explicit_missing_library_fails_instead_of_falling_back(monkeypatch, tmp_path, load_op):
    load_op.cache_clear()
    try:
        monkeypatch.setenv("SPARSEVLLM_VLLM_MOE_LIBRARY", str(tmp_path / "missing.so"))
        with pytest.raises(ExternalKernelContractError, match="does not exist"):
            load_op()
    finally:
        load_op.cache_clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_learned_router_matches_fp64_oracle_and_live_graph_inputs():
    # Protect bias-only selection, unbiased normalized weights, FP32 scaling,
    # and graph replay after both logits and correction bias change.
    if sqrt_softplus_op() is None:
        pytest.skip("optional vLLM MoE library is not configured")
    close_workspace_manager()
    try:
        torch.manual_seed(731)
        spec = MoeRouterOpSpec(256, 6, torch.float32, True, True, "sqrt_softplus", 33)
        op = VllmSqrtSoftplusRouterProvider.bind(spec, current_platform.get_device_caps(0))
        lock_workspace_manager()
        bias = torch.randn(256, device="cuda") * .1
        logits = torch.randn(33, 256, device="cuda") * 3
        for rows in (1, 7, 33):
            def run():
                return op.run(spec, logits[:rows], bias, routed_scaling_factor=1.5)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                weights, ids = run()
            for _ in range(2):
                logits.mul_(.75)
                bias.neg_()
                graph.replay()
                scores = F.softplus(logits[:rows].double()).sqrt()
                expected_ids = (scores + bias.double()).topk(6, dim=-1).indices
                expected = scores.gather(1, expected_ids)
                expected = expected / expected.sum(-1, keepdim=True) * 1.5
                torch.testing.assert_close(ids.long(), expected_ids, rtol=0, atol=0)
                torch.testing.assert_close(weights.double(), expected, rtol=1e-5, atol=1e-6)
            graph.reset()
        assert op.run(spec, logits[:0], bias)[0].shape == (0, 6)
        with pytest.raises(ValueError, match="capacity"):
            op.run(spec, logits.repeat(2, 1), bias)
        # Upstream may flush tiny scores to zero. Whatever its underflow
        # convention, routing must remain finite and bias must still select
        # experts; NaN weights would corrupt the entire expert contribution.
        logits.fill_(-99.)
        weights, ids = op.run(spec, logits, bias)
        expected_ids = bias.expand_as(logits).topk(6, dim=-1).indices
        torch.testing.assert_close(ids.long(), expected_ids, rtol=0, atol=0)
        assert torch.isfinite(weights).all() and (weights >= 0).all()
        assert (weights.sum(-1) <= 1. + 1e-6).all()
    finally:
        close_workspace_manager()
