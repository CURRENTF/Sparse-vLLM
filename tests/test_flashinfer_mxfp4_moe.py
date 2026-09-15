import pytest
import torch
import torch.nn.functional as F
from types import SimpleNamespace

from sparsevllm.operators.mxfp4_moe import FlashInferMxfp4MoeProvider, Mxfp4MoeSpec
from sparsevllm.operators.mxfp4_marlin import VllmMarlinMxfp4MoeProvider
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


def test_unsupported_partition_rejects_before_loading_external_library(monkeypatch):
    from sparsevllm.kernels.external.flashinfer import mxfp4_moe
    monkeypatch.setattr(mxfp4_moe, "mxfp4_moe_ops", lambda: pytest.fail("must reject before importing kernels"))
    caps = SimpleNamespace(platform=PlatformEnum.CUDA, compute_capability=(9, 0),
                           supports_bfloat16=True, supports_graph_capture=True)
    spec = Mxfp4MoeSpec(256, 128, 8, 3, 1, 2, 7, 10.)
    assert not FlashInferMxfp4MoeProvider.supports(spec, caps).supported


def test_broken_legal_external_dependency_is_not_silently_rejected(monkeypatch):
    from sparsevllm.kernels.external.flashinfer import mxfp4_moe
    from sparsevllm.kernels.external.support import ExternalKernelContractError
    def fail():
        raise ExternalKernelContractError("flashinfer-python", "MXFP4", "missing workspace API")
    monkeypatch.setattr(mxfp4_moe, "mxfp4_moe_ops", fail)
    caps = SimpleNamespace(platform=PlatformEnum.CUDA, compute_capability=(9, 0),
                           supports_bfloat16=True, supports_graph_capture=True)
    spec = Mxfp4MoeSpec(256, 128, 8, 4, 0, 2, 7, 10.)
    with pytest.raises(ExternalKernelContractError, match="workspace"):
        FlashInferMxfp4MoeProvider.supports(spec, caps)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("provider_type", [FlashInferMxfp4MoeProvider, VllmMarlinMxfp4MoeProvider])
def test_checkpoint_packing_partition_and_live_routes_in_graph(rank, provider_type):
    # Catch per-projection packing/order bugs and stale routing or output data
    # when a graph changes from local work to a wholly remote expert set.
    if provider_type is FlashInferMxfp4MoeProvider and torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("FlashInfer W4A16 requires SM90")
    close_workspace_manager()
    try:
        torch.manual_seed(731)
        rows, hidden, intermediate, local = 7, 256, 128, 4
        spec = Mxfp4MoeSpec(hidden, intermediate, 8, local, rank * local, 2, rows, 10.)
        caps = current_platform.get_device_caps(0)
        support = provider_type.supports(spec, caps)
        if not support.supported:
            pytest.skip(support.reason)
        provider = provider_type.bind(spec, caps)
        storage = provider.allocate_weights()
        lut = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], device="cuda")
        logical = []
        for name, n, k in [("gate", intermediate, hidden), ("up", intermediate, hidden),
                           ("down", hidden, intermediate)]:
            w = torch.randint(0, 256, (local, n, k // 2), device="cuda", dtype=torch.uint8)
            s = torch.randint(120, 124, (local, n, k // 32), device="cuda", dtype=torch.uint8)
            dense = lut[torch.stack((w & 15, w >> 4), -1).long()].flatten(-2)
            logical.append(dense * torch.exp2(s.float() - 127).repeat_interleave(32, -1))
            for expert in range(local):
                provider.load_projection(storage, expert, name, w[expert].cpu(), s[expert].cpu())
        lock_workspace_manager()
        x = torch.randn(rows, hidden, device="cuda", dtype=torch.bfloat16) * 3
        ids = torch.rand(rows, 8, device="cuda").argsort(-1)[:, :2].contiguous()
        weights = torch.rand(rows, 2, device="cuda", dtype=torch.float32)

        def run():
            return provider.run(x, ids, weights, storage)

        def oracle():
            out = torch.zeros_like(x, dtype=torch.float32)
            for e in range(local):
                tokens, routes = torch.where(ids == e + rank * local)
                if not len(tokens):
                    continue
                g = (x[tokens].float() @ logical[0][e].T).bfloat16().float().clamp(max=10.)
                u = (x[tokens].float() @ logical[1][e].T).bfloat16().float().clamp(-10., 10.)
                y = (F.silu(g) * u).bfloat16().float() @ logical[2][e].T
                out.index_add_(0, tokens, y * weights[tokens, routes, None])
            return out

        assert provider.run(x[:0], ids[:0], weights[:0], storage).shape == (0, hidden)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = run()
        for _ in range(2):
            x.mul_(.75)
            ids.add_(1).remainder_(8)
            weights.mul_(.8)
            graph.replay()
            expected = oracle()
            assert captured.dtype == torch.float32
            assert torch.isfinite(captured).all()
            assert (captured - expected).norm() / expected.norm() < .01
            torch.testing.assert_close(captured, run(), rtol=0, atol=0)
        ids.fill_((1 - rank) * local)
        graph.replay()
        torch.testing.assert_close(captured, torch.zeros_like(captured), rtol=0, atol=0)
        ids.fill_(-1)
        graph.replay()
        torch.testing.assert_close(captured, torch.zeros_like(captured), rtol=0, atol=0)
        with pytest.raises(ValueError, match="capacity"):
            provider.run(x.repeat(2, 1), ids.repeat(2, 1), weights.repeat(2, 1), storage)
    finally:
        close_workspace_manager()
