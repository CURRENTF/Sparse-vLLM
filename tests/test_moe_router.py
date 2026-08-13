from __future__ import annotations

import pytest
import torch

from sparsevllm.operators.moe_router import (
    GlmBiasedSigmoidRouterProvider,
    MoeRouterOpSpec,
)


def _router():
    spec = MoeRouterOpSpec(64, 4, torch.float32, True, True, "biased_sigmoid")
    return spec, GlmBiasedSigmoidRouterProvider()


def _reference_routes(
    logits: torch.Tensor,
    correction_bias: torch.Tensor,
    scaling: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    routing_weights = torch.sigmoid(logits)
    ids = torch.topk(
        routing_weights + correction_bias,
        4,
        dim=-1,
        sorted=False,
    ).indices
    weights = routing_weights.gather(1, ids)
    weights /= weights.sum(dim=-1, keepdim=True) + 1e-20
    return weights * scaling, ids


def _assert_same_routes(
    actual_weights: torch.Tensor,
    actual_ids: torch.Tensor,
    reference_weights: torch.Tensor,
    reference_ids: torch.Tensor,
) -> None:
    reference_order = reference_ids.argsort(dim=-1)
    actual_order = actual_ids.argsort(dim=-1)
    assert torch.equal(
        actual_ids.gather(1, actual_order),
        reference_ids.gather(1, reference_order),
    )
    torch.testing.assert_close(
        actual_weights.gather(1, actual_order),
        reference_weights.gather(1, reference_order),
        rtol=2e-6,
        atol=2e-7,
        equal_nan=True,
    )


def test_glm_router_rejects_wrong_shape() -> None:
    spec = MoeRouterOpSpec(32, 4, torch.float32, True, True, "biased_sigmoid")
    from sparsevllm.platforms import DeviceCaps, PlatformEnum

    caps = DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="test",
        supports_triton=True,
    )
    assert not GlmBiasedSigmoidRouterProvider.supports(spec, caps).supported


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 32])
def test_glm_router_matches_reference_and_replays_updated_graph(
    num_tokens: int,
) -> None:
    torch.manual_seed(20260810 + num_tokens)
    logits = (
        torch.randn(num_tokens, 64, dtype=torch.float32, device="cuda") * 3
    )
    correction_bias = (
        torch.randn(64, dtype=torch.float32, device="cuda") * 0.1
    )
    scaling = 1.8
    reference_weights, reference_ids = _reference_routes(
        logits,
        correction_bias,
        scaling,
    )

    spec, router = _router()
    actual_weights, actual_ids = router.run(
        spec,
        logits,
        correction_bias,
        routed_scaling_factor=scaling,
    )
    _assert_same_routes(
        actual_weights,
        actual_ids,
        reference_weights,
        reference_ids,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_weights, graph_ids = router.run(
            spec,
            logits,
            correction_bias,
            routed_scaling_factor=scaling,
        )

    logits.copy_(torch.randn_like(logits) * 4)
    correction_bias.copy_(torch.randn_like(correction_bias) * 0.2)
    replay_reference_weights, replay_reference_ids = _reference_routes(
        logits,
        correction_bias,
        scaling,
    )
    graph.replay()
    torch.cuda.synchronize()
    _assert_same_routes(
        graph_weights,
        graph_ids,
        replay_reference_weights,
        replay_reference_ids,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_glm_router_handles_ties_extremes_and_nonfinite_scores() -> None:
    logits = torch.linspace(
        -80,
        80,
        64,
        dtype=torch.float32,
        device="cuda",
    ).repeat(3, 1)
    correction_bias = torch.zeros(64, dtype=torch.float32, device="cuda")
    logits[0].zero_()
    logits[2, 7] = float("nan")
    scaling = 1.8
    spec, router = _router()
    weights, ids = router.run(
        spec,
        logits,
        correction_bias,
        routed_scaling_factor=scaling,
    )

    assert torch.equal(ids[0].sort().values, torch.arange(4, device="cuda"))
    torch.testing.assert_close(
        weights[0],
        torch.full((4,), scaling / 4, device="cuda"),
        rtol=0,
        atol=0,
    )
    reference_weights, reference_ids = _reference_routes(
        logits[1:],
        correction_bias,
        scaling,
    )
    _assert_same_routes(
        weights[1:],
        ids[1:],
        reference_weights,
        reference_ids,
    )
