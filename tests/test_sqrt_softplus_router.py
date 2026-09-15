import pytest
import torch
import torch.nn.functional as F

from sparsevllm.operators.moe_router import MoeRouterOpSpec, resolve_moe_router_provider
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("use_hash", [False, True])
def test_sqrt_softplus_router_matches_reference_and_replays_changed_metadata(use_hash):
    close_workspace_manager()
    try:
        torch.manual_seed(731)
        rows, experts, top_k, vocab = 7, 256, 6, 37
        spec = MoeRouterOpSpec(experts, top_k, torch.float32, True, True,
                               "hash_sqrt_softplus" if use_hash else "sqrt_softplus", rows + 5)
        provider = resolve_moe_router_provider(spec, device_index=0)
        logits = torch.randn((rows, experts), device="cuda", dtype=torch.float32) * 3
        logits[0].zero_()
        logits[1].fill_(-99.)
        logits[2] = torch.linspace(-30, 30, experts, device="cuda")
        bias = torch.zeros(experts, device="cuda", dtype=torch.float32)
        table = torch.rand((vocab, experts), device="cuda").argsort(-1)[:, :top_k].int().contiguous()
        tokens = torch.tensor([0, 7, 9, -1, 3, 36, 1], device="cuda", dtype=torch.int64)
        lock_workspace_manager()

        def forward():
            return provider.run(spec, logits, None if use_hash else bias, routed_scaling_factor=1.5,
                                hash_indices=table if use_hash else None, input_ids=tokens if use_hash else None)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            forward()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_weights, captured_ids = forward()
        for step in range(2):
            if step:
                logits[1].normal_()
                bias.normal_()
                tokens[0] = 36
                table.copy_(table.roll(1, 0))
            graph.replay()
            scores = F.softplus(logits).sqrt()
            expected_ids = table[tokens.clamp_min(0)].long() if use_hash else (scores + bias).topk(top_k, dim=-1).indices
            expected_weights = scores.gather(1, expected_ids)
            expected_weights = expected_weights / expected_weights.sum(-1, keepdim=True) * 1.5
            if use_hash:
                expected_ids[tokens < 0] = -1
                expected_weights[tokens < 0] = 0
            torch.testing.assert_close(captured_ids, expected_ids, rtol=0, atol=0)
            torch.testing.assert_close(captured_weights, expected_weights, rtol=1e-6, atol=1e-7)
            snapshot = captured_weights.clone()
            torch.testing.assert_close(forward()[0], snapshot, rtol=0, atol=0)
    finally:
        close_workspace_manager()
