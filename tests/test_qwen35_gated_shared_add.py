import pytest
import torch

from sparsevllm.operators.gated_shared_add import gated_shared_add


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


@pytest.mark.parametrize("num_tokens", [1, 2, 4, 8, 1024])
def test_gated_shared_add_matches_torch(num_tokens):
    torch.manual_seed(num_tokens)
    routed = torch.randn((num_tokens, 2048), device="cuda", dtype=torch.bfloat16)
    shared = torch.randn_like(routed)
    padded_gate = torch.randn((num_tokens, 257), device="cuda", dtype=torch.bfloat16)
    gate_logits = padded_gate[:, -1:]

    actual = gated_shared_add(routed, shared, gate_logits)
    expected = routed + torch.sigmoid(gate_logits) * shared

    torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.005)
