import pytest
import torch

from sparsevllm.kernels.triton.qwen3_5.fla.ops.l2norm import (
    fused_qk_l2norm_fwd,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required",
)


@pytest.mark.parametrize(
    ("tokens", "heads", "head_dim"),
    [(1, 8, 128), (37, 5, 64), (128, 16, 128), (31, 8, 256)],
)
def test_fused_qk_l2norm_matches_torch_for_strided_projection_views(
    tokens,
    heads,
    head_dim,
):
    torch.manual_seed(tokens + heads + head_dim)
    width = heads * head_dim
    packed = torch.randn(
        (tokens, width * 3), dtype=torch.bfloat16, device="cuda"
    )
    q = packed[:, :width].view(1, tokens, heads, head_dim)
    k = packed[:, width : 2 * width].view(1, tokens, heads, head_dim)
    if tokens > 1:
        assert q.stride(1) == 3 * width
        assert k.stride(1) == 3 * width

    actual_q, actual_k = fused_qk_l2norm_fwd(q, k)

    expected = []
    for value in (q.squeeze(0).float(), k.squeeze(0).float()):
        expected.append(
            value / torch.sqrt(torch.sum(value * value, dim=-1, keepdim=True) + 1e-6)
        )
    assert actual_q.is_contiguous() and actual_k.is_contiguous()
    torch.testing.assert_close(actual_q.float(), expected[0], rtol=1e-2, atol=2e-3)
    torch.testing.assert_close(actual_k.float(), expected[1], rtol=1e-2, atol=2e-3)
