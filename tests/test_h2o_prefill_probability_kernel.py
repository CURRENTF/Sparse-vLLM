"""CUDA numerical checks; run on an idle device against independent attention."""

import pytest
import torch

from sparsevllm.kernels.triton.prefill_score import (
    prefill_score_fwd, prefill_score_from_lse_fwd,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires CUDA')


@pytest.mark.parametrize('heads,kv_heads,dim', [(4, 4, 32), (8, 2, 128), (4, 4, 192)])
@pytest.mark.parametrize('query_lengths', [(7, 3), (137, 19)])
@pytest.mark.parametrize('use_lse', [False, True])
def test_head_probability_sums_match_causal_attention(heads, kv_heads, dim, query_lengths, use_lse):
    # MLA uses already projected explicit K for scoring; dimension 192 exercises
    # its non-power-of-two projection, with no persistent KV expansion here.
    torch.manual_seed(91)
    device = torch.device('cuda')
    prefixes = [11, 5]
    contexts = [p + q for p, q in zip(prefixes, query_lengths)]
    width = max(contexts) + 9
    q = torch.randn(sum(query_lengths), heads, dim, device=device, dtype=torch.bfloat16)
    k = torch.randn(2 * width, kv_heads, dim, device=device, dtype=torch.bfloat16)
    slots = torch.randperm(2 * width, device=device).reshape(2, width).int()
    starts = torch.tensor([0, query_lengths[0]], device=device, dtype=torch.int32)
    context_tensor = torch.tensor(contexts, device=device, dtype=torch.int32)
    prefix_tensor = torch.tensor(prefixes, device=device, dtype=torch.int32)
    reqs = torch.arange(2, device=device, dtype=torch.int32)
    expected = torch.zeros(2, heads, width, device=device)
    lse = torch.empty(heads, sum(query_lengths), device=device)
    scale = .07  # Verifies the actual attention scale is carried through.
    for batch, (prefix, query_len, context) in enumerate(zip(prefixes, query_lengths, contexts)):
        q_start = int(starts[batch])
        for head in range(heads):
            keys = k[slots[batch, :context].long(), head // (heads // kv_heads)].float()
            logits = q[q_start:q_start + query_len, head].float() @ keys.T * scale
            mask = torch.arange(context, device=device)[None, :] > (prefix + torch.arange(query_len, device=device))[:, None]
            logits.masked_fill_(mask, -torch.inf)
            expected[batch, head, :context] = logits.softmax(-1).sum(0)
            lse[head, q_start:q_start + query_len] = logits.logsumexp(-1)
    # A strided output checks the public wrapper, not just contiguous kernels.
    backing = torch.full((2, heads, width * 2), float('nan'), device=device)
    output = backing[..., ::2]
    args = (output, reqs, starts, context_tensor, prefix_tensor, max(query_lengths), slots, prefix_tensor, context_tensor)
    for _ in range(2):
        if use_lse:
            prefill_score_from_lse_fwd(q, k, lse, *args, per_head=True, softmax_scale=scale)
        else:
            prefill_score_fwd(q, k, *args, per_head=True, softmax_scale=scale)
        torch.testing.assert_close(output, expected, rtol=5e-3, atol=5e-3)
        torch.testing.assert_close(output.sum(-1), torch.tensor(query_lengths, device=device).float()[:, None].expand(-1, heads), rtol=5e-3, atol=5e-3)
        assert torch.isnan(backing[..., 1::2]).all()


def test_mla_score_uses_projected_nonrope_and_rope_logits():
    torch.manual_seed(27)
    device = torch.device('cuda')
    length, heads, nope_dim, rope_dim, latent_dim = 17, 4, 128, 64, 512
    latent = torch.randn(length, latent_dim, device=device, dtype=torch.bfloat16)
    rope = torch.randn(length, rope_dim, device=device, dtype=torch.bfloat16)
    projection = torch.randn(latent_dim, heads, nope_dim, device=device, dtype=torch.bfloat16) * .04
    expanded_nope = (latent @ projection.flatten(1)).reshape(length, heads, nope_dim)
    q_nope = torch.randn(length, heads, nope_dim, device=device, dtype=torch.bfloat16)
    q_rope = torch.randn(length, heads, rope_dim, device=device, dtype=torch.bfloat16)
    q = torch.cat((q_nope, q_rope), -1)
    k = torch.cat((expanded_nope, rope[:, None, :].expand(-1, heads, -1)), -1)
    expected = torch.empty(heads, length, device=device)
    scale = (nope_dim + rope_dim) ** -.5
    causal_mask = torch.arange(length, device=device)[None, :] > torch.arange(length, device=device)[:, None]
    for head in range(heads):
        # Independent decomposition of the two MLA attention terms.
        logits = (q_nope[:, head].float() @ expanded_nope[:, head].float().T
                  + q_rope[:, head].float() @ rope.float().T) * scale
        expected[head] = logits.masked_fill(causal_mask, -torch.inf).softmax(-1).sum(0)
    output = torch.empty(1, heads, length, device=device)
    zero = torch.zeros(1, device=device, dtype=torch.int32)
    end = torch.tensor([length], device=device, dtype=torch.int32)
    slots = torch.arange(length, device=device, dtype=torch.int32)[None]
    prefill_score_fwd(q, k, output, zero, zero, end, zero, length, slots, zero, end,
                      per_head=True, softmax_scale=scale)
    torch.testing.assert_close(output[0], expected, rtol=5e-3, atol=5e-3)
