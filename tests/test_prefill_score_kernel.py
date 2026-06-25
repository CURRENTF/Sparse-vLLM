import unittest

import torch


def _prefill_score_baseline(
    q,
    k_cache,
    req_to_tokens,
    b_req_idx,
    b_start_loc,
    context_lens,
    prompt_cache_lens,
    score_starts,
    score_ends,
    candidate_start,
    num_recent_tokens,
):
    batch = int(context_lens.numel())
    num_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[1])
    kv_group = num_heads // num_kv_heads
    max_len = int(req_to_tokens.shape[1])
    out = torch.zeros((batch, max_len), dtype=torch.float32, device=q.device)
    for b in range(batch):
        q_start = int(b_start_loc[b].item())
        prompt_cache_len = int(prompt_cache_lens[b].item())
        context_len = int(context_lens[b].item())
        score_start = int(score_starts[b].item())
        score_end = int(score_ends[b].item())
        req_row = int(b_req_idx[b].item())
        cand_start = int(candidate_start)
        cand_end = max(cand_start, context_len - int(num_recent_tokens))
        q_scores = []
        for h in range(num_heads):
            kv_h = h // kv_group
            head_scores = torch.zeros((max_len,), dtype=torch.float32, device=q.device)
            for q_pos in range(score_start, score_end):
                if q_pos < prompt_cache_len or q_pos >= context_len:
                    continue
                q_vec = q[q_start + q_pos - prompt_cache_len, h].float()
                logits = []
                positions = []
                for k_pos in range(cand_start, cand_end):
                    if q_pos < k_pos:
                        continue
                    slot = int(req_to_tokens[req_row, k_pos].item())
                    logits.append(torch.dot(q_vec, k_cache[slot, kv_h].float()) * (q.shape[-1] ** -0.5))
                    positions.append(k_pos)
                if not logits:
                    continue
                probs = torch.softmax(torch.stack(logits), dim=-1)
                for pos, prob in zip(positions, probs):
                    head_scores[pos] += prob / max(1, score_end - score_start)
            q_scores.append(head_scores)
        if q_scores:
            out[b] = torch.stack(q_scores).max(dim=0).values
    return out


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for prefill score Triton tests.")
class PrefillScoreKernelTest(unittest.TestCase):
    def test_prefill_score_matches_torch_for_query_range(self):
        from sparsevllm.triton_kernel.prefill_score import prefill_score_fwd

        torch.manual_seed(7)
        device = "cuda"
        dtype = torch.float32
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        context_lens = torch.tensor([12, 8], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor([7, 5], dtype=torch.int32, device=device)
        chunk_lens = context_lens - prompt_cache_lens
        b_start_loc = torch.tensor([0, int(chunk_lens[0].item())], dtype=torch.int32, device=device)
        max_len = int(context_lens.max().item())
        req_to_tokens = torch.full((3, max_len), -1, dtype=torch.int32, device=device)
        req_to_tokens[0, :8] = torch.arange(20, 28, dtype=torch.int32, device=device)
        req_to_tokens[1, :12] = torch.arange(0, 12, dtype=torch.int32, device=device)
        b_req_idx = torch.tensor([1, 0], dtype=torch.int32, device=device)
        score_starts = torch.tensor([8, 6], dtype=torch.int32, device=device)
        score_ends = torch.tensor([11, 8], dtype=torch.int32, device=device)
        candidate_start = 1
        num_recent_tokens = 2

        total_q = int(chunk_lens.sum().item())
        q = torch.randn((total_q, num_heads, head_dim), dtype=dtype, device=device)
        k_cache = torch.randn((32, num_kv_heads, head_dim), dtype=dtype, device=device)
        attn_score = torch.zeros((2, max_len), dtype=torch.float32, device=device)

        prefill_score_fwd(
            q,
            k_cache,
            attn_score,
            b_req_idx,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            int(chunk_lens.max().item()),
            req_to_tokens,
            score_starts,
            score_ends,
            candidate_start=candidate_start,
            num_recent_tokens=num_recent_tokens,
        )
        torch.cuda.synchronize()

        expected = _prefill_score_baseline(
            q,
            k_cache,
            req_to_tokens,
            b_req_idx,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            score_starts,
            score_ends,
            candidate_start,
            num_recent_tokens,
        )
        torch.testing.assert_close(attn_score, expected, rtol=2e-2, atol=2e-2)

    def test_prefill_score_handles_offset_query_window(self):
        from sparsevllm.triton_kernel.prefill_score import prefill_score_fwd

        torch.manual_seed(11)
        device = "cuda"
        dtype = torch.float32
        num_heads = 3
        num_kv_heads = 1
        head_dim = 16
        prompt_len = 14
        prompt_cache_len = 9
        chunk_len = 5
        score_start = 10
        score_end = prompt_len
        req_to_tokens = torch.arange(0, prompt_len, dtype=torch.int32, device=device).unsqueeze(0)
        b_req_idx = torch.tensor([0], dtype=torch.int32, device=device)
        q = torch.randn((chunk_len, num_heads, head_dim), dtype=dtype, device=device)
        k_cache = torch.randn((prompt_len, num_kv_heads, head_dim), dtype=dtype, device=device)
        acc = torch.zeros((1, prompt_len), dtype=torch.float32, device=device)
        context_lens = torch.tensor([prompt_len], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor([prompt_cache_len], dtype=torch.int32, device=device)
        b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
        score_starts = torch.tensor([score_start], dtype=torch.int32, device=device)
        score_ends = torch.tensor([score_end], dtype=torch.int32, device=device)
        candidate_start = 2
        num_recent_tokens = 4
        prefill_score_fwd(
            q,
            k_cache,
            acc,
            b_req_idx,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            chunk_len,
            req_to_tokens,
            score_starts,
            score_ends,
            candidate_start=candidate_start,
            num_recent_tokens=num_recent_tokens,
        )
        torch.cuda.synchronize()

        expected = _prefill_score_baseline(
            q,
            k_cache,
            req_to_tokens,
            b_req_idx,
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([prompt_len], dtype=torch.int32, device=device),
            torch.tensor([prompt_cache_len], dtype=torch.int32, device=device),
            score_starts,
            score_ends,
            candidate_start,
            num_recent_tokens,
        )
        torch.testing.assert_close(acc, expected, rtol=2e-2, atol=2e-2)

    def test_prefill_score_matches_torch_for_gqa_seven_heads(self):
        from sparsevllm.triton_kernel.prefill_score import prefill_score_fwd

        torch.manual_seed(17)
        device = "cuda"
        dtype = torch.bfloat16
        num_heads = 28
        num_kv_heads = 4
        head_dim = 32
        prompt_len = 96
        window = 32
        prompt_cache_len = prompt_len - window
        req_to_tokens = torch.arange(0, prompt_len, dtype=torch.int32, device=device).unsqueeze(0)
        b_req_idx = torch.tensor([0], dtype=torch.int32, device=device)
        q = torch.randn((window, num_heads, head_dim), dtype=dtype, device=device)
        k_cache = torch.randn((prompt_len, num_kv_heads, head_dim), dtype=dtype, device=device)
        acc = torch.zeros((1, prompt_len), dtype=torch.float32, device=device)
        context_lens = torch.tensor([prompt_len], dtype=torch.int32, device=device)
        prompt_cache_lens = torch.tensor([prompt_cache_len], dtype=torch.int32, device=device)
        b_start_loc = torch.tensor([0], dtype=torch.int32, device=device)
        score_starts = torch.tensor([prompt_cache_len], dtype=torch.int32, device=device)
        score_ends = torch.tensor([prompt_len], dtype=torch.int32, device=device)
        candidate_start = 3
        num_recent_tokens = 9

        prefill_score_fwd(
            q,
            k_cache,
            acc,
            b_req_idx,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            window,
            req_to_tokens,
            score_starts,
            score_ends,
            candidate_start=candidate_start,
            num_recent_tokens=num_recent_tokens,
        )
        torch.cuda.synchronize()

        expected = _prefill_score_baseline(
            q,
            k_cache,
            req_to_tokens,
            b_req_idx,
            b_start_loc,
            context_lens,
            prompt_cache_lens,
            score_starts,
            score_ends,
            candidate_start,
            num_recent_tokens,
        )
        torch.testing.assert_close(acc, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
