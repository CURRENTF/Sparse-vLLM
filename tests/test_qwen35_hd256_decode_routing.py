import unittest
from unittest.mock import patch

import torch

from sparsevllm.engine.cache_manager import DecodeComputeView
from sparsevllm.layers.attention_backend import TritonAttentionBackend


class Qwen35Hd256DecodeRoutingTest(unittest.TestCase):
    def _make_view(self, *, head_dim: int, attn_score=None):
        active_slots = torch.tensor([[0, 1, 2]], dtype=torch.int32)
        req_indices = torch.tensor([0], dtype=torch.int32)
        context_lens = torch.tensor([3], dtype=torch.int32)
        k_cache = torch.zeros(4, 4, head_dim, dtype=torch.float32)
        v_cache = torch.zeros_like(k_cache)
        return DecodeComputeView(
            k_cache=k_cache,
            v_cache=v_cache,
            active_slots=active_slots,
            req_indices=req_indices,
            context_lens=context_lens,
            attn_score=attn_score,
            max_context_len=3,
        )

    def test_head_dim_256_uses_hd256_gqa_decode_kernels(self):
        q = torch.zeros(1, 16, 256)
        view = self._make_view(head_dim=256)
        mid_o = torch.empty(1, 16, 1, 256)
        mid_lse = torch.empty(1, 16, 1)
        calls = {"stage1": 0, "stage2": 0}

        def stage1_hd256(*args, **kwargs):
            calls["stage1"] += 1

        def stage2_hd256(mid_o, mid_o_logsumexp, context_lens, o, block_seq):
            calls["stage2"] += 1
            o.fill_(7.0)

        with (
            patch("sparsevllm.layers.attention_backend.gqa_flash_decode_stage1_hd256", side_effect=stage1_hd256),
            patch("sparsevllm.layers.attention_backend.flash_decode_stage2_hd256", side_effect=stage2_hd256),
            patch(
                "sparsevllm.layers.attention_backend.gqa_flash_decode_stage1",
                side_effect=AssertionError("old gqa called"),
            ),
            patch("sparsevllm.layers.attention_backend.flash_decode_stage2", side_effect=AssertionError("old stage2 called")),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=mid_o,
                mid_o_logexpsum=mid_lse,
                max_len_in_batch=3,
                block_seq=256,
                num_heads=16,
                num_kv_heads=4,
            )

        self.assertEqual(calls, {"stage1": 1, "stage2": 1})
        self.assertTrue(torch.equal(out, torch.full_like(out, 7.0)))

    def test_head_dim_256_with_score_uses_hd256_score_kernel(self):
        attn_score = torch.zeros(1, 16, 3)
        q = torch.zeros(1, 16, 256)
        view = self._make_view(head_dim=256, attn_score=attn_score)
        mid_o = torch.empty(1, 16, 1, 256)
        mid_lse = torch.empty(1, 16, 1)
        calls = {"stage1_score": 0, "stage2": 0}

        def stage1_hd256_with_score(*args, **kwargs):
            calls["stage1_score"] += 1

        def stage2_hd256(mid_o, mid_o_logsumexp, context_lens, o, block_seq):
            calls["stage2"] += 1
            o.fill_(11.0)

        with (
            patch(
                "sparsevllm.layers.attention_backend.gqa_flash_decode_stage1_hd256_with_score",
                side_effect=stage1_hd256_with_score,
            ),
            patch("sparsevllm.layers.attention_backend.flash_decode_stage2_hd256", side_effect=stage2_hd256),
            patch(
                "sparsevllm.layers.attention_backend.gqa_flash_decode_stage1_with_score",
                side_effect=AssertionError("old score called"),
            ),
            patch("sparsevllm.layers.attention_backend.flash_decode_stage2", side_effect=AssertionError("old stage2 called")),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=mid_o,
                mid_o_logexpsum=mid_lse,
                max_len_in_batch=3,
                block_seq=256,
                num_heads=16,
                num_kv_heads=4,
            )

        self.assertEqual(calls, {"stage1_score": 1, "stage2": 1})
        self.assertTrue(torch.equal(out, torch.full_like(out, 11.0)))

    def test_head_dim_128_keeps_existing_gqa_decode_kernels(self):
        q = torch.zeros(1, 16, 128)
        view = self._make_view(head_dim=128)
        mid_o = torch.empty(1, 16, 1, 128)
        mid_lse = torch.empty(1, 16, 1)
        calls = {"stage1": 0, "stage2": 0}

        def stage1(*args, **kwargs):
            calls["stage1"] += 1

        def stage2(mid_o, mid_o_logsumexp, context_lens, o, block_seq):
            calls["stage2"] += 1
            o.fill_(3.0)

        with (
            patch("sparsevllm.layers.attention_backend.gqa_flash_decode_stage1", side_effect=stage1),
            patch("sparsevllm.layers.attention_backend.flash_decode_stage2", side_effect=stage2),
            patch(
                "sparsevllm.layers.attention_backend.gqa_flash_decode_stage1_hd256",
                side_effect=AssertionError("hd256 called"),
            ),
            patch(
                "sparsevllm.layers.attention_backend.flash_decode_stage2_hd256",
                side_effect=AssertionError("hd256 stage2 called"),
            ),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=mid_o,
                mid_o_logexpsum=mid_lse,
                max_len_in_batch=3,
                block_seq=256,
                num_heads=16,
                num_kv_heads=4,
            )

        self.assertEqual(calls, {"stage1": 1, "stage2": 1})
        self.assertTrue(torch.equal(out, torch.full_like(out, 3.0)))

    def test_head_dim_128_with_score_keeps_existing_gqa_decode_kernels(self):
        attn_score = torch.zeros(1, 16, 3)
        q = torch.zeros(1, 16, 128)
        view = self._make_view(head_dim=128, attn_score=attn_score)
        mid_o = torch.empty(1, 16, 1, 128)
        mid_lse = torch.empty(1, 16, 1)
        calls = {"stage1_score": 0, "stage2": 0}

        def stage1_with_score(*args, **kwargs):
            calls["stage1_score"] += 1

        def stage2(mid_o, mid_o_logsumexp, context_lens, o, block_seq):
            calls["stage2"] += 1
            o.fill_(5.0)

        with (
            patch("sparsevllm.layers.attention_backend.gqa_flash_decode_stage1_with_score", side_effect=stage1_with_score),
            patch("sparsevllm.layers.attention_backend.flash_decode_stage2", side_effect=stage2),
            patch(
                "sparsevllm.layers.attention_backend.gqa_flash_decode_stage1_hd256_with_score",
                side_effect=AssertionError("hd256 score called"),
            ),
            patch(
                "sparsevllm.layers.attention_backend.flash_decode_stage2_hd256",
                side_effect=AssertionError("hd256 stage2 called"),
            ),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=mid_o,
                mid_o_logexpsum=mid_lse,
                max_len_in_batch=3,
                block_seq=256,
                num_heads=16,
                num_kv_heads=4,
            )

        self.assertEqual(calls, {"stage1_score": 1, "stage2": 1})
        self.assertTrue(torch.equal(out, torch.full_like(out, 5.0)))


if __name__ == "__main__":
    unittest.main()
