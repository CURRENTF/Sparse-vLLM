import os
import unittest
from unittest.mock import patch

import torch

from sparsevllm.engine.cache_manager import DecodeComputeView, PrefillComputeView
from sparsevllm.layers.attention_backend import TritonAttentionBackend


class FakeAttentionBackendTest(unittest.TestCase):
    def setUp(self):
        self._old_enabled = os.environ.get("SPARSEVLLM_FAKE_ATTENTION")
        self._old_mode = os.environ.get("SPARSEVLLM_FAKE_ATTENTION_MODE")

    def tearDown(self):
        if self._old_enabled is None:
            os.environ.pop("SPARSEVLLM_FAKE_ATTENTION", None)
        else:
            os.environ["SPARSEVLLM_FAKE_ATTENTION"] = self._old_enabled
        if self._old_mode is None:
            os.environ.pop("SPARSEVLLM_FAKE_ATTENTION_MODE", None)
        else:
            os.environ["SPARSEVLLM_FAKE_ATTENTION_MODE"] = self._old_mode

    def _make_prefill_view(self, *, attn_score=None):
        return PrefillComputeView(
            k_cache=torch.ones(8, 2, 4),
            v_cache=torch.ones(8, 2, 4),
            active_slots=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([3], dtype=torch.int32),
            attn_score=attn_score,
            max_context_len=3,
        )

    def _make_decode_view(self, *, attn_score=None):
        return DecodeComputeView(
            k_cache=torch.ones(8, 2, 4),
            v_cache=torch.ones(8, 2, 4),
            active_slots=torch.tensor([[0, 1, 2]], dtype=torch.int32),
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([3], dtype=torch.int32),
            attn_score=attn_score,
            max_context_len=3,
        )

    def test_fake_prefill_returns_zeros_and_skips_kernel(self):
        os.environ["SPARSEVLLM_FAKE_ATTENTION"] = "1"
        q = torch.ones(3, 2, 4)
        attn_score = torch.full((1, 2, 3), 9.0)
        view = self._make_prefill_view(attn_score=attn_score)

        with patch(
            "sparsevllm.layers.attention_backend.context_attention_fwd",
            side_effect=AssertionError("real prefill kernel called"),
        ):
            out = TritonAttentionBackend().run_prefill(
                q,
                view,
                b_start_loc=torch.tensor([0], dtype=torch.int32),
                chunk_lens=torch.tensor([3], dtype=torch.int32),
                max_input_len=3,
            )

        self.assertTrue(torch.equal(out, torch.zeros_like(q)))
        self.assertTrue(torch.equal(attn_score, torch.zeros_like(attn_score)))

    def test_fake_decode_copy_mode_skips_kernels(self):
        os.environ["SPARSEVLLM_FAKE_ATTENTION"] = "1"
        os.environ["SPARSEVLLM_FAKE_ATTENTION_MODE"] = "copy"
        q = torch.arange(8, dtype=torch.float32).view(1, 2, 4)
        attn_score = torch.full((1, 2, 3), -1e20)
        view = self._make_decode_view(attn_score=attn_score)

        with (
            patch(
                "sparsevllm.layers.attention_backend.gqa_flash_decode_stage1",
                side_effect=AssertionError("real decode stage1 called"),
            ),
            patch(
                "sparsevllm.layers.attention_backend.flash_decode_stage2",
                side_effect=AssertionError("real decode stage2 called"),
            ),
        ):
            out = TritonAttentionBackend().run_decode(
                q,
                view,
                mid_o=torch.empty(1, 2, 1, 4),
                mid_o_logexpsum=torch.empty(1, 2, 1),
                max_len_in_batch=3,
                block_seq=256,
                num_heads=2,
                num_kv_heads=1,
            )

        self.assertIsNot(out, q)
        self.assertTrue(torch.equal(out, q))
        self.assertTrue(torch.equal(attn_score, torch.zeros_like(attn_score)))


if __name__ == "__main__":
    unittest.main()
