import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from sparsevllm.config import Config
from sparsevllm.engine.cache_manager.base import LayerBatchStates
from sparsevllm.engine.cache_manager.rkv import RKVCacheManager
from sparsevllm.engine.cache_manager.skipkv import (
    SkipKVCacheManager,
    SkipKVSentence,
    SkipKVSequenceState,
)
from sparsevllm.engine.activation_controller import ActivationController
from sparsevllm.engine.sequence import Sequence
from sparsevllm.method_registry import (
    get_default_prefill_schedule_policy,
    normalize_sparse_method,
    PREFILL_POLICY_ALL_CHUNKED,
)


class RKVSkipKVMethodTest(unittest.TestCase):
    def _hf_config(self):
        return SimpleNamespace(
            model_type="qwen2",
            torch_dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
        )

    def test_rkv_aliases_and_prefill_policy(self):
        self.assertEqual(normalize_sparse_method("r-kv"), "rkv")
        self.assertEqual(normalize_sparse_method("r_kv"), "rkv")
        self.assertEqual(normalize_sparse_method("skip-kv"), "skipkv")
        self.assertEqual(get_default_prefill_schedule_policy("r-kv"), PREFILL_POLICY_ALL_CHUNKED)
        self.assertEqual(get_default_prefill_schedule_policy("skipkv"), PREFILL_POLICY_ALL_CHUNKED)

    def test_rkv_config_warns_about_approximate_official_adaptation(self):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            with (
                patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=self._hf_config()),
                patch("sparsevllm.config.log_once") as log_once,
            ):
                Config(
                    model=str(model_dir),
                    vllm_sparse_method="rkv",
                    max_model_len=32768,
                    rkv_redundancy_window=64,
                )

        log_once.assert_called()
        warning, level = log_once.call_args.args[0], log_once.call_args.kwargs["level"]
        self.assertEqual(level, "WARNING")
        self.assertIn("approximation of the official implementation", warning)
        self.assertIn("per-KV-head token selection", warning)
        self.assertIn("rkv_redundancy_window=64", warning)

    def test_rkv_redundancy_scoring_fails_fast_when_unbounded(self):
        keys = torch.randn(5, 1, 4)
        with self.assertRaisesRegex(RuntimeError, "rkv_max_redundancy_tokens"):
            RKVCacheManager.redundancy_scores_from_keys(
                keys,
                similarity_threshold=0.8,
                recent_similar_keep=1,
                max_tokens=4,
            )

    def test_rkv_joint_retention_score_uses_paper_lambda(self):
        importance = torch.tensor([0.2, 0.8], dtype=torch.float32)
        redundancy = torch.tensor([0.9, 0.1], dtype=torch.float32)
        score = RKVCacheManager.joint_retention_scores(importance, redundancy, alpha=0.25)

        expected = 0.25 * importance - 0.75 * redundancy
        self.assertTrue(torch.allclose(score, expected))

    def test_skipkv_segment_penalty_marks_older_similar_segment(self):
        keys = torch.tensor(
            [
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[0.0, 1.0]],
            ]
        )
        penalty = SkipKVCacheManager.segment_redundancy_penalty(
            keys,
            segment_size=2,
            similarity_threshold=0.95,
        )
        self.assertGreater(float(penalty[0]), 0.9)
        self.assertGreater(float(penalty[1]), 0.9)
        self.assertEqual(float(penalty[2]), 0.0)
        self.assertEqual(float(penalty[-1]), 0.0)

    def test_skipkv_sentence_scoring_marks_older_redundant_sentence(self):
        manager = object.__new__(SkipKVCacheManager)
        manager.config = SimpleNamespace(
            skipkv_enable_sentence_scoring=True,
            skipkv_similarity_threshold=0.95,
            skipkv_sentence_min_tokens=1,
            skipkv_sentence_max_tokens=16,
            skipkv_max_tracked_sentences=16,
        )
        manager._skipkv_delimiter_token_ids = {99}
        manager._skipkv_non_execution_token_ids = set()
        manager._skipkv_seq_states = {}

        seq = Sequence([1])
        seq.num_prompt_tokens = 0
        for pos, token_id in enumerate([11, 12, 99, 21, 22, 99]):
            seq.num_tokens = pos + 1
            seq.last_token = token_id
            manager.record_skipkv_decode_hidden_states(
                [seq],
                torch.tensor([[1.0, 0.0]]),
            )

        state = manager._skipkv_seq_states[seq.seq_id]
        self.assertEqual(len(state.sentences), 2)
        self.assertGreater(state.sentences[0].redundancy, 0.95)
        self.assertEqual(state.redundant_sentence_count, 1)
        self.assertEqual(state.non_execution_count, 0)

    def test_skipkv_non_execution_marker_counts_completed_sentence(self):
        manager = object.__new__(SkipKVCacheManager)
        manager.config = SimpleNamespace(
            skipkv_enable_sentence_scoring=True,
            skipkv_similarity_threshold=0.95,
            skipkv_sentence_min_tokens=1,
            skipkv_sentence_max_tokens=16,
            skipkv_max_tracked_sentences=16,
        )
        manager._skipkv_delimiter_token_ids = {99}
        manager._skipkv_non_execution_token_ids = {42}
        manager._skipkv_seq_states = {}

        seq = Sequence([1])
        seq.num_prompt_tokens = 0
        for pos, token_id in enumerate([11, 42, 99]):
            seq.num_tokens = pos + 1
            seq.last_token = token_id
            manager.record_skipkv_decode_hidden_states(
                [seq],
                torch.tensor([[1.0, 0.0]]),
            )

        state = manager._skipkv_seq_states[seq.seq_id]
        self.assertEqual(len(state.sentences), 1)
        self.assertEqual(state.non_execution_count, 1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for activation controller buffers")
    def test_skipkv_activation_steering_uses_signed_non_execution_count(self):
        class FakeCacheManager:
            def __init__(self):
                self.delimiters = set()
                self.non_execution_markers = set()

            def set_skipkv_delimiter_token_ids(self, token_ids):
                self.delimiters = set(token_ids)

            def set_skipkv_non_execution_token_ids(self, token_ids):
                self.non_execution_markers = set(token_ids)

            def skipkv_non_execution_count(self, _seq_id):
                return 3

        config = SimpleNamespace(
            vllm_sparse_method="skipkv",
            hf_config=SimpleNamespace(num_hidden_layers=28, torch_dtype=torch.float32, hidden_size=4),
            skipkv_sentence_embedding_layer=-1,
            skipkv_steering_layer=20,
            skipkv_steering_vector_path=None,
            skipkv_enable_activation_steering=True,
            skipkv_steering_alpha=-1.25,
            skipkv_steering_alpha_increment=-0.02,
            skipkv_steering_alpha_max=0.0,
            max_decoding_seqs=2,
        )
        controller = ActivationController.create(config, FakeCacheManager())
        controller._steering_vector = torch.ones(4, device="cuda")
        controller.set_tokenizer_metadata(delimiter_token_ids={99}, non_execution_token_ids={42})

        seq = Sequence([1])
        seq.num_prompt_tokens = 0
        seq.num_tokens = 2
        seq.last_token = 99
        controller.prepare_forward([seq], is_prefill=False)

        hidden = torch.zeros((1, 4), device="cuda")
        updated, _ = controller.apply_layer_hook(20, hidden, None, None)

        self.assertTrue(torch.allclose(updated.cpu(), torch.full((1, 4), -1.31)))

    def test_skipkv_sentence_penalty_uses_cache_range_mapping(self):
        manager = object.__new__(SkipKVCacheManager)
        manager.config = SimpleNamespace(
            skipkv_enable_sentence_scoring=True,
            skipkv_sentence_score_weight=1.0,
        )
        manager._skipkv_seq_states = {}
        manager._skipkv_row_gen_indices = [{0: [0, 1, 2, 3, 4, 5]}]
        seq = Sequence([1])
        seq.num_prompt_tokens = 0
        sentence = SkipKVSentence(
            start_gen=0,
            end_gen=3,
            embedding=torch.tensor([1.0, 0.0]),
            redundancy=0.97,
        )
        manager._skipkv_seq_states[seq.seq_id] = SkipKVSequenceState(
            num_prompt_tokens=0,
            sentences=[sentence],
        )

        penalty = manager._sentence_redundancy_penalty(
            0,
            seq,
            0,
            candidate_start=0,
            candidate_end=6,
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(penalty)
        self.assertGreater(float(penalty[0]), 0.9)
        self.assertGreater(float(penalty[2]), 0.9)
        self.assertEqual(float(penalty[3]), 0.0)

    def test_rkv_selection_preserves_sink_recent_and_budget(self):
        manager = object.__new__(RKVCacheManager)
        manager.config = SimpleNamespace(
            num_sink_tokens=1,
            num_recent_tokens=1,
            rkv_similarity_threshold=0.8,
            rkv_recent_similar_keep=1,
            rkv_max_redundancy_tokens=16,
            rkv_redundancy_window=16,
            rkv_alpha=0.1,
        )
        seq = Sequence(list(range(8)))
        manager.seq_id_to_row = [{seq.seq_id: 0}]
        manager.buffer_req_to_token_slots = [torch.arange(8, dtype=torch.int32).view(1, 8)]
        manager.kv_cache = [(torch.randn(8, 1, 4), torch.randn(8, 1, 4))]

        keep = manager.select_rkv_indices(
            0,
            seq,
            torch.linspace(0.0, 1.0, steps=8),
            kv_len=8,
            budget=5,
        )

        self.assertEqual(int(keep.numel()), 5)
        self.assertIn(0, [int(x) for x in keep.tolist()])
        self.assertIn(7, [int(x) for x in keep.tolist()])

    def test_rkv_zero_redundancy_window_scores_full_candidate_set(self):
        manager = object.__new__(RKVCacheManager)
        manager.config = SimpleNamespace(
            num_sink_tokens=1,
            num_recent_tokens=1,
            rkv_similarity_threshold=0.8,
            rkv_recent_similar_keep=1,
            rkv_max_redundancy_tokens=16,
            rkv_redundancy_window=0,
            rkv_alpha=0.1,
        )
        seq = Sequence(list(range(8)))
        manager.seq_id_to_row = [{seq.seq_id: 0}]
        manager.buffer_req_to_token_slots = [torch.arange(8, dtype=torch.int32).view(1, 8)]
        manager.kv_cache = [(torch.randn(8, 1, 4), torch.randn(8, 1, 4))]
        seen_key_lengths = []

        def fake_redundancy(keys, *, similarity_threshold, recent_similar_keep, max_tokens):
            seen_key_lengths.append(int(keys.shape[0]))
            return torch.zeros((keys.shape[0],), dtype=torch.float32, device=keys.device)

        with patch.object(RKVCacheManager, "redundancy_scores_from_keys", side_effect=fake_redundancy):
            keep = manager.select_rkv_indices(
                0,
                seq,
                torch.linspace(0.0, 1.0, steps=8),
                kv_len=8,
                budget=5,
            )

        self.assertEqual(seen_key_lengths, [6])
        self.assertEqual(int(keep.numel()), 5)

    def test_rkv_query_cache_tracks_observation_tokens_not_interval(self):
        manager = object.__new__(RKVCacheManager)
        manager._rkv_observation_tokens = 3
        manager.device = torch.device("cpu")
        manager._rkv_query_cache = [torch.zeros((1, 3, 1, 2), dtype=torch.float32)]
        manager._rkv_query_positions = [torch.full((1, 3), -1, dtype=torch.int32)]

        q_prefill = torch.arange(10, dtype=torch.float32).view(5, 1, 2)
        view = SimpleNamespace(
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([5], dtype=torch.int32),
        )
        manager.record_prefill_query(
            0,
            q_prefill,
            view,
            b_start_loc=torch.tensor([0], dtype=torch.int32),
            chunk_lens=torch.tensor([5], dtype=torch.int32),
        )

        cols = torch.tensor([2, 0, 1], dtype=torch.long)
        self.assertEqual(manager._rkv_query_positions[0][0, cols].tolist(), [2, 3, 4])
        torch.testing.assert_close(manager._rkv_query_cache[0][0, cols], q_prefill[2:5])

        manager.layer_batch_states = [
            LayerBatchStates(
                req_indices=torch.tensor([0], dtype=torch.int32),
                context_lens=torch.tensor([6], dtype=torch.int32),
            )
        ]
        q_decode = torch.tensor([[[100.0, 101.0]]], dtype=torch.float32)
        manager.record_decode_query(0, q_decode)

        cols = torch.tensor([0, 1, 2], dtype=torch.long)
        self.assertEqual(manager._rkv_query_positions[0][0, cols].tolist(), [3, 4, 5])
        expected = torch.cat((q_prefill[3:5], q_decode), dim=0)
        torch.testing.assert_close(manager._rkv_query_cache[0][0, cols], expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for R-KV query score kernel tests.")
    def test_rkv_query_attention_scores_reuse_prefill_score_kernel(self):
        from tests.test_prefill_score_kernel import _prefill_score_baseline

        torch.manual_seed(29)
        device = torch.device("cuda")
        dtype = torch.float32
        kv_len = 6
        num_heads = 4
        num_kv_heads = 2
        head_dim = 16
        observation_tokens = 2
        candidate_start = 1
        num_recent_tokens = 1

        seq = Sequence([1])
        manager = object.__new__(RKVCacheManager)
        manager.config = SimpleNamespace(
            rkv_observation_tokens=observation_tokens,
            sparse_attn_score_dtype="float32",
        )
        manager.device = device
        manager._rkv_observation_tokens = observation_tokens
        manager.seq_id_to_row = [{seq.seq_id: 0}]
        manager.buffer_req_to_token_slots = [
            torch.arange(kv_len, dtype=torch.int32, device=device).view(1, kv_len)
        ]
        k_cache = torch.randn((kv_len, num_kv_heads, head_dim), dtype=dtype, device=device)
        manager.kv_cache = [(k_cache, torch.empty_like(k_cache))]
        manager._rkv_query_cache = [
            torch.empty((1, observation_tokens, num_heads, head_dim), dtype=dtype, device=device)
        ]
        manager._rkv_query_positions = [
            torch.full((1, observation_tokens), -1, dtype=torch.int32, device=device)
        ]

        q_window = torch.randn((observation_tokens, num_heads, head_dim), dtype=dtype, device=device)
        positions = torch.tensor([4, 5], dtype=torch.long, device=device)
        cols = positions.remainder(observation_tokens)
        manager._rkv_query_cache[0][0, cols] = q_window
        manager._rkv_query_positions[0][0, cols] = positions.to(torch.int32)

        scores = manager.rkv_query_attention_scores(
            0,
            seq,
            kv_len,
            candidate_start=candidate_start,
            num_recent_tokens=num_recent_tokens,
        )
        torch.cuda.synchronize()

        expected = _prefill_score_baseline(
            q_window,
            k_cache,
            manager.buffer_req_to_token_slots[0],
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([0], dtype=torch.int32, device=device),
            torch.tensor([kv_len], dtype=torch.int32, device=device),
            torch.tensor([kv_len - observation_tokens], dtype=torch.int32, device=device),
            torch.tensor([kv_len - observation_tokens], dtype=torch.int32, device=device),
            torch.tensor([kv_len], dtype=torch.int32, device=device),
            candidate_start,
            num_recent_tokens,
        )[0, :kv_len]
        torch.testing.assert_close(scores, expected, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
