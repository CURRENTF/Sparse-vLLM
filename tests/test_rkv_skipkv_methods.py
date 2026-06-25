from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

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
    def test_rkv_aliases_and_prefill_policy(self):
        self.assertEqual(normalize_sparse_method("r-kv"), "rkv")
        self.assertEqual(normalize_sparse_method("r_kv"), "rkv")
        self.assertEqual(normalize_sparse_method("skip-kv"), "skipkv")
        self.assertEqual(get_default_prefill_schedule_policy("r-kv"), PREFILL_POLICY_ALL_CHUNKED)
        self.assertEqual(get_default_prefill_schedule_policy("skipkv"), PREFILL_POLICY_ALL_CHUNKED)

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


if __name__ == "__main__":
    unittest.main()
