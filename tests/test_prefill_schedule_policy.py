import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.config import Config
from sparsevllm.engine.cache_manager.deltakv import DeltaKVCacheManager
from sparsevllm.engine.scheduler import Scheduler
from sparsevllm.engine.sequence import Sequence
from sparsevllm.method_registry import (
    PREFILL_POLICY_ALL_CHUNKED,
    PREFILL_POLICY_AUTO,
    PREFILL_POLICY_BY_METHOD,
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    get_default_prefill_schedule_policy,
    is_decode_cuda_graph_supported,
)


class FakeMemoryOracle:
    def __init__(self, free_slots=1_000_000):
        self._free_slots = int(free_slots)

    @property
    def num_free_slots(self):
        return self._free_slots

    def prefill_step_free_slots(self):
        return self._free_slots

    def reserved_prefill_slots(self, waiting, chunk_prefill_size):
        return 0

    def remaining_prefill_tokens(self, seq):
        return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)

    def prefill_batched_tokens_margin(self):
        return 0

    def prompt_admission_budgets(self, waiting, chunk_prefill_size):
        return {"slots": self._free_slots}

    def prompt_admission_costs(self, seq):
        return {"slots": int(seq.num_prompt_tokens)}

    def prompt_admission_failure_action(self):
        return "raise"

    def on_prompt_admitted(self, seq, costs):
        return None

    def prompt_logical_reservation_cost(self, seq):
        return int(seq.num_prompt_tokens)


def make_scheduler(policy, *, method="", chunk=5, max_tokens=10):
    cfg = SimpleNamespace(
        max_num_seqs_in_batch=4,
        max_num_batched_tokens=max_tokens,
        max_decoding_seqs=16,
        chunk_prefill_size=chunk,
        prefill_schedule_policy=policy,
        eos=-1,
        num_sink_tokens=1,
        num_recent_tokens=1,
        num_top_tokens=4,
        snapkv_window_size=2,
        vllm_sparse_method=method,
    )
    return Scheduler(cfg, FakeMemoryOracle())


def seq_with_len(n):
    return Sequence(list(range(n)))


class PrefillPolicyRegistryTest(unittest.TestCase):
    def test_all_supported_methods_have_one_default_policy(self):
        for method, policy in PREFILL_POLICY_BY_METHOD.items():
            with self.subTest(method=method):
                self.assertIn(
                    get_default_prefill_schedule_policy(method),
                    {PREFILL_POLICY_ALL_CHUNKED, PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH},
                )
                self.assertEqual(get_default_prefill_schedule_policy(method), policy)

    def test_deltakv_family_defaults_to_long_bs1full(self):
        for method in (
            "deltakv",
            "deltakv-triton",
            "deltakv-triton-v2",
            "deltakv-triton-v3",
            "deltakv-triton-v4",
            "deltakv-delta-quant",
            "deltakv_delta_quant",
            "deltakv-standalone",
            "deltakv-snapkv",
        ):
            with self.subTest(method=method):
                self.assertEqual(
                    get_default_prefill_schedule_policy(method),
                    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
                )

    def test_non_deltakv_defaults_to_all_chunked(self):
        for method in ("", "vanilla", "streamingllm", "attention-sink", "snapkv", "pyramidkv", "quest", "omnikv"):
            with self.subTest(method=method):
                self.assertEqual(get_default_prefill_schedule_policy(method), PREFILL_POLICY_ALL_CHUNKED)


class PrefillPolicyConfigTest(unittest.TestCase):
    def hf_config(self):
        return SimpleNamespace(
            model_type="qwen2",
            torch_dtype=torch.float16,
            max_position_embeddings=32768,
            hidden_size=8,
            intermediate_size=32,
            num_hidden_layers=2,
        )

    def make_config(self, **kwargs):
        with tempfile.TemporaryDirectory() as tmp:
            model_dir = Path(tmp)
            with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=self.hf_config()):
                return Config(model=str(model_dir), **kwargs)

    def test_auto_and_empty_policy_resolve_from_registry(self):
        cfg = self.make_config(vllm_sparse_method="vanilla", prefill_schedule_policy=PREFILL_POLICY_AUTO)
        self.assertEqual(cfg.vllm_sparse_method, "")
        self.assertEqual(cfg.prefill_schedule_policy, PREFILL_POLICY_ALL_CHUNKED)

        cfg = self.make_config(vllm_sparse_method="deltakv-standalone", prefill_schedule_policy="")
        self.assertEqual(cfg.prefill_schedule_policy, PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH)

    def test_explicit_matching_policy_passes(self):
        cfg = self.make_config(
            vllm_sparse_method="snapkv",
            prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
        )
        self.assertEqual(cfg.prefill_schedule_policy, PREFILL_POLICY_ALL_CHUNKED)

    def test_explicit_mismatched_policy_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "registry default"):
            self.make_config(
                vllm_sparse_method="deltakv-standalone",
                prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
            )

    def test_invalid_policy_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "Unsupported prefill_schedule_policy"):
            self.make_config(vllm_sparse_method="snapkv", prefill_schedule_policy="old_chunk_mode")

    def test_decode_cuda_graph_supports_non_deltakv_methods(self):
        for method in (
            "vanilla",
            "streamingllm",
            "attention-sink",
            "attention_sink",
            "snapkv",
            "pyramidkv",
            "quest",
            "omnikv",
        ):
            with self.subTest(method=method):
                cfg = self.make_config(vllm_sparse_method=method, decode_cuda_graph=True)
                self.assertTrue(cfg.decode_cuda_graph)
                self.assertTrue(is_decode_cuda_graph_supported(cfg.vllm_sparse_method))

        cfg = self.make_config(vllm_sparse_method="omnikv", omnikv_decode_cuda_graph=True)
        self.assertTrue(cfg.decode_cuda_graph)
        self.assertTrue(cfg.omnikv_decode_cuda_graph)

        with self.assertRaisesRegex(ValueError, "only valid"):
            self.make_config(vllm_sparse_method="snapkv", omnikv_decode_cuda_graph=True)

    def test_decode_cuda_graph_rejects_deltakv_family(self):
        for method in (
            "deltakv",
            "deltakv-triton",
            "deltakv-triton-v2",
            "deltakv-triton-v3",
            "deltakv-triton-v4",
            "deltakv-delta-quant",
            "deltakv_delta_quant",
            "deltakv-standalone",
            "deltakv-snapkv",
        ):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, "does not support DeltaKV"):
                    self.make_config(vllm_sparse_method=method, decode_cuda_graph=True)

    def test_decode_cuda_graph_requires_single_tp(self):
        with self.assertRaisesRegex(ValueError, "tensor_parallel_size=1"):
            self.make_config(
                vllm_sparse_method="omnikv",
                decode_cuda_graph=True,
                tensor_parallel_size=2,
            )

    def test_decode_cuda_graph_capture_sampling_requires_graph(self):
        with self.assertRaisesRegex(ValueError, "requires decode_cuda_graph"):
            self.make_config(
                vllm_sparse_method="omnikv",
                decode_cuda_graph_capture_sampling=True,
            )

    def test_decode_cuda_graph_auto_capture_sizes_pad_to_power_of_two(self):
        cfg = self.make_config(
            vllm_sparse_method="omnikv",
            decode_cuda_graph=True,
            max_decoding_seqs=6,
        )
        self.assertEqual(cfg.decode_cuda_graph_capture_sizes, [1, 2, 4, 8])

    def test_decode_cuda_graph_explicit_capture_sizes_are_validated(self):
        cfg = self.make_config(
            vllm_sparse_method="omnikv",
            decode_cuda_graph=True,
            max_decoding_seqs=6,
            decode_cuda_graph_capture_sizes="1,4,8,8",
        )
        self.assertEqual(cfg.decode_cuda_graph_capture_sizes, [1, 4, 8])

        with self.assertRaisesRegex(ValueError, "cover max_decoding_seqs"):
            self.make_config(
                vllm_sparse_method="omnikv",
                decode_cuda_graph=True,
                max_decoding_seqs=6,
                decode_cuda_graph_capture_sizes=[1, 2, 4],
            )


class SchedulerPrefillPolicyTest(unittest.TestCase):
    def test_all_chunked_keeps_long_and_short_separate(self):
        scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED, method="")
        long_seq = seq_with_len(20)
        short_seq = seq_with_len(4)
        scheduler.add(long_seq)
        scheduler.add(short_seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [long_seq])
        self.assertEqual(long_seq.current_chunk_size, 5)
        self.assertEqual(short_seq.current_chunk_size, None)

    def test_all_chunked_caps_each_prefill_by_chunk_size(self):
        scheduler = make_scheduler(PREFILL_POLICY_ALL_CHUNKED, method="", chunk=5, max_tokens=20)
        seq_a = seq_with_len(20)
        seq_b = seq_with_len(12)
        scheduler.add(seq_a)
        scheduler.add(seq_b)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertTrue(all(seq.current_chunk_size <= 5 for seq in scheduled))

    def test_long_bs1full_policy_schedules_long_as_single_full_prefill(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv-standalone",
            chunk=5,
            max_tokens=10,
        )
        long_a = seq_with_len(20)
        long_b = seq_with_len(30)
        scheduler.add(long_a)
        scheduler.add(long_b)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [long_a])
        self.assertEqual(long_a.current_chunk_size, 20)
        self.assertEqual(long_b.current_chunk_size, None)

    def test_long_bs1full_policy_batches_short_chunked_prefill(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv-standalone",
            chunk=5,
            max_tokens=10,
        )
        short_a = seq_with_len(6)
        short_b = seq_with_len(4)
        scheduler.add(short_a)
        scheduler.add(short_b)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [short_a, short_b])
        self.assertEqual(short_a.current_chunk_size, 5)
        self.assertEqual(short_b.current_chunk_size, 4)


class DeltaKVFullPrefillStagingTest(unittest.TestCase):
    def test_full_prefill_staging_only_for_single_complete_long_prefill(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            chunk_prefill_size=5,
        )
        manager.deltakv_layer_ids = [0]
        seq = seq_with_len(20)

        seq.current_chunk_size = 20
        self.assertTrue(DeltaKVCacheManager._should_use_full_prefill_staging(manager, [seq]))

        seq.current_chunk_size = 5
        self.assertFalse(DeltaKVCacheManager._should_use_full_prefill_staging(manager, [seq]))

        seq.current_chunk_size = 20
        seq.num_prefilled_tokens = 5
        self.assertFalse(DeltaKVCacheManager._should_use_full_prefill_staging(manager, [seq]))

        seq.num_prefilled_tokens = 0
        other = seq_with_len(20)
        other.current_chunk_size = 20
        self.assertFalse(DeltaKVCacheManager._should_use_full_prefill_staging(manager, [seq, other]))

        manager.config.prefill_schedule_policy = PREFILL_POLICY_ALL_CHUNKED
        self.assertFalse(DeltaKVCacheManager._should_use_full_prefill_staging(manager, [seq]))

    def test_full_prefill_plan_keeps_only_persistent_final_representation(self):
        plan = DeltaKVCacheManager._deltakv_full_prefill_plan_cpu(
            20,
            sink=2,
            recent=4,
            cluster_step=4,
        )

        self.assertEqual(plan.evict_start, 2)
        self.assertEqual(plan.evict_end, 14)
        self.assertEqual(plan.center_positions, (2, 6, 10))
        self.assertIn(3, plan.latent_positions)
        self.assertLess(len(plan.keep_positions), plan.total_len)
        self.assertNotIn(3, plan.keep_positions)

    def test_layer_attention_end_triggers_layer_local_staging_compression(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager._deltakv_prefill_staging_active = True
        manager.deltakv_layer_to_idx = {0: 0}
        manager.deltakv_layer_ids = [0]
        manager._deltakv_full_prefill_compressed_layers = set()
        calls = []

        def compress(layer_idx):
            calls.append(layer_idx)
            manager._deltakv_full_prefill_compressed_layers.add(layer_idx)

        manager._deltakv_compress_full_prefill_layer = compress

        DeltaKVCacheManager.on_layer_attention_end(manager, 0)

        self.assertEqual(calls, [0])
        self.assertFalse(manager._deltakv_prefill_staging_active)


if __name__ == "__main__":
    unittest.main()
