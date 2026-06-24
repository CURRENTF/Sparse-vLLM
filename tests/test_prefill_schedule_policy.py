import os
import tempfile
import unittest
from collections import OrderedDict
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.config import Config
from sparsevllm.engine.cache_manager.standard import StandardCacheManager
from sparsevllm.engine.cache_manager.deltakv import DeltaKVCacheManager
from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager
from sparsevllm.engine.cache_manager.deltakv_less_memory_cuda_graph import (
    DeltaKVLessMemoryCudaGraphCacheManager,
)
from sparsevllm.engine.decode_cuda_graph import DecodeCudaGraphKey, DecodeCudaGraphRunner, DecodeCudaGraphState
from sparsevllm.engine.llm_engine import _deltakv_graph_warmup_profile, _use_graph_scaled_warmup
from sparsevllm.engine.scheduler import Scheduler
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.sparse_controller import SparseController
from sparsevllm.method_registry import (
    PREFILL_POLICY_ALL_CHUNKED,
    PREFILL_POLICY_AUTO,
    PREFILL_POLICY_BY_METHOD,
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    get_default_prefill_schedule_policy,
    is_decode_cuda_graph_supported,
)


class FakeMemoryOracle:
    def __init__(
        self,
        free_slots=1_000_000,
        *,
        step_free_slots=None,
        force_full_prefill=False,
        prefix_hit_len=0,
        prefix_hit_blocks=0,
    ):
        self._free_slots = int(free_slots)
        self._step_free_slots = int(step_free_slots) if step_free_slots is not None else int(free_slots)
        self._force_full_prefill = bool(force_full_prefill)
        self.prefix_hit_len = int(prefix_hit_len)
        self.prefix_hit_blocks = int(prefix_hit_blocks)
        self.refresh_calls = 0
        self.clear_calls = 0

    @property
    def num_free_slots(self):
        return self._free_slots

    def prefill_step_free_slots(self):
        return self._step_free_slots

    def should_schedule_full_prefill(self, seq):
        return self._force_full_prefill and int(seq.num_prefilled_tokens) == 0

    def is_full_prefill_step(self, seqs):
        return False

    def prefill_step_free_slots_for(self, seq):
        return self._free_slots

    def prefill_step_reservation_cost(self, seq, scheduled_tokens):
        return int(scheduled_tokens)

    def decode_step_free_slots(self):
        return self._free_slots

    def decode_step_free_slots_for(self, seq):
        return self._free_slots

    def decode_step_reservation_cost(self, seq):
        return 1

    def reserved_prefill_slots(self, waiting, chunk_prefill_size):
        return 0

    def remaining_prefill_tokens(self, seq):
        virtual_prefilled = max(seq.num_prefilled_tokens, seq.prefix_cache_hit_len)
        return int(seq.num_prompt_tokens - virtual_prefilled)

    def prefill_batched_tokens_margin(self):
        return 0

    def prompt_admission_budgets(self, waiting, chunk_prefill_size):
        return {"slots": self._free_slots}

    def prompt_admission_free_slots(self):
        return self._free_slots

    def prompt_admission_costs(self, seq):
        return {"slots": int(seq.num_prompt_tokens - seq.prefix_cache_hit_len)}

    def prompt_admission_failure_action(self):
        return "raise"

    def on_prompt_admitted(self, seq, costs):
        return None

    def prompt_logical_reservation_cost(self, seq):
        return int(seq.num_prompt_tokens - seq.prefix_cache_hit_len)

    def refresh_prefix_cache_hit(self, seq):
        self.refresh_calls += 1
        seq.clear_prefix_cache_hit()
        if self.prefix_hit_len <= 0:
            return
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = self.prefix_hit_len
        seq.prefix_cache_hit_blocks = self.prefix_hit_blocks
        seq.prefix_cache_hit_last_key = b"test"
        seq.prefix_cache_block_size = 4
        seq.prefix_cache_method = ""

    def clear_prefix_cache_hit(self, seq):
        self.clear_calls += 1
        seq.clear_prefix_cache_hit()

    def free_slot_stats(self):
        return {"free_slots": int(self._free_slots), "step_free_slots": int(self._step_free_slots)}


def make_scheduler(policy, *, method="", chunk=5, max_tokens=10, oracle=None):
    cfg = SimpleNamespace(
        max_num_seqs_in_batch=4,
        max_num_batched_tokens=max_tokens,
        max_decoding_seqs=16,
        chunk_prefill_size=chunk,
        prefill_schedule_policy=policy,
        eos=-1,
        num_sink_tokens=1,
        num_recent_tokens=1,
        decode_keep_tokens=4,
        snapkv_window_size=2,
        vllm_sparse_method=method,
    )
    return Scheduler(cfg, oracle or FakeMemoryOracle())


def make_sparse_controller_config():
    return SimpleNamespace(
        vllm_sparse_method="deltakv",
        obs_layer_ids=[0],
        full_attn_layers=[0],
        hf_config=SimpleNamespace(
            num_hidden_layers=1,
            hidden_size=8,
            num_attention_heads=1,
        ),
        num_sink_tokens=1,
        num_recent_tokens=1,
        decode_keep_tokens=4,
        sparse_attn_score_dtype="float32",
    )


def make_scheduler_with_oracle(policy, oracle, *, method="", chunk=5, max_tokens=10):
    cfg = SimpleNamespace(
        max_num_seqs_in_batch=4,
        max_num_batched_tokens=max_tokens,
        max_decoding_seqs=16,
        chunk_prefill_size=chunk,
        prefill_schedule_policy=policy,
        eos=-1,
        num_sink_tokens=1,
        num_recent_tokens=1,
        decode_keep_tokens=4,
        snapkv_window_size=2,
        vllm_sparse_method=method,
    )
    return Scheduler(cfg, oracle)


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

    def test_full_prefill_methods_default_to_long_bs1full(self):
        for method in (
            "pyramidkv",
            "deltakv",
            "deltakv-less-memory",
            "deltakv",
        ):
            with self.subTest(method=method):
                self.assertEqual(
                    get_default_prefill_schedule_policy(method),
                    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
                )

    def test_other_non_deltakv_defaults_to_all_chunked(self):
        for method in ("", "vanilla", "streamingllm", "attention-sink", "snapkv", "quest", "omnikv"):
            with self.subTest(method=method):
                self.assertEqual(get_default_prefill_schedule_policy(method), PREFILL_POLICY_ALL_CHUNKED)

    def test_deltakv_topk_tiebreak_env_defaults_off(self):
        with patch.dict(os.environ, {}, clear=True):
            controller = SparseController(make_sparse_controller_config(), SimpleNamespace())

        self.assertFalse(controller.dynamic_deltakv_topk_tiebreak)
        self.assertFalse(controller.sparse_config["dynamic_deltakv_topk_tiebreak"])

    def test_deltakv_topk_tiebreak_env_can_enable(self):
        with patch.dict(os.environ, {"SPARSEVLLM_DELTAKV_DETERMINISTIC_TOPK_TIEBREAK": "1"}, clear=True):
            controller = SparseController(make_sparse_controller_config(), SimpleNamespace())

        self.assertTrue(controller.dynamic_deltakv_topk_tiebreak)
        self.assertTrue(controller.sparse_config["dynamic_deltakv_topk_tiebreak"])

    def test_deltakv_topk_tiebreak_env_rejects_invalid_value(self):
        with patch.dict(os.environ, {"SPARSEVLLM_DELTAKV_DETERMINISTIC_TOPK_TIEBREAK": "maybe"}, clear=True):
            with self.assertRaisesRegex(ValueError, "SPARSEVLLM_DELTAKV_DETERMINISTIC_TOPK_TIEBREAK"):
                SparseController(make_sparse_controller_config(), SimpleNamespace())


class StandardCacheManagerAdmissionTest(unittest.TestCase):
    def test_prompt_admission_tracks_row_budget(self):
        manager = object.__new__(StandardCacheManager)
        manager._num_free_slots = 100
        manager.free_rows = deque([0, 1])

        budgets = manager.prompt_admission_budgets(deque(), chunk_prefill_size=16)
        costs = manager.prompt_admission_costs(seq_with_len(10))

        self.assertEqual(budgets["slots"], 100)
        self.assertEqual(budgets["rows"], 2)
        self.assertEqual(costs["slots"], 10)
        self.assertEqual(costs["rows"], 1)


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

        cfg = self.make_config(
            vllm_sparse_method="deltakv-less-memory",
            prefill_schedule_policy="",
            allow_missing_deltakv_path=True,
            kv_quant_bits=0,
        )
        self.assertEqual(cfg.prefill_schedule_policy, PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH)

        cfg = self.make_config(vllm_sparse_method="pyramidkv", prefill_schedule_policy=None)
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
                vllm_sparse_method="deltakv",
                prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
                allow_missing_deltakv_path=True,
            )

        with self.assertRaisesRegex(ValueError, "registry default"):
            self.make_config(
                vllm_sparse_method="pyramidkv",
                prefill_schedule_policy=PREFILL_POLICY_ALL_CHUNKED,
            )

    def test_invalid_policy_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "Unsupported prefill_schedule_policy"):
            self.make_config(vllm_sparse_method="snapkv", prefill_schedule_policy="old_chunk_mode")

    def test_removed_full_layer_kivi_experiment_knobs_fail_fast(self):
        with self.assertRaisesRegex(ValueError, "fused_decode was removed"):
            self.make_config(
                vllm_sparse_method="deltakv-less-memory",
                allow_missing_deltakv_path=True,
                enable_full_layer_kivi_fused_decode=True,
            )

        with self.assertRaisesRegex(ValueError, "grouped_decode was removed"):
            self.make_config(
                vllm_sparse_method="deltakv-less-memory",
                allow_missing_deltakv_path=True,
                enable_full_layer_kivi_grouped_decode=True,
            )

    def test_legacy_full_layer_kivi_dense_decode_knob_is_noop(self):
        cfg = self.make_config(
            vllm_sparse_method="deltakv-less-memory",
            allow_missing_deltakv_path=True,
            kv_quant_bits=0,
            enable_full_layer_kivi_dense_decode=True,
        )
        self.assertTrue(cfg.enable_full_layer_kivi_dense_decode)

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

    def test_decode_cuda_graph_supports_all_deltakv_methods(self):
        for method in ("deltakv", "deltakv-less-memory", "deltakv-less-memory-cudagraph"):
            with self.subTest(method=method):
                cfg = self.make_config(
                    vllm_sparse_method=method,
                    decode_cuda_graph=True,
                    allow_missing_deltakv_path=True,
                    kv_quant_bits=0,
                )
                self.assertTrue(cfg.decode_cuda_graph)
                self.assertTrue(is_decode_cuda_graph_supported(cfg.vllm_sparse_method))

    def test_deltakv_legacy_graph_method_name_is_alias(self):
        cfg = self.make_config(
            vllm_sparse_method="deltakv-less-memory-cudagraph",
            allow_missing_deltakv_path=True,
            kv_quant_bits=0,
        )
        self.assertEqual(cfg.vllm_sparse_method, "deltakv")
        self.assertTrue(cfg.decode_cuda_graph)

    def test_prefill_cuda_graph_is_not_a_supported_config_key(self):
        with self.assertRaisesRegex(TypeError, "prefill_cuda_graph"):
            self.make_config(
                vllm_sparse_method="deltakv-less-memory",
                prefill_cuda_graph=True,
                allow_missing_deltakv_path=True,
            )

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

    def test_decode_cuda_graph_auto_context_sizes_use_powers_of_two_from_1k(self):
        cfg = self.make_config(
            vllm_sparse_method="omnikv",
            decode_cuda_graph=True,
            max_model_len=9000,
        )
        self.assertEqual(cfg.decode_cuda_graph_context_sizes, [1024, 2048, 4096, 8192, 16384])

    def test_decode_cuda_graph_explicit_context_sizes_are_sorted(self):
        cfg = self.make_config(
            vllm_sparse_method="quest",
            decode_cuda_graph=True,
            decode_cuda_graph_context_sizes="4096,1024,4096,2048",
        )
        self.assertEqual(cfg.decode_cuda_graph_context_sizes, [1024, 2048, 4096])


class DecodeCudaGraphCapacityPolicyTest(unittest.TestCase):
    def make_graph_manager(self, *, context_policy="current", max_cached_graphs=None):
        manager = object.__new__(DeltaKVLessMemoryCudaGraphCacheManager)
        manager.config = SimpleNamespace(
            decode_cuda_graph=True,
            decode_cuda_graph_context_policy=context_policy,
            decode_cuda_graph_max_cached_graphs=max_cached_graphs,
        )
        return manager

    def make_runner(self, method="quest", cache_manager=None):
        runner = object.__new__(DecodeCudaGraphRunner)
        runner.method = method
        runner.cache_manager = cache_manager if cache_manager is not None else SimpleNamespace()
        runner.max_context_len_override = None
        runner._graphs = {}
        runner.capture_sizes = [1, 2, 4, 8, 16]
        runner.context_sizes = [1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072]
        return runner

    def make_seq(self, *, prompt_len=100, max_tokens=900, num_tokens=101):
        return SimpleNamespace(
            num_prompt_tokens=prompt_len,
            max_tokens=max_tokens,
            num_tokens=num_tokens,
        )

    def test_deltakv_graph_defaults_to_current_bucket(self):
        runner = self.make_runner(
            "deltakv",
            cache_manager=self.make_graph_manager(),
        )
        seqs = [self.make_seq(prompt_len=4096, max_tokens=120000, num_tokens=4097)]

        with patch.dict(os.environ, {}, clear=True):
            context_capacity, allow_larger = runner._graph_context_capacity_policy(seqs)

        self.assertEqual(context_capacity, 8192)
        self.assertFalse(allow_larger)

    def test_requested_context_policy_uses_final_length_bucket(self):
        runner = self.make_runner(
            "deltakv",
            cache_manager=self.make_graph_manager(context_policy="requested"),
        )
        seqs = [self.make_seq(prompt_len=4096, max_tokens=120000, num_tokens=4097)]

        context_capacity, allow_larger = runner._graph_context_capacity_policy(seqs)

        self.assertEqual(context_capacity, 131072)
        self.assertFalse(allow_larger)

    def test_legacy_deltakv_context_env_does_not_override_shared_policy(self):
        runner = self.make_runner(
            "deltakv",
            cache_manager=self.make_graph_manager(),
        )
        seqs = [self.make_seq(prompt_len=100, max_tokens=9000, num_tokens=101)]

        with patch.dict(
            os.environ,
            {
                "SPARSEVLLM_DELTAKV_GRAPH_ALLOW_LARGER_CONTEXT": "1",
                "SPARSEVLLM_DELTAKV_GRAPH_CONTEXT_CAP": "requested",
                "SPARSEVLLM_DELTAKV_GRAPH_CURRENT_CAP": "0",
            },
            clear=True,
        ):
            context_capacity, allow_larger = runner._graph_context_capacity_policy(seqs)

        self.assertEqual(context_capacity, 1024)
        self.assertFalse(allow_larger)

    def test_deltakv_graph_eager_static_uses_current_capacity_policy(self):
        graph_manager = self.make_graph_manager()
        runner = self.make_runner(
            "deltakv",
            cache_manager=graph_manager,
        )
        runner.cache_manager = SimpleNamespace(
            decode_cuda_graph_context_capacity=graph_manager.decode_cuda_graph_context_capacity,
            select_decode_cuda_graph_batch_size=graph_manager.select_decode_cuda_graph_batch_size,
            set_decode_static_max_context_len=lambda value: setattr(runner, "last_static_max_context_len", value),
            prepare_decode_static=lambda seqs, input_ids, positions, slot_mapping, context_lens, req_indices: (
                input_ids,
                positions,
                None,
            ),
        )
        runner.sparse_controller = SimpleNamespace(prepare_forward=lambda seqs, is_prefill: None)
        runner.is_long_text_batch = lambda seqs, is_prefill: False
        runner.run_model = lambda input_ids, positions, is_prefill: torch.zeros((input_ids.shape[0], 4))
        runner.max_cached_graphs = None
        runner.last_state_key = None
        runner.last_real_batch_size = None
        seqs = [self.make_seq(prompt_len=100, max_tokens=900, num_tokens=101)]
        real_empty = torch.empty

        def empty_on_cpu(shape, *, dtype=None, device=None):
            del device
            return real_empty(shape, dtype=dtype)

        with patch.dict(os.environ, {}, clear=True):
            with patch("sparsevllm.engine.decode_cuda_graph.torch.empty", side_effect=empty_on_cpu):
                runner.run_eager_static(seqs)

        self.assertIsNotNone(runner.last_state_key)
        self.assertEqual(runner.last_state_key.context_capacity, 1024)
        self.assertEqual(runner.last_static_max_context_len, 1024)

    def test_exact_current_policy_does_not_reuse_larger_warmup_state(self):
        runner = self.make_runner("quest")
        warmup_key = DecodeCudaGraphKey("quest", 1, 16384, False, False)
        warmup_state = DecodeCudaGraphState(key=warmup_key)
        runner._graphs[warmup_key] = warmup_state
        real_empty = torch.empty

        def empty_on_cpu(shape, *, dtype=None, device=None):
            del device
            return real_empty(shape, dtype=dtype)

        with patch("sparsevllm.engine.decode_cuda_graph.torch.empty", side_effect=empty_on_cpu):
            state = runner._select_state(
                method="quest",
                batch_size=1,
                context_capacity=1024,
                is_long_text=False,
                capture_sampling=False,
                allow_larger_context_capacity=False,
            )

        self.assertIsNot(state, warmup_state)
        self.assertEqual(state.key.context_capacity, 1024)

    def test_deltakv_graph_cache_is_unbounded_by_default(self):
        runner = self.make_runner(
            "deltakv",
            cache_manager=self.make_graph_manager(),
        )

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(runner._resolve_max_cached_graphs())

    def test_non_deltakv_graph_cache_is_unbounded_by_default(self):
        runner = self.make_runner("quest")

        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(runner._resolve_max_cached_graphs())

    def test_graph_cache_limit_env_must_be_positive_integer(self):
        runner = self.make_runner(
            "deltakv",
            cache_manager=self.make_graph_manager(),
        )

        for value in ("0", "-1", "many"):
            with patch.dict(os.environ, {"SPARSEVLLM_DELTAKV_MAX_CUDAGRAPHS": value}, clear=True):
                with self.assertRaisesRegex(ValueError, "SPARSEVLLM_DELTAKV_MAX_CUDAGRAPHS"):
                    runner._resolve_max_cached_graphs()

    def test_evict_cached_graphs_releases_oldest_unprotected_state(self):
        runner = self.make_runner("deltakv")
        runner.max_cached_graphs = 1
        runner._graphs = OrderedDict()
        old_key = DecodeCudaGraphKey("deltakv", 1, 1024, False, False)
        new_key = DecodeCudaGraphKey("deltakv", 1, 2048, False, False)
        old_state = DecodeCudaGraphState(key=old_key)
        old_state.keepalive.append(object())
        old_state.sparse_state_refs[0] = {"attn_score": object()}
        new_state = DecodeCudaGraphState(key=new_key)
        runner._graphs[old_key] = old_state
        runner._graphs[new_key] = new_state

        runner._evict_cached_graphs(new_key)

        self.assertNotIn(old_key, runner._graphs)
        self.assertIn(new_key, runner._graphs)
        self.assertEqual(old_state.keepalive, [])
        self.assertEqual(old_state.sparse_state_refs, {})

    def test_deltakv_graph_uses_shared_decode_batch_bucket(self):
        runner = self.make_runner(
            "deltakv",
            cache_manager=self.make_graph_manager(),
        )

        self.assertEqual(runner._select_graph_batch_size(3), 4)

    def test_non_deltakv_still_uses_capture_size_bucket(self):
        runner = self.make_runner("quest")

        self.assertEqual(runner._select_graph_batch_size(3), 4)


class DecodeCudaGraphWarmupPolicyTest(unittest.TestCase):
    def make_config(self, method="deltakv", decode_cuda_graph=True):
        return SimpleNamespace(
            vllm_sparse_method=method,
            decode_cuda_graph=decode_cuda_graph,
        )

    def test_deltakv_graph_defaults_to_eager_sized_engine_warmup(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_deltakv_graph_warmup_profile(self.make_config()), "prefill_only")
            self.assertFalse(_use_graph_scaled_warmup(self.make_config()))

    def test_deltakv_graph_warmup_can_reproduce_old_policy(self):
        with patch.dict(os.environ, {"SPARSEVLLM_DELTAKV_GRAPH_WARMUP": "graph"}, clear=True):
            self.assertEqual(_deltakv_graph_warmup_profile(self.make_config()), "graph")
            self.assertTrue(_use_graph_scaled_warmup(self.make_config()))

    def test_deltakv_graph_warmup_supports_diagnostic_profiles(self):
        for env_value, expected in (
            ("decode_1seq", "decode_1seq"),
            ("big_prefill_only", "big_prefill_only"),
            ("prefill_only", "prefill_only"),
        ):
            with self.subTest(env_value=env_value):
                with patch.dict(os.environ, {"SPARSEVLLM_DELTAKV_GRAPH_WARMUP": env_value}, clear=True):
                    self.assertEqual(_deltakv_graph_warmup_profile(self.make_config()), expected)
                    self.assertFalse(_use_graph_scaled_warmup(self.make_config()))

    def test_non_deltakv_keeps_graph_scaled_engine_warmup(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_deltakv_graph_warmup_profile(self.make_config(method="omnikv")), "graph")
            self.assertTrue(_use_graph_scaled_warmup(self.make_config(method="omnikv")))


class DeltaKVLessMemoryCudaGraphReserveTest(unittest.TestCase):
    def make_manager(self, decode_cuda_graph=True):
        manager = object.__new__(DeltaKVLessMemoryCudaGraphCacheManager)
        manager.config = SimpleNamespace(decode_cuda_graph=decode_cuda_graph)
        return manager

    def test_regular_less_memory_has_no_graph_workspace_reserve(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)

        self.assertEqual(manager._extra_workspace_reserve_bytes(), 0)

    def test_graph_mode_reserves_capture_memory_by_default(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                self.make_manager(decode_cuda_graph=True)._decode_cuda_graph_memory_reserve_bytes(),
                4 * 1024**3,
            )

    def test_non_graph_mode_does_not_reserve_capture_memory(self):
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(
                self.make_manager(decode_cuda_graph=False)._decode_cuda_graph_memory_reserve_bytes(),
                0,
            )

    def test_graph_reserve_env_overrides_default(self):
        with patch.dict(os.environ, {"SPARSEVLLM_DELTAKV_CUDAGRAPH_RESERVE_BYTES": "12345"}, clear=True):
            self.assertEqual(
                self.make_manager(decode_cuda_graph=True)._decode_cuda_graph_memory_reserve_bytes(),
                12345,
            )

    def test_graph_reserve_env_must_be_non_negative_integer(self):
        manager = self.make_manager(decode_cuda_graph=True)

        for value in ("-1", "large"):
            with patch.dict(os.environ, {"SPARSEVLLM_DELTAKV_CUDAGRAPH_RESERVE_BYTES": value}, clear=True):
                with self.assertRaisesRegex(ValueError, "SPARSEVLLM_DELTAKV_CUDAGRAPH_RESERVE_BYTES"):
                    manager._decode_cuda_graph_memory_reserve_bytes()


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
            method="deltakv",
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
            method="deltakv",
            chunk=5,
            max_tokens=10,
        )
        short_a = seq_with_len(5)
        short_b = seq_with_len(4)
        scheduler.add(short_a)
        scheduler.add(short_b)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [short_a, short_b])
        self.assertEqual(short_a.current_chunk_size, 5)
        self.assertEqual(short_b.current_chunk_size, 4)

    def test_deltakv_short_prefill_defers_when_step_free_cannot_fit_whole_prompt(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv-less-memory",
            chunk=8192,
            max_tokens=16384,
            oracle=FakeMemoryOracle(step_free_slots=32),
        )
        short_seq = seq_with_len(8192)
        decode_seq = seq_with_len(3)
        decode_seq.num_prefilled_tokens = decode_seq.num_prompt_tokens
        scheduler.add(short_seq)
        scheduler.decoding.append(decode_seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertFalse(is_prefill)
        self.assertEqual(scheduled, [decode_seq])
        self.assertEqual(short_seq.current_chunk_size, None)
        self.assertEqual(list(scheduler.waiting), [short_seq])

    def test_deltakv_short_prefill_fails_fast_when_no_work_can_free_slots(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv",
            chunk=8192,
            max_tokens=16384,
            oracle=FakeMemoryOracle(step_free_slots=32),
        )
        short_seq = seq_with_len(8192)
        scheduler.add(short_seq)

        with self.assertRaisesRegex(RuntimeError, "DeltaKV short prefill cannot be split"):
            scheduler.schedule()

    def test_full_prefill_hook_routes_short_bucket_as_single_full_prefill(self):
        scheduler = make_scheduler(
            PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            method="deltakv-less-memory",
            chunk=8192,
            max_tokens=16384,
            oracle=FakeMemoryOracle(step_free_slots=32, force_full_prefill=True),
        )
        short_seq = seq_with_len(8192)
        scheduler.add(short_seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [short_seq])
        self.assertEqual(short_seq.current_chunk_size, 8192)

    def test_prefix_cache_hit_reduces_prefill_work_for_fresh_prompt(self):
        oracle = FakeMemoryOracle(prefix_hit_len=8, prefix_hit_blocks=2)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=5,
            max_tokens=20,
        )
        seq = seq_with_len(20)
        scheduler.add(seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(oracle.refresh_calls, 1)
        self.assertTrue(seq.prefix_cache_enabled)
        self.assertEqual(seq.num_prefilled_tokens, 8)
        self.assertEqual(seq.current_chunk_size, 5)

    def test_prefix_cache_lookup_skips_preempted_completion_replay(self):
        oracle = FakeMemoryOracle(prefix_hit_len=8, prefix_hit_blocks=2)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=5,
            max_tokens=20,
        )
        seq = seq_with_len(20)
        seq.append_token(99)
        seq.num_prefilled_tokens = 0
        scheduler.add(seq)

        scheduled, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(scheduled, [seq])
        self.assertEqual(oracle.refresh_calls, 0)
        self.assertFalse(seq.prefix_cache_enabled)
        self.assertEqual(seq.num_prefilled_tokens, 0)

    def test_decode_preemption_after_generation_fails_fast(self):
        oracle = FakeMemoryOracle(free_slots=0)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="",
            chunk=5,
            max_tokens=20,
        )
        seq = seq_with_len(8)
        seq.num_prefilled_tokens = seq.num_prompt_tokens
        seq.append_token(99)
        scheduler.decoding.append(seq)

        with self.assertRaisesRegex(RuntimeError, "Decode preemption replay"):
            scheduler.schedule()

    def test_decode_preempts_when_no_candidate_can_use_partial_capacity(self):
        class PartialPageOracle(FakeMemoryOracle):
            def decode_step_free_slots(self):
                return 1

            def decode_step_free_slots_for(self, seq):
                return 0

        oracle = PartialPageOracle(free_slots=0)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="quest",
            chunk=5,
            max_tokens=20,
        )
        seq = seq_with_len(8)
        seq.num_prefilled_tokens = seq.num_prompt_tokens
        scheduler.decoding.append(seq)

        scheduled, is_prefill, preempted = scheduler.schedule()

        self.assertEqual(scheduled, [])
        self.assertFalse(is_prefill)
        self.assertEqual(preempted, [seq])
        self.assertEqual(list(scheduler.decoding), [])
        self.assertEqual(list(scheduler.waiting), [seq])
        self.assertEqual(scheduler.total_preemptions, 1)

    def test_prefill_fails_fast_when_no_candidate_can_use_partial_capacity(self):
        class PartialPageOracle(FakeMemoryOracle):
            def prefill_step_free_slots(self):
                return 1

            def prefill_step_free_slots_for(self, seq):
                return 0

        oracle = PartialPageOracle(free_slots=0, step_free_slots=1)
        scheduler = make_scheduler_with_oracle(
            PREFILL_POLICY_ALL_CHUNKED,
            oracle,
            method="quest",
            chunk=5,
            max_tokens=20,
        )
        scheduler.add(seq_with_len(8))

        with self.assertRaisesRegex(RuntimeError, "No prefill candidate can use"):
            scheduler.schedule()

    def test_sequence_setstate_accepts_pre_prefix_cache_ipc_state(self):
        seq = seq_with_len(4)
        seq.current_chunk_size = 4
        old_state = (
            seq.seq_id,
            seq.status,
            seq.num_tokens,
            seq.num_prompt_tokens,
            seq.num_prefilled_tokens,
            seq.current_chunk_size,
            seq.temperature,
            seq.top_p,
            seq.top_k,
            seq.max_tokens,
            seq.ignore_eos,
            seq.logprobs,
            seq.token_ids,
        )
        restored = object.__new__(Sequence)
        restored.__setstate__(old_state)

        self.assertFalse(restored.prefix_cache_enabled)
        self.assertEqual(restored.prefix_cache_hit_len, 0)


class DeltaKVFullPrefillStagingTest(unittest.TestCase):
    def test_static_decode_resets_deltakv_view_cache_before_validation(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager._deltakv_view_cache_key = (1, 1, 2, 1, 4)
        manager._deltakv_view_cache_value = object()
        reset_calls = []

        def reset_view_cache():
            reset_calls.append(True)
            manager._deltakv_view_cache_key = None
            manager._deltakv_view_cache_value = None

        manager._deltakv_reset_view_cache = reset_view_cache

        with self.assertRaisesRegex(ValueError, "non-empty real decode batch"):
            DeltaKVCacheManager.prepare_decode_static(
                manager,
                [],
                torch.empty((1,), dtype=torch.int64),
                torch.empty((1,), dtype=torch.int64),
                torch.empty((1,), dtype=torch.int32),
                torch.empty((1,), dtype=torch.int32),
                torch.empty((1,), dtype=torch.int32),
            )

        self.assertEqual(reset_calls, [True])
        self.assertIsNone(manager._deltakv_view_cache_key)
        self.assertIsNone(manager._deltakv_view_cache_value)

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

    def test_prompt_admission_counts_sparse_raw_keep_positions_after_graph_reserve(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager.config = SimpleNamespace(
            num_sink_tokens=8,
            num_recent_tokens=128,
            cluster_ratio=0.1,
        )
        manager._num_free_slots_full = 100_000
        manager._num_free_slots_deltakv_full = 16_988
        manager._deltakv_temp_full_reserve = 16_384
        manager._deltakv_centers_capacity = 10_000
        manager._deltakv_centers_reserved_total = 0

        seq = seq_with_len(8192)
        budgets = DeltaKVCacheManager.prompt_admission_budgets(manager, deque(), chunk_prefill_size=2048)
        costs = DeltaKVCacheManager.prompt_admission_costs(manager, seq)

        self.assertEqual(costs["deltakv_raw"], 1050)
        self.assertEqual(budgets["deltakv_raw"], 604)
        self.assertLess(budgets["deltakv_raw"], costs["deltakv_raw"])

        manager._deltakv_static_temp_slots_reserved_total = 12_000
        budgets = DeltaKVCacheManager.prompt_admission_budgets(manager, deque(), chunk_prefill_size=2048)
        self.assertEqual(budgets["deltakv_raw"], 12_604)

    def test_temp_deltakv_full_allocation_does_not_alias_free_stack(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager.free_slots_stack_deltakv_full = torch.arange(16, dtype=torch.int32)
        manager._num_free_slots_deltakv_full = 16
        manager._deltakv_temp_full_reserve = 0

        slots = DeltaKVCacheManager._allocate_temp_deltakv_full(manager, 4)

        self.assertEqual(manager._num_free_slots_deltakv_full, 12)
        torch.testing.assert_close(slots, torch.tensor([12, 13, 14, 15], dtype=torch.int32))

        manager.free_slots_stack_deltakv_full[12:16] = torch.tensor([1, 1, 1, 1], dtype=torch.int32)
        torch.testing.assert_close(slots, torch.tensor([12, 13, 14, 15], dtype=torch.int32))

    def test_layer_attention_end_triggers_layer_local_staging_compression(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager._deltakv_prefill_staging_active = True
        manager.deltakv_layer_to_idx = {0: 0}
        manager.deltakv_layer_ids = [0]
        manager._deltakv_full_prefill_compressed_layers = set()
        manager._deltakv_full_prefill_plans = {}
        calls = []

        def compress(layer_idx):
            calls.append(layer_idx)
            manager._deltakv_full_prefill_compressed_layers.add(layer_idx)

        manager._deltakv_compress_full_prefill_layer = compress

        DeltaKVCacheManager.on_layer_attention_end(manager, 0)

        self.assertEqual(calls, [0])
        self.assertFalse(manager._deltakv_prefill_staging_active)


class DeltaKVLessMemoryStorageContractTest(unittest.TestCase):
    def test_compressor_residual_quant_group_size_uses_payload_dim(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.head_dim = 128
        manager.config = SimpleNamespace(kv_quant_group_size=0, use_compression=True)

        self.assertEqual(DeltaKVLessMemoryCacheManager._quant_group_size(manager, 1024), 1024)

        manager.config.kv_quant_group_size = 32
        self.assertEqual(DeltaKVLessMemoryCacheManager._quant_group_size(manager, 1024), 32)

    def test_compressed_residual_default_quant_group_size_uses_payload_dim(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.head_dim = 128
        manager.config = SimpleNamespace(kv_quant_group_size=0, use_compression=True)

        self.assertEqual(DeltaKVLessMemoryCacheManager._quant_group_size(manager, 256), 256)

    def test_context_does_not_own_attention_transients(self):
        from sparsevllm.utils.context import get_context, reset_context, set_context

        reset_context()
        set_context(is_prefill=True)

        ctx = get_context()
        for name in (
            "pre_qk_norm_k",
            "pre_rope_k",
            "pre_rope_v",
            "full_layer_k_post_rope_for_store",
            "full_layer_q_post_rope_for_score",
        ):
            self.assertFalse(hasattr(ctx, name), name)
        reset_context()

    def test_delta_quant_full_kivi_stages_first_prefill_even_below_chunk_threshold(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            chunk_prefill_size=32768,
            enable_full_layer_kivi_quant=True,
            full_layer_kv_quant_bits=4,
        )
        manager.deltakv_layer_ids = [2]

        seq = seq_with_len(11766)
        seq.current_chunk_size = 11766

        self.assertTrue(DeltaKVLessMemoryCacheManager._should_use_full_prefill_staging(manager, [seq]))

        partial = seq_with_len(11766)
        partial.current_chunk_size = 4096
        self.assertFalse(DeltaKVLessMemoryCacheManager._should_use_full_prefill_staging(manager, [partial]))

    def test_delta_quant_full_kivi_requests_full_prefill_when_step_slots_are_too_small(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            enable_full_layer_kivi_quant=True,
            full_layer_kv_quant_bits=4,
        )
        manager.deltakv_layer_ids = [2]
        manager.deltakv_prefill_staging_num_slots = 32768
        manager.prefill_step_free_slots = lambda: 32

        seq = seq_with_len(11766)
        self.assertTrue(DeltaKVLessMemoryCacheManager.should_schedule_full_prefill(manager, seq))

        tiny = seq_with_len(16)
        self.assertFalse(DeltaKVLessMemoryCacheManager.should_schedule_full_prefill(manager, tiny))

    def test_delta_quant_raw_overhead_does_not_depend_on_prefill_chunk(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        max_seqs = 8
        sink = 8
        recent = 32
        top_decode = 1342

        persistent = DeltaKVLessMemoryCacheManager._resident_sparse_raw_overhead_slots(
            max_seqs,
            sink,
            recent,
        )
        scratch = DeltaKVLessMemoryCacheManager._decode_reconstruct_scratch_slots(
            max_seqs,
            top_decode,
            sink,
            recent,
        )

        self.assertEqual(persistent, max_seqs * (sink + 2 * recent + 1))
        self.assertEqual(scratch, max_seqs * 2 * top_decode)

    def test_deltakv_storage_hooks_keep_sparse_raw_and_full_postrope(self):
        manager = object.__new__(DeltaKVCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        stores = []
        rope_hooks = []
        slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
        manager._store_layer_kv = lambda layer_idx, k, v: stores.append((layer_idx, k, v)) or slot_mapping
        manager.on_kv_stored = (
            lambda layer_idx, k, slots, **kwargs: rope_hooks.append((layer_idx, k, slots, kwargs))
        )

        raw_k = torch.full((2, 1, 4), 3.0)
        raw_v = torch.full((2, 1, 4), 4.0)
        postrope_k = torch.ones((2, 1, 4))
        value = torch.full((2, 1, 4), 2.0)

        DeltaKVCacheManager.save_raw_kv_if_needed(manager, 1, raw_k, raw_v)
        DeltaKVCacheManager.save_rope_kv_if_needed(manager, 1, postrope_k, value)

        self.assertEqual(len(stores), 1)
        self.assertEqual(stores[0][0], 1)
        self.assertIs(stores[0][1], raw_k)
        self.assertIs(stores[0][2], raw_v)
        self.assertEqual(len(rope_hooks), 0)

        DeltaKVCacheManager.save_rope_kv_if_needed(manager, 0, postrope_k, value)

        self.assertEqual(len(stores), 2)
        self.assertEqual(stores[1][0], 0)
        self.assertIs(stores[1][1], postrope_k)
        self.assertIs(stores[1][2], value)
        self.assertEqual(len(rope_hooks), 1)

    def test_deltakv_materializes_raw_sparse_view_before_attention(self):
        from sparsevllm.layers.rotary_embedding import apply_rotary_emb
        from sparsevllm.utils.context import reset_context, set_context

        reset_context()
        manager = object.__new__(DeltaKVCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.num_kv_heads = 1
        manager.head_dim = 4
        manager.deltakv_full_kv_cache = torch.empty((2, 1, 4, 1, 4), dtype=torch.float32)
        manager.deltakv_full_kv_cache[0, 0] = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 1.0]],
                [[9.0, 9.0, 9.0, 9.0]],
                [[0.0, 0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0, 0.0]],
            ]
        )
        manager.deltakv_full_kv_cache[1, 0] = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
        manager.deltakv_slot_to_pos = torch.tensor([2, 1, -1, -1], dtype=torch.int32)
        manager._deltakv_postrope_slot_marker = torch.zeros((1, 4), dtype=torch.int32)
        cos = torch.tensor(
            [
                [1.0, 1.0],
                [0.9, 0.8],
                [0.7, 0.6],
            ],
            dtype=torch.float32,
        )
        sin = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.3, 0.4],
            ],
            dtype=torch.float32,
        )
        manager.cos_sin_cache = torch.cat([cos, sin], dim=-1).unsqueeze(1)
        manager._allocate_temp_deltakv_full = lambda size: torch.tensor([2, 3], dtype=torch.int32)[:size]

        set_context(is_prefill=False, cache_manager=manager)
        active_slots = torch.tensor([[0, 1]], dtype=torch.int32)
        context_lens = torch.tensor([2], dtype=torch.int32)
        out_active, temp_slots = DeltaKVCacheManager._materialize_deltakv_active_postrope_view(
            manager,
            1,
            active_slots.clone(),
            context_lens,
            already_postrope_slots=torch.tensor([1], dtype=torch.int32),
        )

        self.assertTrue(torch.equal(out_active, torch.tensor([[2, 1]], dtype=torch.int32)))
        self.assertEqual(int(temp_slots.numel()), 0)
        raw_k = torch.tensor([[[1.0, 0.0, 0.0, 1.0]]], dtype=torch.float32)
        cos_sin = manager.cos_sin_cache[torch.tensor([2])]
        expected_cos, expected_sin = cos_sin.chunk(2, dim=-1)
        expected_k = apply_rotary_emb(raw_k, expected_cos, expected_sin)
        torch.testing.assert_close(manager.deltakv_full_kv_cache[0, 0, 2], expected_k[0])
        torch.testing.assert_close(manager.deltakv_full_kv_cache[1, 0, 2], manager.deltakv_full_kv_cache[1, 0, 0])
        self.assertEqual(int(manager.deltakv_slot_to_pos[2]), 2)
        reset_context()

    def test_delta_quant_sparse_store_uses_raw_space_only_outside_staging(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.has_prefill_staging_view = lambda layer_idx: False

        postrope_k = torch.ones((2, 1, 4))
        value = torch.full((2, 1, 4), 2.0)
        raw_k = torch.full((2, 1, 4), 3.0)
        raw_v = torch.full((2, 1, 4), 4.0)

        store_k, store_v = DeltaKVLessMemoryCacheManager.get_layer_store_tensors(
            manager,
            1,
            k_post_rope=postrope_k,
            v=value,
            pre_rope_k=raw_k,
            pre_rope_v=raw_v,
        )
        self.assertIs(store_k, raw_k)
        self.assertIs(store_v, raw_v)

        full_k, full_v = DeltaKVLessMemoryCacheManager.get_layer_store_tensors(
            manager,
            0,
            k_post_rope=postrope_k,
            v=value,
            pre_rope_k=raw_k,
            pre_rope_v=raw_v,
        )
        self.assertIs(full_k, postrope_k)
        self.assertIs(full_v, value)

        manager.has_prefill_staging_view = lambda layer_idx: True
        staging_k, staging_v = DeltaKVLessMemoryCacheManager.get_layer_store_tensors(
            manager,
            1,
            k_post_rope=postrope_k,
            v=value,
            pre_rope_k=raw_k,
            pre_rope_v=raw_v,
        )
        self.assertIs(staging_k, postrope_k)
        self.assertIs(staging_v, value)

    def test_delta_quant_sparse_store_uses_explicit_pre_rope_state(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager
        from sparsevllm.utils.context import reset_context, set_context

        reset_context()
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.has_prefill_staging_view = lambda layer_idx: False

        postrope_k = torch.ones((2, 1, 4))
        value = torch.full((2, 1, 4), 2.0)
        raw_k = torch.full((2, 1, 4), 3.0)
        raw_v = torch.full((2, 1, 4), 4.0)

        set_context(is_prefill=True, cache_manager=manager)
        store_k, store_v = DeltaKVLessMemoryCacheManager.get_layer_store_tensors(
            manager,
            1,
            k_post_rope=postrope_k,
            v=value,
            pre_rope_k=raw_k,
            pre_rope_v=raw_v,
        )

        self.assertIs(store_k, raw_k)
        self.assertIs(store_v, raw_v)
        reset_context()

    def test_delta_quant_storage_hooks_separate_raw_and_rope_paths(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.has_prefill_staging_view = lambda layer_idx: False
        manager._prefill_pre_rope_stage_active = lambda: False
        slot_mapping = torch.tensor([0, 1], dtype=torch.int64)
        stores = []
        raw_hooks = []
        rope_hooks = []
        manager._store_layer_kv = lambda layer_idx, k, v: stores.append((layer_idx, k, v)) or slot_mapping
        manager.on_pre_rope_kv_stored = (
            lambda layer_idx, k, v, slots: raw_hooks.append((layer_idx, k, v, slots))
        )
        manager.on_kv_stored = (
            lambda layer_idx, k, slots, **kwargs: rope_hooks.append((layer_idx, k, slots, kwargs))
        )

        postrope_k = torch.ones((2, 1, 4))
        value = torch.full((2, 1, 4), 2.0)
        raw_k = torch.full((2, 1, 4), 3.0)
        raw_v = torch.full((2, 1, 4), 4.0)

        DeltaKVLessMemoryCacheManager.save_raw_kv_if_needed(manager, 1, raw_k, raw_v)
        DeltaKVLessMemoryCacheManager.save_rope_kv_if_needed(manager, 1, postrope_k, value)

        self.assertEqual(len(stores), 1)
        self.assertEqual(stores[0][0], 1)
        self.assertIs(stores[0][1], raw_k)
        self.assertIs(stores[0][2], raw_v)
        self.assertEqual(len(raw_hooks), 1)
        self.assertEqual(len(rope_hooks), 0)

        DeltaKVLessMemoryCacheManager.save_rope_kv_if_needed(manager, 0, postrope_k, value)

        self.assertEqual(len(stores), 2)
        self.assertEqual(stores[1][0], 0)
        self.assertIs(stores[1][1], postrope_k)
        self.assertIs(stores[1][2], value)
        self.assertEqual(len(rope_hooks), 1)

    def test_full_layer_kivi_prefill_does_not_allocate_fp32_shadow_staging(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager
        from sparsevllm.utils.context import reset_context

        reset_context()
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager._full_layer_kivi_enabled = lambda: True
        manager.full_layer_to_idx = {0: 0}
        manager.has_prefill_staging_view = lambda layer_idx: True
        manager.full_layer_k_norm_weight = None
        manager.deltakv_prefill_staging_num_slots = 3
        manager.full_layer_kivi_prefill_k_cache_fp32 = None

        postrope_k = torch.arange(12, dtype=torch.float32).view(3, 1, 4)
        slot_mapping = torch.tensor([0, -1, 2], dtype=torch.int64)

        DeltaKVLessMemoryCacheManager.on_kv_stored(manager, 0, postrope_k, slot_mapping)

        self.assertIsNone(manager.full_layer_kivi_prefill_k_cache_fp32)
        reset_context()

    def test_full_layer_kivi_prefill_compute_uses_high_precision_staging(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager
        from sparsevllm.utils.context import reset_context

        reset_context()
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.full_layer_to_idx = {0: 0}
        manager._full_layer_kivi_enabled = lambda: True
        manager.has_prefill_staging_view = lambda layer_idx: True

        postrope_k = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
        value = torch.arange(100, 116, dtype=torch.float32).view(4, 1, 4)
        manager.deltakv_prefill_staging_kv_cache = [postrope_k.clone(), value.clone()]

        k_compute, v_compute = DeltaKVLessMemoryCacheManager.get_layer_compute_tensors(manager, 0)

        self.assertIs(k_compute, manager.deltakv_prefill_staging_kv_cache[0])
        self.assertIs(v_compute, manager.deltakv_prefill_staging_kv_cache[1])
        torch.testing.assert_close(manager.deltakv_prefill_staging_kv_cache[0], postrope_k)
        torch.testing.assert_close(manager.deltakv_prefill_staging_kv_cache[1], value)
        reset_context()

    def test_delta_quant_materializes_raw_sparse_cache_for_attention(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager
        from sparsevllm.layers.rotary_embedding import apply_rotary_emb

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.has_prefill_staging_view = lambda layer_idx: False
        manager.num_kv_heads = 1
        manager.head_dim = 4
        manager.deltakv_materialized_compute_num_slots = 4
        manager.deltakv_full_kv_cache = torch.empty((2, 1, 4, 1, 4), dtype=torch.float32)
        manager.deltakv_full_kv_cache[0, 0] = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 1.0]],
                [[0.5, 0.5, 1.0, 0.0]],
                [[0.0, 1.0, 0.5, 0.5]],
                [[1.0, 1.0, 1.0, 1.0]],
            ]
        )
        manager.deltakv_full_kv_cache[1, 0] = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
        manager.deltakv_slot_to_pos = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        cos = torch.tensor(
            [
                [1.0, 1.0],
                [0.9, 0.8],
                [0.7, 0.6],
                [0.5, 0.4],
            ],
            dtype=torch.float32,
        )
        sin = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],
            dtype=torch.float32,
        )
        manager.cos_sin_cache = torch.cat([cos, sin], dim=-1).unsqueeze(1)
        manager.deltakv_materialized_kv_cache = torch.empty((2, 4, 1, 4), dtype=torch.float32)
        manager._deltakv_materialized_active_slots = None
        manager._deltakv_materialized_local_req = None

        active_slots = torch.tensor([[2, 1]], dtype=torch.int32)
        context_lens = torch.tensor([2], dtype=torch.int32)
        k_cache, v_cache, local_active, local_req, out_lens = DeltaKVLessMemoryCacheManager.get_layer_compute_view(
            manager,
            1,
            active_slots=active_slots,
            req_indices=torch.tensor([7], dtype=torch.int32),
            context_lens=context_lens,
            selection=None,
        )

        raw_k = manager.deltakv_full_kv_cache[0, 0, active_slots.reshape(-1).long()]
        raw_v = manager.deltakv_full_kv_cache[1, 0, active_slots.reshape(-1).long()]
        pos = manager.deltakv_slot_to_pos[active_slots.reshape(-1).long()].long()
        cos_sin = manager.cos_sin_cache[pos]
        expected_cos, expected_sin = cos_sin.chunk(2, dim=-1)
        expected_k = apply_rotary_emb(raw_k, expected_cos, expected_sin)

        self.assertTrue(torch.equal(local_active, torch.tensor([[0, 1]], dtype=torch.int32)))
        self.assertTrue(torch.equal(local_req, torch.tensor([0], dtype=torch.int32)))
        self.assertIs(out_lens, context_lens)
        self.assertTrue(torch.allclose(k_cache, expected_k))
        self.assertTrue(torch.equal(v_cache, raw_v))

    def test_delta_quant_materialized_view_does_not_rerope_postrope_slots(self):
        from sparsevllm.engine.cache_manager.deltakv_less_memory import DeltaKVLessMemoryCacheManager
        from sparsevllm.layers.rotary_embedding import apply_rotary_emb

        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.deltakv_layer_to_idx = {1: 0}
        manager.has_prefill_staging_view = lambda layer_idx: False
        manager.num_kv_heads = 1
        manager.head_dim = 4
        manager.deltakv_materialized_compute_num_slots = 4
        manager.deltakv_full_kv_cache = torch.empty((2, 1, 4, 1, 4), dtype=torch.float32)
        manager.deltakv_full_kv_cache[0, 0] = torch.tensor(
            [
                [[1.0, 0.0, 0.0, 1.0]],
                [[0.5, 0.5, 1.0, 0.0]],
                [[0.0, 1.0, 0.5, 0.5]],
                [[1.0, 1.0, 1.0, 1.0]],
            ]
        )
        manager.deltakv_full_kv_cache[1, 0] = torch.arange(16, dtype=torch.float32).view(4, 1, 4)
        manager.deltakv_slot_to_pos = torch.tensor([0, 1, 2, 3], dtype=torch.int32)
        cos = torch.tensor(
            [
                [1.0, 1.0],
                [0.9, 0.8],
                [0.7, 0.6],
                [0.5, 0.4],
            ],
            dtype=torch.float32,
        )
        sin = torch.tensor(
            [
                [0.0, 0.0],
                [0.1, 0.2],
                [0.3, 0.4],
                [0.5, 0.6],
            ],
            dtype=torch.float32,
        )
        manager.cos_sin_cache = torch.cat([cos, sin], dim=-1).unsqueeze(1)
        manager.deltakv_materialized_kv_cache = torch.empty((2, 4, 1, 4), dtype=torch.float32)
        manager._deltakv_materialized_active_slots = None
        manager._deltakv_materialized_local_req = None
        manager._deltakv_postrope_slot_mask = torch.zeros((1, 4), dtype=torch.bool)
        manager._deltakv_postrope_slot_mask[0, 1] = True

        active_slots = torch.tensor([[2, 1]], dtype=torch.int32)
        context_lens = torch.tensor([2], dtype=torch.int32)
        k_cache, _, _, _, _ = DeltaKVLessMemoryCacheManager.get_layer_compute_view(
            manager,
            1,
            active_slots=active_slots,
            req_indices=torch.tensor([7], dtype=torch.int32),
            context_lens=context_lens,
            selection=None,
        )

        raw_k = manager.deltakv_full_kv_cache[0, 0, active_slots.reshape(-1).long()]
        pos = manager.deltakv_slot_to_pos[active_slots.reshape(-1).long()].long()
        cos_sin = manager.cos_sin_cache[pos]
        expected_cos, expected_sin = cos_sin.chunk(2, dim=-1)
        expected_raw_rope = apply_rotary_emb(raw_k, expected_cos, expected_sin)
        expected = expected_raw_rope.clone()
        expected[1] = raw_k[1]

        self.assertTrue(torch.allclose(k_cache, expected))


class DeltaKVStaticDecodeRobustnessTest(unittest.TestCase):
    def test_no_graph_static_workspace_accepts_auto_capture_sizes(self):
        manager = object.__new__(DeltaKVLessMemoryCudaGraphCacheManager)
        manager.config = SimpleNamespace(
            max_decoding_seqs=16,
            decode_cuda_graph_capture_sizes="auto",
        )

        self.assertEqual(manager._decode_graph_capture_size_capacity(3), 16)

    def test_kivi_short_batch_prefill_does_not_use_singleton_staging_slots(self):
        manager = object.__new__(DeltaKVLessMemoryCacheManager)
        manager.config = SimpleNamespace(
            prefill_schedule_policy=PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
            chunk_prefill_size=8192,
        )
        manager.deltakv_layer_ids = [1]
        manager._full_layer_kivi_enabled = lambda: True
        manager._deltakv_less_memory_prepare_full_prefill_staging = False
        seq = seq_with_len(1024)
        seq.current_chunk_size = 1024

        self.assertFalse(manager._should_stage_full_layer_kivi_prefill(seq, 1024))

        manager._deltakv_less_memory_prepare_full_prefill_staging = True
        self.assertTrue(manager._should_stage_full_layer_kivi_prefill(seq, 1024))


if __name__ == "__main__":
    unittest.main()
