import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sparsevllm.config import Config
from sparsevllm.engine.scheduler import Scheduler
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.sparse_controller import SparseController
from sparsevllm.layers.rotary_embedding import get_rope
from sparsevllm.layers.seq_chunk import apply_seq_chunked
from sparsevllm.sampling_params import SamplingParams


class _FakeMemoryOracle:
    num_free_slots = 1_000_000

    def reserved_prefill_slots(self, waiting, chunk_prefill_size):
        return 0

    def prompt_admission_budgets(self, waiting, chunk_prefill_size):
        return {"slots": self.num_free_slots}

    def prefill_batched_tokens_margin(self):
        return 0

    def remaining_prefill_tokens(self, seq):
        return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)

    def prompt_admission_costs(self, seq):
        return {"slots": int(seq.num_prompt_tokens)}

    def prompt_admission_failure_action(self):
        return "raise"

    def on_prompt_admitted(self, seq, costs):
        return None

    def prompt_logical_reservation_cost(self, seq):
        return 0

    def free_slot_stats(self):
        return {"free_slots": self.num_free_slots}

    def debug_live_seq_slots(self):
        return {}


def _hf_config():
    return SimpleNamespace(
        model_type="qwen2",
        torch_dtype=torch.float16,
        max_position_embeddings=128_000,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
    )


def _config(**kwargs):
    defaults = dict(
        max_model_len=128,
        max_num_batched_tokens=16,
        max_num_seqs_in_batch=4,
        max_decoding_seqs=8,
        chunk_prefill_size=8,
        num_sink_tokens=1,
        num_recent_tokens=1,
        num_top_tokens=4,
        num_top_tokens_in_prefill=4,
        throughput_log_interval_s=0,
    )
    defaults.update(kwargs)
    with tempfile.TemporaryDirectory() as tmp:
        with patch("sparsevllm.config.AutoConfig.from_pretrained", return_value=_hf_config()):
            return Config(model=tmp, **defaults)


def _seq(length):
    return Sequence(list(range(length)), SamplingParams(max_tokens=1, temperature=0.0))


class SparseVLLMPrefillScheduleTests(unittest.TestCase):
    def test_long_policy_runs_long_prefill_as_single_full_chunk(self):
        cfg = _config(prefill_schedule_policy="long_bs1full_short_batch")
        scheduler = Scheduler(cfg, _FakeMemoryOracle())
        long_seq = _seq(40)
        scheduler.add(long_seq)

        seqs, is_prefill, preempted = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(preempted, [])
        self.assertEqual(seqs, [long_seq])
        self.assertEqual(long_seq.current_chunk_size, 40)

    def test_long_policy_keeps_short_prefill_batched(self):
        cfg = _config(prefill_schedule_policy="long_bs1full_short_batch")
        scheduler = Scheduler(cfg, _FakeMemoryOracle())
        short_a = _seq(6)
        short_b = _seq(6)
        scheduler.add(short_a)
        scheduler.add(short_b)

        seqs, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [short_a, short_b])
        self.assertEqual([seq.current_chunk_size for seq in seqs], [6, 6])

    def test_all_chunked_preserves_chunk_prefill_for_long_inputs(self):
        cfg = _config(prefill_schedule_policy="all_chunked")
        scheduler = Scheduler(cfg, _FakeMemoryOracle())
        long_seq = _seq(40)
        scheduler.add(long_seq)

        seqs, is_prefill, _ = scheduler.schedule()

        self.assertTrue(is_prefill)
        self.assertEqual(seqs, [long_seq])
        self.assertEqual(long_seq.current_chunk_size, 8)

    def test_invalid_prefill_schedule_policy_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "prefill_schedule_policy"):
            _config(prefill_schedule_policy="bad-policy")

    def test_deltakv_all_chunked_logs_accuracy_warning(self):
        with patch("sparsevllm.config.logger.warning") as warning:
            _config(vllm_sparse_method="deltakv-delta-quant", prefill_schedule_policy="all_chunked")

        self.assertTrue(warning.called)
        self.assertIn("all_chunked", warning.call_args.args[0])

    def test_seq_chunk_helper_preserves_tokenwise_output(self):
        torch.manual_seed(0)
        linear = torch.nn.Sequential(
            torch.nn.Linear(3, 7),
            torch.nn.GELU(),
            torch.nn.Linear(7, 5),
        )
        x = torch.randn(11, 3)

        expected = linear(x)
        actual = apply_seq_chunked(x, 4, linear)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-6, rtol=1e-6))

    def test_deltakv_single_pass_prefill_guard(self):
        single = _seq(9)
        single.current_chunk_size = 9
        batched = _seq(7)
        batched.current_chunk_size = 7
        self.assertTrue(SparseController._is_single_pass_prefill([single, batched]))

        chunked = _seq(9)
        chunked.current_chunk_size = 4
        self.assertFalse(SparseController._is_single_pass_prefill([chunked]))

        continued = _seq(9)
        continued.num_prefilled_tokens = 4
        continued.current_chunk_size = 5
        self.assertFalse(SparseController._is_single_pass_prefill([continued]))

    def test_llama3_rope_scaling_matches_transformers_parameters(self):
        from transformers import LlamaConfig
        from transformers.modeling_rope_utils import _compute_llama3_parameters

        rope_scaling = {
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
            "rope_type": "llama3",
        }
        rope = get_rope(
            head_size=128,
            rotary_dim=128,
            max_position=16,
            base=500000.0,
            rope_scaling=rope_scaling,
        )
        cached = get_rope(
            head_size=128,
            rotary_dim=128,
            max_position=16,
            base=500000.0,
            rope_scaling=dict(reversed(list(rope_scaling.items()))),
        )
        self.assertIs(rope, cached)
        cfg = LlamaConfig(
            hidden_size=4096,
            num_attention_heads=32,
            max_position_embeddings=131072,
            rope_theta=500000.0,
            rope_scaling=rope_scaling,
        )
        inv_freq, _ = _compute_llama3_parameters(cfg, device=torch.device("cpu"))
        positions = torch.arange(16, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", positions, inv_freq)
        expected = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze(1)

        self.assertTrue(torch.allclose(rope.cos_sin_cache, expected, atol=1e-6, rtol=1e-6))


if __name__ == "__main__":
    unittest.main()
