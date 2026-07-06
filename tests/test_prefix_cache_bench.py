import json
import random
import types

import pytest

from deltakv.configs.runtime_params import normalize_runtime_params
from sparsevllm.config import Config
from scripts.benchmarks import bench_prefix_cache as bench


class FakeTokenizer:
    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(token_id) for token_id in token_ids)


def test_prefix_cache_bench_flags_impossible_cache_hit(tmp_path):
    spec = bench.RequestSpec(
        request_key="req",
        workload="shared_prefix",
        phase="bench",
        session_id=0,
        turn=0,
        prompt_token_ids=[1, 2, 3, 4],
        output_len=2,
        eligible_cache_tokens=4,
        expected_reuse_tokens=4,
    )
    state = bench.RequestState(
        spec=spec,
        seq_id=7,
        add_s=1.0,
        first_token_s=1.5,
        finish_s=2.0,
        generated_token_ids=[9, 10],
        prefix_cache_hit_len=8,
        prefix_cache_hit_blocks=2,
    )

    records = bench._write_request_records(
        states={7: state},
        tokenizer=FakeTokenizer(),
        per_turn_path=tmp_path / "per_turn_results.jsonl",
        raw_output_path=tmp_path / "raw_outputs.jsonl",
        batch_start_s=1.0,
        block_size=4,
    )

    assert records[0]["status"] == "metric_failed"
    assert records[0]["eligible_cache_tokens"] == 4
    assert "exceeds planned_eligible_cache_tokens" in records[0]["error_message"]


def test_prefix_cache_bench_token_plan_uses_max_bounds():
    args = types.SimpleNamespace(
        system_prompt_len=100,
        session_prefix_min_len=10,
        session_prefix_len=20,
        user_min_len=3,
        user_len=7,
        output_len=5,
        turns=3,
        shared_prefix_len=50,
        shared_suffix_min_len=4,
        shared_suffix_len=9,
    )

    plan = bench._token_count_plan(args)

    assert plan["session_prefix_min"] == 10
    assert plan["session_prefix_max"] == 20
    assert plan["user_min"] == 3
    assert plan["user_max"] == 7
    assert plan["shared_suffix_min"] == 4
    assert plan["shared_suffix_max"] == 9
    assert plan["multiturn_first_prompt"] == 127
    assert plan["multiturn_max_prompt"] == 100 + 20 + 3 * (7 + 5)
    assert plan["shared_prefix_max_prompt"] == 59


def test_prefix_cache_bench_engine_kwargs_are_sparsevllm_config_fields():
    args = types.SimpleNamespace(
        gpu_memory_utilization=0.65,
        tensor_parallel_size=1,
        max_active_requests=4,
        max_num_batched_tokens=8192,
        chunk_prefill_size=4096,
        num_sink_tokens=8,
        num_recent_tokens=256,
        num_top_tokens=2048,
        num_top_tokens_in_prefill=2048,
        chunk_prefill_accel_omnikv=True,
        full_attention_layers="0,1,2,4,7,14",
        quest_chunk_size=16,
        quest_token_budget=4096,
        prefix_cache_block_size=16,
        prefix_cache_max_blocks=None,
        prefix_cache_salt="prefix-cache-bench-test",
        output_len=128,
        max_model_len_margin=64,
        hyper_params="{}",
    )

    kwargs = bench._case_engine_kwargs(args, "baseline_full", max_prompt_len=4096)

    normalized = normalize_runtime_params(kwargs, backend="sparsevllm")
    config_fields = set(Config.__dataclass_fields__)
    unknown = sorted(set(normalized.infer_config) - config_fields)
    assert unknown == []


def test_prefix_cache_bench_sample_length_validates_bounds():
    rng = random.Random(1)

    for _ in range(20):
        assert 4 <= bench._sample_length(rng, 9, 4) <= 9
    with pytest.raises(ValueError, match="min must be <= max"):
        bench._sample_length(rng, 3, 4)


def test_prefix_cache_bench_multiturn_allows_shared_system_reuse_on_first_turn():
    assert (
        bench._multiturn_reusable_prefix_len(
            turn=0,
            session_id=0,
            shared_system_len=8192,
            history_len=12288,
        )
        == 0
    )
    assert (
        bench._multiturn_reusable_prefix_len(
            turn=0,
            session_id=2,
            shared_system_len=8192,
            history_len=12288,
        )
        == 8192
    )
    assert (
        bench._multiturn_reusable_prefix_len(
            turn=1,
            session_id=0,
            shared_system_len=8192,
            history_len=12352,
        )
        == 12352
    )


def test_prefix_cache_bench_writes_failed_samples_on_batch_error(tmp_path):
    class StuckLLM:
        def __init__(self):
            self.scheduler = type("SchedulerView", (), {"waiting": [], "decoding": []})()
            self.last_step_token_outputs = []
            self.next_seq_id = 1

        def add_request(self, prompt_token_ids, sampling_params):
            del prompt_token_ids, sampling_params
            seq_id = self.next_seq_id
            self.next_seq_id += 1
            return seq_id

        def step(self):
            return [], 0

    specs = [
        bench.RequestSpec(
            request_key=f"req_{idx}",
            workload="shared_prefix",
            phase="bench",
            session_id=idx,
            turn=0,
            prompt_token_ids=[idx, idx + 1],
            output_len=2,
            eligible_cache_tokens=0,
            expected_reuse_tokens=0,
        )
        for idx in range(2)
    ]
    per_turn_path = tmp_path / "per_turn_results.jsonl"
    raw_output_path = tmp_path / "raw_outputs.jsonl"

    with pytest.raises(RuntimeError, match="Exceeded max_steps"):
        bench._run_request_batch(
            llm=StuckLLM(),
            specs=specs,
            tokenizer=FakeTokenizer(),
            per_turn_path=per_turn_path,
            raw_output_path=raw_output_path,
            block_size=4,
            max_steps=1,
        )

    rows = [json.loads(line) for line in per_turn_path.read_text(encoding="utf-8").splitlines()]
    raw_rows = [json.loads(line) for line in raw_output_path.read_text(encoding="utf-8").splitlines()]
    assert [row["status"] for row in rows] == ["model_failed", "model_failed"]
    assert [row["request_key"] for row in rows] == ["req_0", "req_1"]
    assert len(raw_rows) == 2
