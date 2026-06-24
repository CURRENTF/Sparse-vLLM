import json

import pytest

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
