from types import SimpleNamespace

from sparsevllm.engine.llm_engine import LLMEngine
from sparsevllm.engine.startup import DeviceMemorySnapshot, MemoryProfileMeasurement


def test_engine_rebuilds_production_runtime_before_final_graph_warmup():
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        engine_prefill_chunk_size=8,
        max_model_len=32,
        max_num_batched_tokens=16,
        max_num_seqs_in_batch=4,
        max_decoding_seqs=3,
        gpu_memory_utilization=0.9,
        model_spec=SimpleNamespace(num_experts_field=None),
        hf_config=SimpleNamespace(vocab_size=128),
    )
    engine.scheduler = object()
    calls = []
    batches = []
    snapshot = DeviceMemorySnapshot(700, 1000, 300, 300)

    def profile_record(phase):
        measurement = {
            "prefill": MemoryProfileMeasurement(0, 100),
            "cuda_graph": MemoryProfileMeasurement(50, 20),
            "decode": MemoryProfileMeasurement(0, 120),
        }[phase]
        return [{"world_rank": 0, "measurement": measurement}]

    def runner_call(method, *args):
        calls.append((method, *args))
        if method == "finish_startup_memory_profile":
            return profile_record(args[0])
        if method == "release_profiling_cache_runtime":
            assert engine.scheduler is None
            return [{"world_rank": 0, "snapshot": snapshot}]
        if method == "build_production_cache_runtime":
            assert args == (430,)
            return [{"world_rank": 0, "num_kvcache_slots": 64}]
        if method == "capture_startup_memory_snapshot":
            return [{"world_rank": 0, "snapshot": snapshot}]
        return None

    engine.model_runner = SimpleNamespace(call=runner_call)
    engine._create_scheduler = lambda: "production-scheduler"

    def run_batch(prompt_lengths, sampling_params, prompt_offset):
        batches.append(
            (
                tuple(prompt_lengths),
                int(sampling_params.max_tokens),
                bool(sampling_params.ignore_eos),
            )
        )
        return prompt_offset + len(prompt_lengths)

    engine._run_startup_batch = run_batch
    engine._capture_startup_decode_graphs = lambda prompt_offset, **kwargs: (
        calls.append(("capture_graphs", prompt_offset, kwargs)) or prompt_offset
    )

    engine._warmup()

    assert engine.scheduler == "production-scheduler"
    assert [batch[1:] for batch in batches] == [
        (1, False),
        (1, False),
        (2, True),
        (2, True),
    ]
    assert [call[0] for call in calls].count("capture_graphs") == 2
    assert calls.index(("build_production_cache_runtime", 430)) < calls.index(
        ("capture_graphs", 0, {"respect_runtime_capacity": True})
    )
