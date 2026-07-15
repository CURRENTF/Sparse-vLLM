import os
import threading
import time
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from unittest.mock import patch
from uuid import uuid4

import torch
import torch.distributed as dist

from sparsevllm.engine.model_runner import (
    ModelRunner,
    PREFIX_CACHE_CONTROL_RPC_METHODS,
    TP_RPC_STATUS_SYNC_METHODS,
    TP_SHM_NAME_PREFIX,
    make_tp_shm_name,
)


def test_write_shm_waits_until_worker_reads_command():
    ctx = get_context("spawn")
    event = ctx.Event()
    shm = SharedMemory(
        name=f"sparsevllm_test_rpc_{os.getpid()}_{uuid4().hex}",
        create=True,
        size=2**20,
    )
    rank0 = SimpleNamespace(world_size=2, rank=0, event=[event], shm=shm)
    worker = SimpleNamespace(world_size=2, rank=1, event=event, shm=shm)
    errors: list[BaseException] = []

    def write_command():
        try:
            ModelRunner.write_shm(rank0, "free_slots", 123)
        except BaseException as exc:  # pragma: no cover - surfaced below.
            errors.append(exc)

    writer = threading.Thread(target=write_command)
    writer.start()

    try:
        assert event.wait(timeout=1.0)
        time.sleep(0.02)
        assert writer.is_alive()

        method_name, args = ModelRunner.read_shm(worker)
        writer.join(timeout=1.0)

        assert not writer.is_alive()
        assert errors == []
        assert method_name == "free_slots"
        assert args == [123]
    finally:
        if event.is_set():
            event.clear()
        writer.join(timeout=1.0)
        shm.close()
        shm.unlink()


def test_tp_shm_name_is_unique_per_engine_instance():
    names = {make_tp_shm_name() for _ in range(3)}

    assert len(names) == 3
    assert all(name.startswith(TP_SHM_NAME_PREFIX) for name in names)
    assert "sparsevllm" not in names


def test_free_slots_batch_releases_each_seq_id():
    freed: list[int] = []

    class FakeRuntimeState:
        def free_seq(self, seq_id: int):
            freed.append(int(seq_id))

    runner = object.__new__(ModelRunner)
    runner.runtime_state = FakeRuntimeState()

    ModelRunner.free_slots_batch(runner, [3, 5, 8])

    assert freed == [3, 5, 8]


def test_prefix_cache_control_rpc_reports_any_tp_worker_failure():
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.device = torch.device("cpu")
    runner.parallel_context = SimpleNamespace(
        world_all_reduce=lambda tensor, op: dist.all_reduce(tensor, op=op)
    )

    def mark_failed(tensor, op=None):
        assert op == dist.ReduceOp.MAX
        tensor.fill_(1)

    with patch.object(dist, "is_initialized", return_value=True), patch.object(dist, "all_reduce", side_effect=mark_failed):
        try:
            ModelRunner._sync_prefix_cache_control_rpc_status(runner, "prefix_cache_delete_subtree", None)
        except RuntimeError as exc:
            assert "At least one world worker failed" in str(exc)
        else:
            raise AssertionError("expected worker failure to be surfaced on rank 0")


def test_prefix_cache_match_uses_tp_failure_synchronized_control_path():
    assert "prefix_cache_match" in PREFIX_CACHE_CONTROL_RPC_METHODS
    assert "prefix_cache_match" in TP_RPC_STATUS_SYNC_METHODS


def test_run_rpc_reports_any_tp_worker_failure():
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.device = torch.device("cpu")
    runner.parallel_context = SimpleNamespace(
        world_all_reduce=lambda tensor, op: dist.all_reduce(tensor, op=op)
    )

    def mark_failed(tensor, op=None):
        assert op == dist.ReduceOp.MAX
        tensor.fill_(1)

    with patch.object(dist, "is_initialized", return_value=True), patch.object(dist, "all_reduce", side_effect=mark_failed):
        try:
            ModelRunner._sync_tp_rpc_status(runner, "run", None)
        except RuntimeError as exc:
            assert "At least one world worker failed during run" in str(exc)
        else:
            raise AssertionError("expected worker failure to be surfaced on rank 0")


def test_warmup_reset_uses_failure_synchronized_world_rpc():
    assert "reset_after_warmup" in TP_RPC_STATUS_SYNC_METHODS


def test_prefix_cache_lookup_uses_failure_synchronized_world_rpc():
    assert "refresh_prefix_cache_hit" in TP_RPC_STATUS_SYNC_METHODS


def test_prefix_cache_lookup_rpc_checks_rank_results():
    runner = object.__new__(ModelRunner)
    runner.world_size = 1
    runner.rank = 0
    calls = []
    result = {"enabled": False, "hit_len": 0}
    runner.refresh_prefix_cache_hit = lambda seq: calls.append(("lookup", seq)) or result
    runner._sync_tp_rpc_status = lambda method, error: calls.append(("status", method, error))
    runner._sync_prefix_cache_lookup_result = lambda value: calls.append(("result", value))

    seq = object()
    actual = ModelRunner.call(runner, "refresh_prefix_cache_hit", seq)

    assert actual is result
    assert calls == [
        ("lookup", seq),
        ("status", "refresh_prefix_cache_hit", None),
        ("result", result),
    ]


def test_prefix_cache_lookup_rejects_rank_divergence():
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.parallel_context = SimpleNamespace(
        world=SimpleNamespace(process_group=object())
    )

    def gather(results, local_result, group=None):
        assert group is runner.parallel_context.world.process_group
        results[:] = [local_result, {**local_result, "hit_len": 0}]

    with patch.object(dist, "all_gather_object", side_effect=gather):
        try:
            ModelRunner._sync_prefix_cache_lookup_result(
                runner,
                {"enabled": True, "hit_len": 8},
            )
        except RuntimeError as exc:
            assert "lookup diverged across world ranks" in str(exc)
        else:
            raise AssertionError("expected divergent prefix-cache lookup to fail")


def test_model_runner_prefix_cache_lookup_returns_sequence_metadata():
    runner = object.__new__(ModelRunner)

    def refresh(seq):
        seq.prefix_cache_enabled = True
        seq.prefix_cache_hit_len = 8
        seq.prefix_cache_hit_block_count = 2
        seq.prefix_cache_hit_last_block_id = b"block"
        seq.prefix_cache_block_size = 4
        seq.prefix_cache_method = "quest"

    runner.runtime_state = SimpleNamespace(refresh_prefix_cache_hit=refresh)
    seq = SimpleNamespace(
        prefix_cache_enabled=False,
        prefix_cache_hit_len=0,
        prefix_cache_hit_block_count=0,
        prefix_cache_hit_last_block_id=None,
        prefix_cache_block_size=0,
        prefix_cache_method="",
    )

    result = ModelRunner.refresh_prefix_cache_hit(runner, seq)

    assert result == {
        "enabled": True,
        "hit_len": 8,
        "hit_block_count": 2,
        "hit_last_block_id": b"block",
        "block_size": 4,
        "method": "quest",
    }


def test_hidden_state_debug_uses_failure_synchronized_world_rpc():
    assert "debug_hidden_states_cpu" in TP_RPC_STATUS_SYNC_METHODS
    assert "debug_moe_states_cpu" in TP_RPC_STATUS_SYNC_METHODS


def test_model_runner_reset_after_warmup_resets_local_runtime_state():
    calls = []
    runner = object.__new__(ModelRunner)
    runner.runtime_state = SimpleNamespace(
        reset_after_warmup=lambda: calls.append("runtime")
    )
    runner.decode_cuda_graph_runner = SimpleNamespace(
        clear_captured_graphs=lambda: calls.append("graphs")
    )
    runner.sparse_controller = SimpleNamespace(
        clear_decode_attn_score_buffers=lambda: calls.append("scores")
    )

    with patch.dict(
        os.environ,
        {
            "SPARSEVLLM_DELTAKV_CLEAR_GRAPHS_AFTER_WARMUP": "0",
            "SPARSEVLLM_DELTAKV_CLEAR_ATTN_SCORE_BUFFERS_AFTER_WARMUP": "0",
        },
    ):
        ModelRunner.reset_after_warmup(runner)

    assert calls == ["runtime"]


def test_tp_worker_decode_skips_rank0_sampling_path():
    calls: list[str] = []

    runner = object.__new__(ModelRunner)
    runner.rank = 1
    runner.config = SimpleNamespace(decode_cuda_graph=False)
    runner.decode_cuda_graph_runner = SimpleNamespace(
        run_eager_static=lambda seqs: calls.append("decode") or None
    )
    runner.sparse_controller = SimpleNamespace(
        post_forward=lambda seqs, is_prefill: calls.append(f"sparse_post:{is_prefill}")
    )
    runner.runtime_state = SimpleNamespace(
        on_forward_end=lambda seqs, is_prefill: calls.append(f"cache_post:{is_prefill}")
    )
    runner.sampler = lambda *args, **kwargs: calls.append("sample")
    runner._collect_logprobs = lambda *args, **kwargs: calls.append("logprobs")

    token_ids, logprobs = ModelRunner.run(runner, [SimpleNamespace()], is_prefill=False)

    assert token_ids is None
    assert logprobs is None
    assert calls == ["decode", "sparse_post:False", "cache_post:False"]
