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

from sparsevllm.engine.llm_engine import LLMEngine
from sparsevllm.engine.model_runner import (
    DEFAULT_TP_RPC_SHM_BYTES,
    ModelRunner,
    TP_SHM_NAME_PREFIX,
    make_tp_shm_name,
    resolve_tp_rpc_shm_size,
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


def test_tp_rpc_shm_size_can_be_overridden():
    with patch.dict(os.environ, {}, clear=True):
        assert resolve_tp_rpc_shm_size() == DEFAULT_TP_RPC_SHM_BYTES

    with patch.dict(os.environ, {"SPARSEVLLM_TP_RPC_SHM_BYTES": str(64 * 1024 * 1024)}):
        assert resolve_tp_rpc_shm_size() == 64 * 1024 * 1024


def test_tp_rpc_shm_size_rejects_invalid_values():
    with patch.dict(os.environ, {"SPARSEVLLM_TP_RPC_SHM_BYTES": "bad"}):
        try:
            resolve_tp_rpc_shm_size()
        except ValueError as exc:
            assert "must be an integer" in str(exc)
        else:
            raise AssertionError("expected invalid shm size to fail")

    with patch.dict(os.environ, {"SPARSEVLLM_TP_RPC_SHM_BYTES": "128"}):
        try:
            resolve_tp_rpc_shm_size()
        except ValueError as exc:
            assert "must be >= 4096" in str(exc)
        else:
            raise AssertionError("expected tiny shm size to fail")


def test_free_slots_batch_releases_each_seq_id():
    freed: list[int] = []

    class FakeCacheManager:
        def free_seq(self, seq_id: int):
            freed.append(int(seq_id))

    runner = object.__new__(ModelRunner)
    runner.cache_manager = FakeCacheManager()

    ModelRunner.free_slots_batch(runner, [3, 5, 8])

    assert freed == [3, 5, 8]


def test_free_rank_slots_batch_releases_only_local_owner_ids():
    freed: list[int] = []
    runner = object.__new__(ModelRunner)
    runner.config = SimpleNamespace(data_parallel_size=3)
    runner.world_size = 3
    runner.rank = 1
    runner.free_slots_batch = lambda seq_ids: freed.extend(int(seq_id) for seq_id in seq_ids)

    ModelRunner.free_rank_slots_batch(runner, [[3], [5, 8], [13]])

    assert freed == [5, 8]


def test_llm_engine_native_dp_free_routes_seq_ids_to_owner_ranks():
    calls: list[tuple[str, list[list[int]]]] = []
    engine = object.__new__(LLMEngine)
    engine.config = SimpleNamespace(
        data_parallel_size=3,
        parallel_world_size=3,
    )
    engine.model_runner = SimpleNamespace(call=lambda method, arg: calls.append((method, arg)))
    engine._dp_seq_owner = {10: 2, 11: 0}

    LLMEngine._free_seq_slots(engine, [10, 11])

    assert calls == [("free_rank_slots_batch", [[11], [], [10]])]
    assert engine._dp_seq_owner == {}


def test_prefix_cache_control_rpc_reports_any_tp_worker_failure():
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.device = torch.device("cpu")

    def mark_failed(tensor, op=None):
        assert op == dist.ReduceOp.MAX
        tensor.fill_(1)

    with patch.object(dist, "is_initialized", return_value=True), patch.object(dist, "all_reduce", side_effect=mark_failed):
        try:
            ModelRunner._sync_prefix_cache_control_rpc_status(runner, "prefix_cache_delete_subtree", None)
        except RuntimeError as exc:
            assert "At least one TP worker failed" in str(exc)
        else:
            raise AssertionError("expected worker failure to be surfaced on rank 0")


def test_run_rpc_reports_any_tp_worker_failure():
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.device = torch.device("cpu")

    def mark_failed(tensor, op=None):
        assert op == dist.ReduceOp.MAX
        tensor.fill_(1)

    with patch.object(dist, "is_initialized", return_value=True), patch.object(dist, "all_reduce", side_effect=mark_failed):
        try:
            ModelRunner._sync_tp_rpc_status(runner, "run", None)
        except RuntimeError as exc:
            assert "At least one TP worker failed during run" in str(exc)
        else:
            raise AssertionError("expected worker failure to be surfaced on rank 0")


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
    runner.cache_manager = SimpleNamespace(
        on_forward_end=lambda seqs, is_prefill: calls.append(f"cache_post:{is_prefill}")
    )
    runner.sampler = lambda *args, **kwargs: calls.append("sample")
    runner._collect_logprobs = lambda *args, **kwargs: calls.append("logprobs")

    token_ids, logprobs = ModelRunner.run(runner, [SimpleNamespace()], is_prefill=False)

    assert token_ids is None
    assert logprobs is None
    assert calls == ["decode", "sparse_post:False", "cache_post:False"]
