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
    TP_RUN_STATUS_FAILED,
    TP_RUN_STATUS_SUCCESS,
    TP_SHM_NAME_PREFIX,
    _create_model,
    make_tp_shm_name,
)
from sparsevllm.models.spec import ModelSpec
from sparsevllm.operators import registry as operator_registry


def test_create_model_delegates_optional_runtime_binding():
    config = object()
    context = object()

    class Model:
        @classmethod
        def build_runtime_kwargs(cls, hf_config, **runtime_kwargs):
            assert hf_config is config
            assert runtime_kwargs == {"parallel_context": context}
            return {"runtime": "bound"}

        def __init__(self, hf_config, *, runtime):
            self.args = hf_config, runtime

    with patch.dict(_create_model.__globals__, {"Model": Model}):
        model = _create_model(
            config,
            ModelSpec("test", runtime_class_name="Model"),
            parallel_context=context,
        )

    assert model.args == (config, "bound")


def test_write_shm_waits_until_worker_reads_command():
    ctx = get_context("spawn")
    command_event = ctx.Event()
    completion_event = ctx.Event()
    event = (command_event, completion_event)
    shm = SharedMemory(
        name=f"sparsevllm_test_rpc_{os.getpid()}_{uuid4().hex}",
        create=True,
        size=2**20,
    )
    rank0 = SimpleNamespace(
        world_size=2,
        rank=0,
        event=[event],
        shm=shm,
        _run_status_offset=lambda rank: len(shm.buf) - 2 + rank,
    )
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
        assert command_event.wait(timeout=1.0)
        time.sleep(0.02)
        assert writer.is_alive()

        method_name, args = ModelRunner.read_shm(worker)
        writer.join(timeout=1.0)

        assert not writer.is_alive()
        assert errors == []
        assert method_name == "free_slots"
        assert args == [123]
    finally:
        if command_event.is_set():
            command_event.clear()
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


def test_operator_implementation_log_runs_only_on_rank_zero():
    rank_zero = object.__new__(ModelRunner)
    rank_zero.parallel_context = SimpleNamespace(world_rank=0)
    rank_one = object.__new__(ModelRunner)
    rank_one.parallel_context = SimpleNamespace(world_rank=1)

    with patch.object(operator_registry, "log_operator_implementations") as log_implementations:
        ModelRunner.log_operator_implementations(rank_zero)
        ModelRunner.log_operator_implementations(rank_one)

    log_implementations.assert_called_once_with()


def test_operator_runtime_stats_gather_one_record_per_world_rank():
    runner = object.__new__(ModelRunner)
    runner.world_size = 2
    runner.rank = 0
    runner.parallel_context = SimpleNamespace(
        world_rank=0,
        world=SimpleNamespace(process_group="world"),
    )
    sync_calls = []
    runner._sync_tp_rpc_status = (
        lambda method_name, error: sync_calls.append((method_name, error))
    )

    def gather(output, local, *, group):
        assert group == "world"
        output[:] = [local, {"world_rank": 1, "bindings": [], "operators": {}}]

    with (
        patch.object(operator_registry, "operator_binding_reports", return_value=[]),
        patch.object(operator_registry, "operator_runtime_stats", return_value={"MLA": []}),
        patch.object(dist, "all_gather_object", side_effect=gather),
    ):
        stats = ModelRunner.operator_runtime_stats(runner)

    assert sync_calls == [("operator_runtime_stats", None)]
    assert stats == [
        {"world_rank": 0, "bindings": [], "operators": {"MLA": []}},
        {"world_rank": 1, "bindings": [], "operators": {}},
    ]


def test_prefix_offload_release_rpc_surfaces_local_failure_after_status_sync():
    for method_name, args in (("free_slots", (7,)), ("free_slots_batch", ([7, 9],))):
        runner = object.__new__(ModelRunner)
        runner.world_size = 1
        runner.rank = 0
        expected = RuntimeError(f"{method_name} failed")
        setattr(runner, method_name, lambda *unused, error=expected: (_ for _ in ()).throw(error))
        calls = []
        runner._sync_tp_rpc_status = lambda method, error: calls.append((method, error))

        try:
            ModelRunner.call(runner, method_name, *args)
        except RuntimeError as exc:
            assert exc is expected
        else:
            raise AssertionError(f"expected {method_name} failure")
        assert calls == [(method_name, expected)]


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


def test_run_rpc_reports_any_tp_worker_failure():
    ctx = get_context("spawn")
    events = (ctx.Event(), ctx.Event())
    shm = SharedMemory(
        name=f"sparsevllm_test_status_{os.getpid()}_{uuid4().hex}",
        create=True,
        size=2**20,
    )
    rank0 = object.__new__(ModelRunner)
    rank0.world_size = 2
    rank0.rank = 0
    rank0.event = [events]
    rank0.shm = shm
    rank0.device = torch.device("cpu")
    rank0.platform = SimpleNamespace(synchronize=lambda: None)
    worker = object.__new__(ModelRunner)
    worker.world_size = 2
    worker.rank = 1
    worker.event = events
    worker.shm = shm
    worker.device = torch.device("cpu")
    worker.platform = SimpleNamespace(synchronize=lambda: None)

    try:
        ModelRunner._sync_tp_run_status(worker, RuntimeError("worker failed"))
        assert shm.buf[ModelRunner._run_status_offset(worker, 1)] == TP_RUN_STATUS_FAILED
        try:
            ModelRunner._sync_tp_run_status(rank0, None)
        except RuntimeError as exc:
            assert "TP worker rank(s) 1 failed during run" in str(exc)
        else:
            raise AssertionError("expected worker failure to be surfaced on rank 0")
    finally:
        shm.close()
        shm.unlink()


def test_run_rpc_uses_host_completion_without_collective():
    ctx = get_context("spawn")
    events = (ctx.Event(), ctx.Event())
    shm = SharedMemory(
        name=f"sparsevllm_test_status_{os.getpid()}_{uuid4().hex}",
        create=True,
        size=2**20,
    )
    sync_calls: list[int] = []
    rank0 = object.__new__(ModelRunner)
    rank0.world_size = 2
    rank0.rank = 0
    rank0.event = [events]
    rank0.shm = shm
    rank0.device = torch.device("cpu")
    rank0.platform = SimpleNamespace(synchronize=lambda: sync_calls.append(0))
    worker = object.__new__(ModelRunner)
    worker.world_size = 2
    worker.rank = 1
    worker.event = events
    worker.shm = shm
    worker.device = torch.device("cpu")
    worker.platform = SimpleNamespace(synchronize=lambda: sync_calls.append(1))

    try:
        ModelRunner._sync_tp_run_status(worker, None)
        assert shm.buf[ModelRunner._run_status_offset(worker, 1)] == TP_RUN_STATUS_SUCCESS
        ModelRunner._sync_tp_run_status(rank0, None)
        assert sync_calls == [1, 0]
    finally:
        shm.close()
        shm.unlink()


def test_run_rpc_uses_host_status_with_decode_graph():
    runner = object.__new__(ModelRunner)
    runner.world_size = 1
    runner.rank = 0
    runner.config = SimpleNamespace(decode_cuda_graph=True)
    runner.run = lambda seqs, is_prefill: (seqs, is_prefill)
    calls = []
    runner._sync_tp_run_status = lambda error: calls.append(("host", error))
    runner._sync_tp_rpc_status = lambda method, error: calls.append(
        ("collective", method, error)
    )

    result = ModelRunner.call(runner, "run", [1], False)

    assert result == ([1], False)
    assert calls == [("host", None)]


def test_run_rpc_keeps_collective_status_without_decode_graph():
    runner = object.__new__(ModelRunner)
    runner.world_size = 1
    runner.rank = 0
    runner.config = SimpleNamespace(decode_cuda_graph=False)
    runner.run = lambda seqs, is_prefill: (seqs, is_prefill)
    calls = []
    runner._sync_tp_run_status = lambda error: calls.append(("host", error))
    runner._sync_tp_rpc_status = lambda method, error: calls.append(
        ("collective", method, error)
    )

    result = ModelRunner.call(runner, "run", [1], False)

    assert result == ([1], False)
    assert calls == [("collective", "run", None)]


def test_model_runner_moe_workspace_warmup_delegates_token_count():
    calls = []
    runner = object.__new__(ModelRunner)
    runner.model = SimpleNamespace(
        warmup_moe=lambda **kwargs: calls.append(kwargs)
    )

    ModelRunner.warmup_moe_workspace(runner, 16_384)

    assert calls == [{"num_tokens": 16_384}]


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


def test_tp_worker_continues_after_multimodal_registration_failure():
    runner = object.__new__(ModelRunner)
    commands = iter(
        [
            ("register_multimodal_shared", []),
            ("free_multimodal", []),
            ("exit", []),
        ]
    )
    calls = []
    runner.read_shm = lambda: next(commands)

    def call(method_name, *_args):
        calls.append(method_name)
        if method_name == "register_multimodal_shared":
            raise ValueError("rank-local encoder failure")

    runner.call = call
    ModelRunner.loop(runner)

    assert calls == ["register_multimodal_shared", "free_multimodal", "exit"]


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


def test_model_runner_exit_drains_graphs_before_barrier():
    calls = []
    runner = object.__new__(ModelRunner)
    runner.platform = SimpleNamespace(
        synchronize=lambda: calls.append("sync"),
        barrier_device_ids=lambda rank: [rank],
    )
    runner.config = SimpleNamespace(decode_cuda_graph=True)
    runner.decode_cuda_graph_runner = SimpleNamespace(
        clear_captured_graphs=lambda: calls.append("clear_graphs")
    )
    runner.model = SimpleNamespace(
        close_runtime_operators=lambda: calls.append("close_ops")
    )
    runner.world_size = 2
    runner.rank = 0
    runner.shm = SimpleNamespace(
        close=lambda: calls.append("close_shm"),
        unlink=lambda: calls.append("unlink_shm"),
    )
    runner.parallel_context = SimpleNamespace(
        world_barrier=lambda **_: calls.append("barrier"),
    )

    with (
        patch(
            "sparsevllm.engine.model_runner.reset_parallel_context",
            side_effect=lambda: calls.append("reset"),
        ),
        patch(
            "sparsevllm.engine.model_runner.dist.destroy_process_group",
            side_effect=lambda: calls.append("destroy"),
        ),
    ):
        ModelRunner.exit(runner)

    assert calls == [
        "sync",
        "clear_graphs",
        "sync",
        "close_ops",
        "sync",
        "close_shm",
        "barrier",
        "unlink_shm",
        "reset",
        "destroy",
    ]


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
