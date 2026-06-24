import os
import threading
import time
from multiprocessing import get_context
from multiprocessing.shared_memory import SharedMemory
from types import SimpleNamespace
from uuid import uuid4

from sparsevllm.engine.model_runner import ModelRunner


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


def test_free_slots_batch_releases_each_seq_id():
    freed: list[int] = []

    class FakeCacheManager:
        def free_seq(self, seq_id: int):
            freed.append(int(seq_id))

    runner = object.__new__(ModelRunner)
    runner.cache_manager = FakeCacheManager()

    ModelRunner.free_slots_batch(runner, [3, 5, 8])

    assert freed == [3, 5, 8]


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
