import os
import threading
import time
from unittest.mock import patch

from sparsevllm.engine.llm_engine import LLMEngine


def test_engine_exit_timeout_still_terminates_workers():
    class BlockingRunner:
        def __init__(self):
            self.call_started = threading.Event()

        def call(self, method_name):
            assert method_name == "exit"
            self.call_started.set()
            threading.Event().wait()

    class Worker:
        pid = 12345

        def __init__(self):
            self.alive = True
            self.terminated = False
            self.killed = False

        def is_alive(self):
            return self.alive

        def terminate(self):
            self.terminated = True
            self.alive = False

        def kill(self):
            self.killed = True
            self.alive = False

        def join(self, timeout=None):
            del timeout

    runner = BlockingRunner()
    worker = Worker()
    engine = object.__new__(LLMEngine)
    engine._exited = False
    engine.model_runner = runner
    engine.ps = [worker]

    with patch.dict(
        os.environ,
        {
            "SPARSEVLLM_ENGINE_EXIT_TIMEOUT_S": "0.05",
            "SPARSEVLLM_WORKER_JOIN_TIMEOUT_S": "0.05",
        },
    ):
        started = time.perf_counter()
        engine.exit()
        elapsed = time.perf_counter() - started

    assert elapsed < 1.0
    assert runner.call_started.is_set()
    assert worker.terminated
    assert not worker.killed
