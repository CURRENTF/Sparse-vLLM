from __future__ import annotations

import os
import threading
from pathlib import Path

from minisweagent.environments.docker import DockerEnvironment

from benchmark.swe_bench_lite.docker_writable_guard import (
    DockerWritableLayerGuard,
    GuardState,
)


_GUARD_LOCK = threading.Lock()
_GUARD: DockerWritableLayerGuard | None = None


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Required Docker writable-layer guard setting is empty: {name}")
    return value


def _get_guard(executable: str) -> DockerWritableLayerGuard:
    global _GUARD
    with _GUARD_LOCK:
        if _GUARD is None:
            _GUARD = DockerWritableLayerGuard(
                executable=executable,
                limit_bytes=int(
                    _required_env("SPARSEVLLM_DOCKER_WRITABLE_LAYER_LIMIT_BYTES")
                ),
                poll_seconds=float(
                    _required_env("SPARSEVLLM_DOCKER_WRITABLE_LAYER_POLL_SECONDS")
                ),
                events_path=Path(
                    _required_env("SPARSEVLLM_DOCKER_WRITABLE_LAYER_EVENTS")
                ),
                run_id=_required_env("SPARSEVLLM_SWE_RUN_ID"),
            )
        return _GUARD


class GuardedDockerEnvironment(DockerEnvironment):
    """mini-SWE-agent Docker environment with a per-container disk limit."""

    def __init__(self, **kwargs):
        self._writable_guard: DockerWritableLayerGuard | None = None
        self._writable_guard_state: GuardState | None = None
        super().__init__(**kwargs)

    def _start_container(self) -> None:
        super()._start_container()
        assert self.container_id is not None
        try:
            self._writable_guard = _get_guard(self.config.executable)
            self._writable_guard_state = self._writable_guard.register(
                self.container_id,
                image=self.config.image,
            )
        except Exception:
            super().cleanup()
            raise

    def _raise_if_guard_failed(self) -> None:
        if self._writable_guard_state is not None:
            self._writable_guard_state.raise_if_failed()

    def execute(self, action: dict, cwd: str = "", *, timeout: int | None = None):
        self._raise_if_guard_failed()
        try:
            return super().execute(action, cwd=cwd, timeout=timeout)
        finally:
            self._raise_if_guard_failed()

    def cleanup(self) -> None:
        container_id = getattr(self, "container_id", None)
        guard = getattr(self, "_writable_guard", None)
        if container_id is not None and guard is not None:
            guard.unregister(container_id)
        super().cleanup()
