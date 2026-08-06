from __future__ import annotations

import json
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


class DockerWritableLayerLimitExceeded(RuntimeError):
    """Raised when a sample container exceeds its writable-layer limit."""


class DockerWritableLayerGuardError(RuntimeError):
    """Raised when the writable-layer guard can no longer monitor safely."""


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _instance_id_from_image(image: str) -> str | None:
    image_name = image.rsplit("/", 1)[-1].split(":", 1)[0]
    image_name = image_name.removeprefix("sweb.eval.x86_64.")
    if "_1776_" not in image_name:
        return None
    return image_name.replace("_1776_", "__", 1)


@dataclass
class GuardState:
    container_id: str
    image: str
    failure: BaseException | None = None
    writable_bytes: int | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def fail(self, error: BaseException, *, writable_bytes: int | None = None) -> bool:
        with self._lock:
            if self.failure is not None:
                return False
            self.failure = error
            self.writable_bytes = writable_bytes
            return True

    def raise_if_failed(self) -> None:
        with self._lock:
            failure = self.failure
        if failure is not None:
            raise failure


class DockerWritableLayerGuard:
    """Poll Docker writable-layer sizes and fail only the responsible samples."""

    def __init__(
        self,
        *,
        executable: str,
        limit_bytes: int,
        poll_seconds: float,
        events_path: Path,
        run_id: str,
        max_monitor_failures: int = 3,
        autostart: bool = True,
    ):
        if limit_bytes <= 0:
            raise ValueError("limit_bytes must be positive")
        if poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        if max_monitor_failures <= 0:
            raise ValueError("max_monitor_failures must be positive")
        self.executable = executable
        self.limit_bytes = limit_bytes
        self.poll_seconds = poll_seconds
        self.events_path = events_path
        self.run_id = run_id
        self.max_monitor_failures = max_monitor_failures
        self._states: dict[str, GuardState] = {}
        self._lock = threading.Lock()
        self._wake = threading.Event()
        self._monitor_failures = 0
        self._thread: threading.Thread | None = None
        if autostart:
            self._thread = threading.Thread(
                target=self._run,
                name="docker-writable-layer-guard",
                daemon=True,
            )
            self._thread.start()

    def register(self, container_id: str, *, image: str) -> GuardState:
        if not container_id:
            raise ValueError("container_id must not be empty")
        state = GuardState(container_id=container_id, image=image)
        with self._lock:
            if container_id in self._states:
                raise ValueError(f"Container is already registered: {container_id}")
            self._states[container_id] = state
        self._wake.set()
        return state

    def unregister(self, container_id: str) -> None:
        with self._lock:
            self._states.pop(container_id, None)

    def _retire(self, state: GuardState) -> None:
        with self._lock:
            if self._states.get(state.container_id) is state:
                self._states.pop(state.container_id)

    def _snapshot(self) -> dict[str, GuardState]:
        with self._lock:
            return dict(self._states)

    def _append_event(self, event: dict[str, Any]) -> None:
        self.events_path.parent.mkdir(parents=True, exist_ok=True)
        with self.events_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")

    def _kill(self, container_id: str) -> None:
        subprocess.run(
            [self.executable, "kill", container_id],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )

    def _fail_state(
        self,
        state: GuardState,
        error: BaseException,
        *,
        event_type: str,
        writable_bytes: int | None = None,
        detail: str | None = None,
    ) -> None:
        newly_failed = state.fail(error, writable_bytes=writable_bytes)
        self._retire(state)
        if not newly_failed:
            return
        event = {
            "time": _now(),
            "event": event_type,
            "run_id": self.run_id,
            "container_id": state.container_id,
            "image": state.image,
            "instance_id": _instance_id_from_image(state.image),
            "writable_bytes": writable_bytes,
            "limit_bytes": self.limit_bytes,
        }
        if detail:
            event["detail"] = detail
        try:
            self._append_event(event)
        finally:
            self._kill(state.container_id)

    def _fail_closed(self, states: dict[str, GuardState], detail: str) -> None:
        for state in states.values():
            self._fail_state(
                state,
                DockerWritableLayerGuardError(
                    "Docker writable-layer monitoring failed repeatedly; "
                    f"container {state.container_id} was terminated: {detail}"
                ),
                event_type="monitor_failed",
                detail=detail,
            )

    def _poll_once(self) -> None:
        states = self._snapshot()
        if not states:
            self._monitor_failures = 0
            return
        command = [
            self.executable,
            "inspect",
            "--size",
            "--format",
            "{{json .}}",
            *states,
        ]
        try:
            result = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=max(30.0, self.poll_seconds * 2),
            )
            records = []
            for line in result.stdout.splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise RuntimeError(
                        f"docker inspect returned a non-object record: {type(record).__name__}"
                    )
                records.append(record)
            if result.returncode != 0 or len(records) != len(states):
                returned_ids = {
                    str(record.get("Id") or "") for record in records
                }
                missing_ids = set(states) - returned_ids
                current_ids = set(self._snapshot())
                stale_ids = missing_ids - current_ids
                error_prefix = "error: no such object: "
                error_lines = [
                    line.strip()
                    for line in result.stderr.splitlines()
                    if line.strip()
                ]
                missing_object_ids = {
                    line[len(error_prefix) :].strip()
                    for line in error_lines
                    if line.lower().startswith(error_prefix)
                }
                cleanup_race = (
                    result.returncode != 0
                    and bool(missing_ids)
                    and stale_ids == missing_ids
                    and len(missing_object_ids) == len(error_lines)
                    and missing_object_ids == missing_ids
                )
                if not cleanup_race:
                    detail = result.stderr.strip() or (
                        f"docker inspect returned {len(records)} of {len(states)} "
                        "containers"
                    )
                    raise RuntimeError(detail)
        except (OSError, subprocess.SubprocessError, json.JSONDecodeError, RuntimeError) as exc:
            self._monitor_failures += 1
            if self._monitor_failures >= self.max_monitor_failures:
                self._fail_closed(self._snapshot(), str(exc))
            return

        self._monitor_failures = 0
        current_states = self._snapshot()
        for record in records:
            container_id = str(record.get("Id") or "")
            state = current_states.get(container_id)
            if state is None:
                continue
            writable_bytes = record.get("SizeRw")
            if not isinstance(writable_bytes, int) or writable_bytes < 0:
                self._fail_state(
                    state,
                    DockerWritableLayerGuardError(
                        f"Docker returned invalid SizeRw for container {container_id}: "
                        f"{writable_bytes!r}"
                    ),
                    event_type="invalid_size",
                    detail=f"SizeRw={writable_bytes!r}",
                )
                continue
            if writable_bytes > self.limit_bytes:
                self._fail_state(
                    state,
                    DockerWritableLayerLimitExceeded(
                        f"Container {container_id} writable layer is {writable_bytes} bytes, "
                        f"exceeding the {self.limit_bytes}-byte sample limit"
                    ),
                    event_type="limit_exceeded",
                    writable_bytes=writable_bytes,
                )

    def _run(self) -> None:
        while True:
            if not self._snapshot():
                self._wake.wait()
                self._wake.clear()
            try:
                self._poll_once()
            except Exception as exc:
                self._fail_closed(self._snapshot(), f"guard thread failed: {exc}")
            time.sleep(self.poll_seconds)
