from __future__ import annotations

from typing import Any


def collect_worker_runtime_status(llm) -> list[dict[str, Any]]:
    """Collect worker diagnostics without extending the public engine API."""
    model_runner = getattr(llm, "model_runner", None)
    call = getattr(model_runner, "call", None)
    if not callable(call):
        raise RuntimeError(
            "Sparse-VLLM runtime validation requires model_runner.call()."
        )
    statuses = call("runtime_diagnostic_status")
    expected = int(getattr(getattr(llm, "config", None), "world_size", 1))
    if (
        not isinstance(statuses, list)
        or len(statuses) != expected
        or not all(isinstance(status, dict) for status in statuses)
    ):
        raise RuntimeError(
            "Runtime validation must return one status object per model worker: "
            f"expected={expected}, got={statuses!r}."
        )
    return statuses
