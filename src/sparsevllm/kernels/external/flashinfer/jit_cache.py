"""Process-local cache ownership for FlashInfer's TensorRT-LLM DeepGEMM JIT."""

from __future__ import annotations

import os
from pathlib import Path
import uuid


# The upstream C++ compiler remembers its first cache directory for the entire
# process. Keep that directory across sequential engines, including rank zero.
_configured_cache: tuple[int, Path, Path] | None = None


def _check_cache_root(root: Path) -> Path:
    configured = _configured_cache
    if configured is not None and configured[0] == os.getpid() and root != configured[1]:
        raise RuntimeError(
            "TensorRT-LLM DeepGEMM cache root cannot change in an initialized "
            f"worker process: {configured[1]} -> {root}. Use a fresh process."
        )
    return root


def resolve_trtllm_cache_root() -> Path:
    """Resolve the user root before spawning workers or rewriting their env."""
    for name in ("SPARSEVLLM_TRTLLM_DG_CACHE_ROOT", "TRTLLM_DG_CACHE_DIR"):
        value = os.environ.get(name)
        if value is None:
            continue
        if not value.strip():
            raise ValueError(f"{name} must name a non-empty cache directory.")
        root = Path(value).expanduser().resolve()
        configured = _configured_cache
        if (
            name == "TRTLLM_DG_CACHE_DIR"
            and configured is not None
            and configured[0] == os.getpid()
            and root == configured[2]
        ):
            return configured[1]
        return _check_cache_root(root)
    cache_home = os.environ.get("XDG_CACHE_HOME")
    base = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return _check_cache_root((base / "sparsevllm" / "trtllm-deepgemm").resolve())


def configure_trtllm_cache(rank: int, root: str | Path | None = None) -> Path:
    """Isolate cold JIT writes before any external CUDA kernel initialization."""
    global _configured_cache
    root = resolve_trtllm_cache_root() if root is None else Path(root).expanduser().resolve()
    _check_cache_root(root)
    configured = _configured_cache
    if configured is not None and configured[0] == os.getpid():
        path = configured[2]
    else:
        # UUID also separates independent engines with the same rank, shared
        # filesystems across hosts, and processes whose OS PIDs are reused.
        path = root / f"worker-{uuid.uuid4().hex}-rank-{rank}"
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise RuntimeError(
            f"Cannot prepare TensorRT-LLM DeepGEMM cache directory {path}: {error}"
        ) from error
    os.environ["TRTLLM_DG_CACHE_DIR"] = str(path)
    _configured_cache = (os.getpid(), root, path)
    return path
