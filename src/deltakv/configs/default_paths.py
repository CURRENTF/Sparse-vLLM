from __future__ import annotations

import os
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]


def _env_path(name: str, default: Path) -> str:
    return os.getenv(name, str(default))


def model_path(name: str) -> str:
    return str(Path(_env_path("DELTAKV_MODEL_ROOT", REPO_ROOT / "models")) / name)


def compressor_path(name: str) -> str:
    return str(Path(_env_path("DELTAKV_COMPRESSOR_ROOT", REPO_ROOT / "checkpoints" / "compressor")) / name)


def dataset_path(*parts: str) -> str:
    return str(Path(_env_path("DELTAKV_DATA_DIR", REPO_ROOT / "data")).joinpath(*parts))


def output_path(*parts: str) -> str:
    return str(Path(_env_path("DELTAKV_OUTPUT_DIR", REPO_ROOT / "outputs")).joinpath(*parts))


def cache_path(*parts: str) -> str:
    return str(Path(_env_path("HF_HOME", REPO_ROOT / ".hf_cache")).joinpath(*parts))


def longbench_root() -> str:
    return os.getenv("DELTAKV_LONGBENCH_DATA_DIR", dataset_path("LongBench"))
