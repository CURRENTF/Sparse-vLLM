# SPDX-License-Identifier: Apache-2.0
"""Loader for Sparse-vLLM C++ metadata acceleration extensions."""
import os
from pathlib import Path
from typing import Any
import torch
from torch.utils.cpp_extension import load

_CACHE_DIR = Path(os.getenv("TMPDIR", "/data2/haojitai/tmp")) / "sparsevllm_cpp_build"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)

_EXT_MODULE: Any = None


def get_cache_metadata_ext() -> Any:
    global _EXT_MODULE
    if _EXT_MODULE is None:
        src_path = Path(__file__).resolve().parent / "cache_metadata.cpp"
        _EXT_MODULE = load(
            name="sparsevllm_c_metadata",
            sources=[str(src_path)],
            extra_cflags=["-O3", "-std=c++17"],
            build_directory=str(_CACHE_DIR),
            verbose=False,
        )
    return _EXT_MODULE
