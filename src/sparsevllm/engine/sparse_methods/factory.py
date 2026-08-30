from __future__ import annotations

from sparsevllm.config import Config
from sparsevllm.engine.cache_manager import CacheManager
from sparsevllm.method_registry import normalize_sparse_method

from .base import SparseMethodRuntime
from .dynamic import DeltaKVRuntime, OmniKVRuntime
from .h2o import H2ORuntime
from .joint import RKVRuntime, SkipKVRuntime
from .passthrough import PassThroughRuntime
from .snapkv import PyramidKVRuntime, SnapKVRuntime
from .streamingllm import StreamingLLMRuntime


RUNTIME_BINDINGS: dict[str, type[SparseMethodRuntime]] = {
    "": PassThroughRuntime,
    "streamingllm": StreamingLLMRuntime,
    "snapkv": SnapKVRuntime,
    "h2o": H2ORuntime,
    "pyramidkv": PyramidKVRuntime,
    "omnikv": OmniKVRuntime,
    "quest": PassThroughRuntime,
    "rkv": RKVRuntime,
    "skipkv": SkipKVRuntime,
    "deltakv": DeltaKVRuntime,
}


def create_sparse_method_runtime(
    config: Config,
    cache_manager: CacheManager,
) -> SparseMethodRuntime:
    method = normalize_sparse_method(config.sparse_method)
    runtime_cls = RUNTIME_BINDINGS.get(method)
    if runtime_cls is None:
        raise ValueError(f"Unsupported sparse_method={method!r}.")
    return runtime_cls(config, cache_manager)
