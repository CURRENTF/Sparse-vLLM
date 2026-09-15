"""Original vLLM DSv4 cache kernels without serving-engine initialization."""

from functools import lru_cache
import importlib.util
import inspect
from pathlib import Path
import sys

from sparsevllm.kernels.external.support import ExternalKernelContractError
from sparsevllm.kernels.external.vllm_support import _load_op, vllm_library


def _source_module(name, path):
    if name in sys.modules:
        module = sys.modules[name]
        if Path(getattr(module, "__file__", "")).resolve() != path.resolve():
            raise ValueError(f"{name} is already loaded from a different installation")
        return module
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        del sys.modules[name]
        raise
    return module


@lru_cache(maxsize=1)
def shared_kv_cache_ops():
    feature = "DSv4 packed shared-KV cache"
    writer = _load_op(
        feature, "_C_stable_libtorch.abi3.so", "_C",
        "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
        ("q_in", "kv", "k_cache", "slot_mapping", "position_ids", "cos_sin_cache",
         "q_head_padded", "eps", "cache_block_size"),
    )
    if writer is None:
        return None
    root = vllm_library(feature, "_C_stable_libtorch.abi3.so").parent.parent
    try:
        for name in ("vllm", "quack"):
            existing = sys.modules.get(name)
            if existing is not None and Path(existing.__file__).resolve().parent != root / name:
                raise ValueError(f"{name} is already loaded from a different installation")
        # CuTe reparses imports during compilation, so these real dependency
        # modules need their upstream names. Loading leaves avoids the root
        # packages' global engine/compiler initialization and fake namespaces.
        helper = _source_module("vllm.cute_utils", root / "vllm/cute_utils/__init__.py")
        helper.cvt = _source_module("vllm.cute_utils.cvt", root / "vllm/cute_utils/cvt.py")
        _source_module("quack.compile_utils", root / "quack/compile_utils.py")
        module = _source_module(
            "sparsevllm.kernels.external._vllm_cache_gather",
            root / "vllm/models/deepseek_v4/nvidia/ops/dequant_gather_k_cutedsl.py",
        )
        gather = module.dequantize_and_gather_k_cache_cutedsl
        if tuple(inspect.signature(gather).parameters) != (
            "out", "k_cache", "seq_lens", "gather_lens", "block_table", "block_size", "offset",
        ):
            raise ValueError("unsupported DSv4 gather interface")
    except (OSError, ImportError, AttributeError, TypeError, ValueError) as error:
        raise ExternalKernelContractError("vllm", feature, str(error)) from error
    return writer, gather
