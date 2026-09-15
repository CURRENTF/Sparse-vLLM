"""Public FlashInfer mHC entry points; no private JIT calls or patched kernels."""

from functools import lru_cache
import importlib
import inspect

from sparsevllm.kernels.external.flashinfer.support import flashinfer_kernel_support
from sparsevllm.kernels.external.support import ExternalKernelContractError


@lru_cache(maxsize=1)
def mhc_ops():
    feature = "four-stream mHC pre/post"
    flashinfer_kernel_support(feature)
    schemas = {
        "mhc_pre_big_fuse_with_prenorm": (
            "dot_mix", "residual", "mhc_scale", "mhc_base", "rms_eps", "mhc_pre_eps",
            "mhc_sinkhorn_eps", "mhc_post_mult_value", "sinkhorn_repeat", "block_size",
        ),
        "mhc_post": ("x", "residual", "post_layer_mix", "comb_res_mix"),
    }
    try:
        module = importlib.import_module("flashinfer.mhc")
        functions = tuple(getattr(module, name) for name in schemas)
        for name, function in zip(schemas, functions):
            if tuple(inspect.signature(function).parameters) != schemas[name]:
                raise TypeError(f"unsupported public schema for {name}")
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        raise ExternalKernelContractError("flashinfer-python", feature, str(error)) from error
    return functions
