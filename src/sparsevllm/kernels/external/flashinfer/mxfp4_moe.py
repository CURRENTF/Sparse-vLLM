"""Public FlashInfer Hopper MXFP4 MoE APIs and checkpoint layout preparation."""

from functools import lru_cache
import importlib
import importlib.metadata
import inspect

from packaging.version import Version

from sparsevllm.kernels.external.flashinfer.support import flashinfer_kernel_support
from sparsevllm.kernels.external.support import ExternalKernelContractError


@lru_cache(maxsize=1)
def mxfp4_moe_ops():
    feature = "Hopper MXFP4 W4A16 MoE"
    flashinfer_kernel_support(feature)
    if Version(importlib.metadata.version("flashinfer-python")) < Version("0.6.18.post1"):
        raise ExternalKernelContractError("flashinfer-python", feature, "requires >=0.6.18.post1")
    schemas = {
        "cutlass_fused_moe": ("swiglu_alpha", "swiglu_beta", "swiglu_limit",
                              "use_w4_group_scaling", "use_fused_finalize", "workspace_buffer"),
        "cutlass_fused_moe_workspace_size": ("max_num_tokens", "ep_size", "ep_rank",
                                             "use_w4_group_scaling", "use_fused_finalize"),
        "interleave_moe_weights_for_sm90_mixed_gemm": ("weight",),
        "interleave_moe_scales_for_sm90_mixed_gemm": ("scales",),
    }
    try:
        module = importlib.import_module("flashinfer.fused_moe")
        functions = tuple(getattr(module, name) for name in schemas)
        for (name, required), function in zip(schemas.items(), functions):
            if not set(required) <= inspect.signature(function).parameters.keys():
                raise TypeError(f"unsupported public schema for {name}")
    except (ImportError, AttributeError, TypeError, ValueError) as error:
        raise ExternalKernelContractError("flashinfer-python", feature, str(error)) from error
    return functions
