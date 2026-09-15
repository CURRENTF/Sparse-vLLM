"""Adapter to the standalone SGL FlashMLA sparse attention interface."""

from functools import lru_cache
import inspect

from sparsevllm.kernels.external.sgl.support import sgl_kernel_support
from sparsevllm.kernels.external.support import ExternalKernelContractError


@lru_cache(maxsize=1)
def sparse_mla_op():
    sgl_kernel_support("indexed shared-KV attention")
    from sgl_kernel.flash_mla import flash_mla_sparse_fwd

    expected = ("q", "kv", "indices", "sm_scale", "d_v", "attn_sink", "topk_length")
    if tuple(inspect.signature(flash_mla_sparse_fwd).parameters) != expected:
        raise ExternalKernelContractError(
            "sglang-kernel", "indexed shared-KV attention",
            f"unsupported flash_mla_sparse_fwd signature: {inspect.signature(flash_mla_sparse_fwd)}",
        )
    # The package can import successfully while the separately compiled
    # FlashMLA extension is broken. Detect that before model execution.
    import torch
    if not hasattr(torch.ops.sgl_kernel, "sparse_prefill_fwd"):
        raise ExternalKernelContractError(
            "sglang-kernel", "indexed shared-KV attention", "FlashMLA extension is unavailable",
        )
    return flash_mla_sparse_fwd
