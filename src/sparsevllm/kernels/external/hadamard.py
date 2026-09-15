"""Public Dao-AILab Fast Hadamard Transform adapter."""

from functools import lru_cache

from sparsevllm.kernels.external.support import ExternalKernelContractError

@lru_cache(maxsize=1)
def hadamard_op():
    try:
        from fast_hadamard_transform import hadamard_transform
    except ImportError as error:
        raise ExternalKernelContractError(
            "fast-hadamard-transform", "BF16 rotation",
            "install the deepseek-v4 extra against the runtime Torch version with build isolation disabled",
        ) from error
    return hadamard_transform
