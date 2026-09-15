"""Optional stable-ABI vLLM MoE kernels, without importing the vLLM engine."""

from functools import lru_cache
import importlib.metadata
import importlib.util
import os
from pathlib import Path

from packaging.version import Version
import torch

from sparsevllm.kernels.external.support import ExternalKernelContractError


def _load_op(feature, library_name, namespace, name, expected):
    location = os.environ.get("SPARSEVLLM_VLLM_MOE_LIBRARY")
    if location:
        library = Path(location).expanduser().resolve()
    else:
        package = importlib.util.find_spec("vllm")
        if package is None or package.origin is None:
            return None
        library = Path(package.origin).parent / "_moe_C_stable_libtorch.abi3.so"
    try:
        if not library.is_file():
            raise OSError("configured vLLM MoE library does not exist")
        if library_name != "_moe_C_stable_libtorch.abi3.so":
            library = library.parent / library_name
        if not library.is_file():
            raise OSError(f"required vLLM library {library_name} does not exist")
        distributions = importlib.metadata.distributions(path=[str(library.parent.parent)])
        version = next(d.version for d in distributions if d.metadata["Name"] == "vllm")
        if not Version("0.29.0") <= Version(version) < Version("0.30"):
            raise ValueError(f"requires vLLM >=0.29,<0.30, found {version}")
        torch.ops.load_library(str(library))
        op = getattr(getattr(torch.ops, namespace), name).default
        if tuple(arg.name for arg in op._schema.arguments) != expected:
            raise ValueError(f"unsupported {name} operator schema")
    except (OSError, RuntimeError, AttributeError, ValueError, StopIteration) as error:
        raise ExternalKernelContractError("vllm", feature, str(error) or "wheel metadata is missing") from error
    return op


@lru_cache(maxsize=1)
def sqrt_softplus_op():
    return _load_op(
        "stable-ABI sqrt-softplus router", "_moe_C_stable_libtorch.abi3.so",
        "_moe_C", "topk_softplus_sqrt",
        ("topk_weights", "topk_indices", "token_expert_indices", "gating_output",
         "renormalize", "routed_scaling_factor", "bias", "input_ids", "tid2eid", "is_padding"),
    )


@lru_cache(maxsize=1)
def clipped_swiglu_op():
    return _load_op(
        "stable-ABI clipped SwiGLU", "_C_stable_libtorch.abi3.so",
        "_C", "silu_and_mul_with_clamp", ("result", "input", "limit", "alpha", "beta"),
    )


@lru_cache(maxsize=1)
def hash_inputs_op():
    def prepare(input_ids, vocab_size, index_dtype):
        return input_ids.to(index_dtype), (input_ids < 0) | (input_ids >= vocab_size)

    return torch.compile(prepare, fullgraph=True, dynamic=True)
