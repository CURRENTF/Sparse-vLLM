"""Optional original vLLM wheel discovery and stable-ABI contract checks."""

import importlib.metadata
import importlib.util
import os
from pathlib import Path

from packaging.version import Version
import torch

from sparsevllm.kernels.external.support import ExternalKernelContractError


def vllm_library(feature, library_name):
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
        return library
    except (OSError, RuntimeError, AttributeError, ValueError, StopIteration) as error:
        raise ExternalKernelContractError("vllm", feature, str(error) or "wheel metadata is missing") from error


def _load_op(feature, library_name, namespace, name, expected):
    library = vllm_library(feature, library_name)
    if library is None:
        return None
    try:
        torch.ops.load_library(str(library))
        op = getattr(getattr(torch.ops, namespace), name).default
        if tuple(arg.name for arg in op._schema.arguments) != expected:
            raise ValueError(f"unsupported {name} operator schema")
    except (OSError, RuntimeError, AttributeError, ValueError) as error:
        raise ExternalKernelContractError("vllm", feature, str(error)) from error
    return op
