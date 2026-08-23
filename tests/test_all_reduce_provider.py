import sys
from types import ModuleType
from unittest.mock import patch

import torch

from sparsevllm.operators.all_reduce import (
    ALL_REDUCE_REGISTRY,
    AllReduceOpSpec,
    FlashInferVllmAllReduceProvider,
    TorchDistributedAllReduceProvider,
)
from sparsevllm.operators.registry import OpResolver, SupportResult
from sparsevllm.platforms import DeviceCaps, PlatformEnum


def _spec(device_ordinals: tuple[int, ...]) -> AllReduceOpSpec:
    return AllReduceOpSpec(
        world_size=2,
        ranks=(2, 3),
        max_rows=16,
        hidden_size=2048,
        dtype=torch.bfloat16,
        cuda_graph=False,
        backend="nccl",
        device_ordinals=device_ordinals,
    )


def _caps() -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="NVIDIA H100 80GB HBM3",
        compute_capability=(9, 0),
        runtime_version="13.0",
        supports_graph_capture=True,
        supports_triton=True,
        supports_bfloat16=True,
    )


def _fake_flashinfer_modules() -> dict[str, ModuleType]:
    comm = ModuleType("flashinfer.comm")
    for name in (
        "CudaRTLibrary",
        "create_shared_buffer",
        "vllm_all_reduce",
        "vllm_dispose",
        "vllm_init_custom_ar",
        "vllm_meta_size",
        "vllm_register_buffer",
    ):
        setattr(comm, name, object())
    flashinfer = ModuleType("flashinfer")
    flashinfer.comm = comm
    return {"flashinfer": flashinfer, "flashinfer.comm": comm}


def test_vllm_all_reduce_uses_explicit_cuda_ordinals_not_global_ranks():
    with (
        patch.dict(sys.modules, _fake_flashinfer_modules()),
        patch(
            "sparsevllm.operators.all_reduce._flashinfer_dependency_support",
            return_value=SupportResult.yes("test dependency"),
        ),
    ):
        resolved = OpResolver(ALL_REDUCE_REGISTRY).resolve(_spec((0, 1)), _caps())

    assert isinstance(resolved.provider, FlashInferVllmAllReduceProvider)


def test_vllm_all_reduce_rejects_duplicate_ordinals_from_multiple_hosts():
    with (
        patch.dict(sys.modules, _fake_flashinfer_modules()),
        patch(
            "sparsevllm.operators.all_reduce._flashinfer_dependency_support",
            return_value=SupportResult.yes("test dependency"),
        ),
    ):
        resolved = OpResolver(ALL_REDUCE_REGISTRY).resolve(_spec((0, 0)), _caps())

    assert isinstance(resolved.provider, TorchDistributedAllReduceProvider)
