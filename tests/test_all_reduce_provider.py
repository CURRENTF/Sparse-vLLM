import os
import sys
from types import ModuleType
from unittest.mock import ANY, patch

import pytest
import torch

from sparsevllm.operators.all_reduce import (
    ALL_REDUCE_REGISTRY,
    AllReduceGraphBufferMetadata,
    AllReduceOpSpec,
    FlashInferVllmAllReduceProvider,
    TorchDistributedAllReduceProvider,
    _expandable_segments_enabled,
)
from sparsevllm.operators.registry import OpResolver, SupportResult
from sparsevllm.platforms import DeviceCaps, PlatformEnum


def _spec(
    device_ordinals: tuple[int, ...],
    *,
    cuda_graph: bool = False,
) -> AllReduceOpSpec:
    return AllReduceOpSpec(
        world_size=2,
        ranks=(2, 3),
        max_rows=16,
        hidden_size=2048,
        dtype=torch.bfloat16,
        cuda_graph=cuda_graph,
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
        "vllm_get_graph_buffer_ipc_meta",
        "vllm_init_custom_ar",
        "vllm_meta_size",
        "vllm_register_buffer",
        "vllm_register_graph_buffers",
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


def test_vllm_all_reduce_is_selected_for_cuda_graph_replay():
    with (
        patch.dict(sys.modules, _fake_flashinfer_modules()),
        patch(
            "sparsevllm.operators.all_reduce._flashinfer_dependency_support",
            return_value=SupportResult.yes("test dependency"),
        ),
    ):
        resolved = OpResolver(ALL_REDUCE_REGISTRY).resolve(
            _spec((0, 1), cuda_graph=True),
            _caps(),
        )

    assert isinstance(resolved.provider, FlashInferVllmAllReduceProvider)


def test_vllm_all_reduce_registers_captured_graph_buffers_across_ranks():
    calls = []
    comm = _fake_flashinfer_modules()["flashinfer.comm"]
    comm.vllm_all_reduce = lambda *args: calls.append(("run", args))
    comm.vllm_get_graph_buffer_ipc_meta = lambda handle: ([10, 11], [100, 200])
    comm.vllm_register_graph_buffers = (
        lambda handle, handles, offsets: calls.append(
            ("register", handle, handles, offsets)
        )
    )
    flashinfer = ModuleType("flashinfer")
    flashinfer.comm = comm
    provider = FlashInferVllmAllReduceProvider()
    provider._handle = 7
    provider._group = "tp"
    provider._rank = 0
    provider._buffer_ptrs = [1000, 2000]
    provider._max_size_bytes = 4096

    with (
        patch.dict(
            sys.modules,
            {"flashinfer": flashinfer, "flashinfer.comm": comm},
        ),
        patch.object(torch.cuda, "is_current_stream_capturing", return_value=True),
    ):
        output = provider.run(
            _spec((0, 1), cuda_graph=True),
            torch.ones((2, 2048), dtype=torch.bfloat16),
            group="tp",
        )
        local_metadata = provider.collect_local_cuda_graph_metadata(
            _spec((0, 1), cuda_graph=True),
            group="tp",
        )
        provider.register_cuda_graph_buffers(
            _spec((0, 1), cuda_graph=True),
            [
                local_metadata,
                AllReduceGraphBufferMetadata(
                    handles=(20, 21),
                    offsets=(100, 200),
                ),
            ],
            group="tp",
        )

    assert output.shape == (2, 2048)
    assert calls == [
        ("run", (7, ANY, ANY, 0, 0, 32)),
        (
            "register",
            7,
            [[10, 11], [20, 21]],
            [[100, 200], [100, 200]],
        ),
    ]


def test_vllm_all_reduce_rejects_expandable_segments_for_cuda_graph():
    provider = FlashInferVllmAllReduceProvider()
    with patch.dict(
        os.environ,
        {"PYTORCH_ALLOC_CONF": "max_split_size_mb:64,expandable_segments:True"},
        clear=True,
    ):
        assert _expandable_segments_enabled()
        with pytest.raises(RuntimeError, match="expandable segments"):
            provider.prepare(
                _spec((0, 1), cuda_graph=True),
                group="tp",
                rank=0,
                device_index=0,
            )


def test_vllm_all_reduce_stages_only_eager_input():
    calls = []
    comm = _fake_flashinfer_modules()["flashinfer.comm"]
    comm.vllm_all_reduce = lambda *args: calls.append(args)
    flashinfer = ModuleType("flashinfer")
    flashinfer.comm = comm
    provider = FlashInferVllmAllReduceProvider()
    provider._handle = 7
    provider._rank = 0
    provider._buffer_ptrs = [1000, 2000]
    provider._max_size_bytes = 4096
    with patch.dict(sys.modules, {"flashinfer": flashinfer, "flashinfer.comm": comm}):
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            graph_output = provider.run(
                _spec((0, 1), cuda_graph=True),
                torch.ones((2, 2048), dtype=torch.bfloat16),
                group="tp",
            )
        with patch.object(torch.cuda, "is_current_stream_capturing", return_value=False):
            eager_output = provider.run(
                _spec((0, 1), cuda_graph=True),
                torch.ones((2, 2048), dtype=torch.bfloat16),
                group="tp",
            )

    assert graph_output.shape == eager_output.shape == (2, 2048)
    assert calls == [
        (7, ANY, ANY, 0, 0, 32),
        (7, ANY, ANY, 1000, 4096, 32),
    ]
