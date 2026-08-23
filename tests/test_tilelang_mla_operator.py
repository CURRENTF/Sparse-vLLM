from __future__ import annotations

import subprocess
import sys
from importlib import metadata
from unittest.mock import Mock, patch

import pytest
import torch

from sparsevllm.engine.cache_manager import (
    AttentionViewMeta,
    DecodeComputeView,
    MlaLatentPayload,
)
from sparsevllm.kernels.tilelang.mla.runtime import (
    TileMlaDecodeKernel,
    tilelang_mla_support,
)
from sparsevllm.kernels.triton.mla import MlaDecodeWorkspace
from sparsevllm.operators.mla_attention import (
    MLA_ATTENTION_REGISTRY,
    MlaAttentionOpSpec,
    MlaSglFa3Provider,
    MlaTileLangScoreProvider,
    MlaTritonProvider,
)
from sparsevllm.operators.registry import OpResolver
from sparsevllm.platforms import DeviceCaps, PlatformEnum


def _spec(*, tp_size: int = 2) -> MlaAttentionOpSpec:
    return MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=tp_size,
        cuda_graph=True,
        may_require_attention_scores=True,
    )


def _h100_caps() -> DeviceCaps:
    return DeviceCaps(
        platform=PlatformEnum.CUDA,
        device_type="cuda",
        device_index=0,
        device_name="NVIDIA H100 80GB HBM3",
        compute_capability=(9, 0),
        runtime_version="13.0",
        supports_graph_capture=True,
        supports_torch_compile=True,
        supports_triton=True,
        supports_pin_memory=True,
        supports_bfloat16=True,
        supports_native_fp8=True,
    )


def _cpu_workspace() -> MlaDecodeWorkspace:
    return MlaDecodeWorkspace(
        block_size=torch.empty(1, dtype=torch.int32),
        batch_start_indices=torch.empty(2, dtype=torch.int32),
        mid_output=torch.empty(20, 1, 512, dtype=torch.float32),
        mid_logsumexp=torch.empty(20, 1, dtype=torch.float32),
    )


def _view(*, score: torch.Tensor | None) -> DecodeComputeView:
    active_slots = torch.full((3, 64), -1, dtype=torch.int32)
    active_slots[2, :3] = torch.tensor([5, 2, 7], dtype=torch.int32)
    return DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=active_slots,
            req_indices=torch.tensor([2, -1], dtype=torch.int32),
            context_lens=torch.tensor([3, 0], dtype=torch.int32),
            max_context_len=64,
            attn_score=score,
        ),
        payload=MlaLatentPayload(
            latent_cache=torch.empty(8, 1, 512, dtype=torch.bfloat16),
            rope_cache=torch.empty(8, 1, 64, dtype=torch.bfloat16),
        ),
    )


def test_tilelang_support_does_not_import_runtime() -> None:
    def version(package: str) -> str:
        return {"tilelang": "0.1.9", "apache-tvm-ffi": "0.1.10"}[package]

    with patch.object(metadata, "version", side_effect=version):
        assert tilelang_mla_support() == (
            True,
            "tilelang 0.1.9, apache-tvm-ffi 0.1.10",
        )


def test_tilelang_runtime_import_does_not_require_tilelang() -> None:
    code = """
import sys

class RejectTileLang:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "tilelang" or fullname.startswith("tilelang."):
            raise ModuleNotFoundError("blocked optional tilelang import")

sys.meta_path.insert(0, RejectTileLang())
from sparsevllm.kernels.tilelang.mla.runtime import tilelang_mla_support
assert "tilelang" not in sys.modules
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("version", "supported"),
    [("0.1.8", False), ("0.1.9", True), ("0.2.0+cu130", False)],
)
def test_tilelang_support_version_boundary(version: str, supported: bool) -> None:
    def installed(package: str) -> str:
        return version if package == "tilelang" else "0.1.10"

    with patch.object(metadata, "version", side_effect=installed):
        assert tilelang_mla_support()[0] is supported


def test_tilelang_support_reports_missing_package() -> None:
    with patch.object(
        metadata,
        "version",
        side_effect=metadata.PackageNotFoundError,
    ):
        assert tilelang_mla_support() == (False, "tilelang is not installed")


def test_tilelang_support_rejects_unvalidated_tvm_ffi() -> None:
    def version(package: str) -> str:
        return {
            "tilelang": "0.1.9",
            "apache-tvm-ffi": "0.1.13.post2",
        }[package]

    with patch.object(metadata, "version", side_effect=version):
        supported, reason = tilelang_mla_support()
    assert not supported
    assert "apache-tvm-ffi==0.1.10" in reason


@pytest.mark.parametrize(("tp_size", "local_heads"), [(1, 20), (2, 10), (4, 5)])
def test_tilelang_provider_binds_rank_local_head_count(
    tp_size: int,
    local_heads: int,
) -> None:
    workspace = _cpu_workspace()
    with (
        patch(
            "sparsevllm.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
        patch("sparsevllm.operators.mla_attention.SglFa3DecodeKernel"),
        patch(
            "sparsevllm.operators.mla_attention.TileMlaDecodeKernel"
        ) as tilelang_cls,
    ):
        MlaTileLangScoreProvider(
            op_spec=_spec(tp_size=tp_size),
            device="cpu",
            max_batch_size=2,
        )

    tilelang_cls.assert_called_once_with(
        device=torch.device("cpu"),
        softmax_scale=256**-0.5,
        valid_heads=local_heads,
    )


def test_missing_tilelang_binds_score_capable_triton_provider() -> None:
    workspace = _cpu_workspace()
    with (
        patch(
            "sparsevllm.operators.mla_attention.sgl_fa3_device_support",
            return_value=(True, "sgl test"),
        ),
        patch(
            "sparsevllm.operators.mla_attention.tilelang_mla_support",
            return_value=(False, "tilelang missing"),
        ),
        patch(
            "sparsevllm.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
        patch("sparsevllm.operators.mla_attention.SglFa3DecodeKernel"),
    ):
        resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            _spec(),
            _h100_caps(),
            op_spec=_spec(),
            device="cpu",
            max_batch_size=2,
        )
    assert type(resolved.provider) is MlaTritonProvider
    assert (
        "sgl_fa3_sm90",
        "does not satisfy the prepared score-output contract",
    ) in resolved.rejected
    assert (
        "tilelang_score_sgl_fa3_h100",
        "tilelang missing",
    ) in resolved.rejected


def _provider_with_mocks() -> tuple[MlaTileLangScoreProvider, Mock, Mock]:
    fa3 = Mock(return_value=torch.empty(2, 10, 512, dtype=torch.bfloat16))
    tilelang = Mock(return_value=torch.empty(2, 10, 512, dtype=torch.bfloat16))
    with (
        patch(
            "sparsevllm.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=_cpu_workspace(),
        ),
        patch(
            "sparsevllm.operators.mla_attention.SglFa3DecodeKernel",
            return_value=fa3,
        ),
        patch(
            "sparsevllm.operators.mla_attention.TileMlaDecodeKernel",
            return_value=tilelang,
        ),
    ):
        provider = MlaTileLangScoreProvider(
            op_spec=_spec(),
            device="cpu",
            max_batch_size=2,
        )
    return provider, fa3, tilelang


def test_score_path_routes_to_tilelang_with_caller_owned_score() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    score = torch.full((2, 64), -1e20, dtype=torch.float32)
    view = _view(score=score)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with patch(
        "sparsevllm.operators.mla_attention.validate_mla_decode_metadata"
    ):
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_called_once_with(
        q_latent,
        q_rope,
        view.payload.latent_cache,
        view.payload.rope_cache,
        view.meta.active_slots,
        view.meta.req_indices,
        view.meta.context_lens,
        output,
        attn_score=score,
        max_context_len=64,
    )
    assert provider.runtime_kernel_stats() == {
        "kernel_paths": {
            "tilelang_score": {
                "cuda_graph_capture_dispatches": 0,
                "eager_dispatches": 1,
            }
        },
        "fallback_reasons": {},
    }


def test_noncontiguous_glm_queries_route_to_tilelang() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    score = torch.full((2, 64), -1e20, dtype=torch.float32)
    view = _view(score=score)
    q_latent = torch.empty(10, 2, 512, dtype=torch.bfloat16).transpose(0, 1)
    q_rope = torch.empty(10, 2, 64, dtype=torch.bfloat16).transpose(0, 1)
    output = torch.empty(q_latent.shape, dtype=q_latent.dtype)

    assert not q_latent.is_contiguous()
    assert not q_rope.is_contiguous()
    with patch(
        "sparsevllm.operators.mla_attention.validate_mla_decode_metadata"
    ):
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_called_once()


def test_runtime_kernel_stats_distinguish_cuda_graph_capture() -> None:
    provider, _, tilelang = _provider_with_mocks()
    view = _view(score=torch.empty(2, 64, dtype=torch.float32))
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with (
        patch(
            "sparsevllm.operators.mla_attention.validate_mla_decode_metadata"
        ),
        patch(
            "sparsevllm.operators.mla_attention.device_runtime.is_stream_capturing",
            return_value=True,
        ),
    ):
        provider.run(q_latent, q_rope, view, output)

    tilelang.assert_called_once()
    assert provider.runtime_kernel_stats()["kernel_paths"]["tilelang_score"] == {
        "cuda_graph_capture_dispatches": 1,
        "eager_dispatches": 0,
    }


def test_no_score_path_remains_fa3() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    view = _view(score=None)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with patch(
        "sparsevllm.operators.mla_attention.validate_mla_decode_metadata"
    ):
        provider.run(q_latent, q_rope, view, output)

    tilelang.assert_not_called()
    fa3.assert_called_once()
    assert fa3.call_args.kwargs["num_splits"] == 0


@pytest.mark.parametrize(
    "score",
    [
        torch.empty(2, 10, 64, dtype=torch.float32),
        torch.empty(2, 64, dtype=torch.bfloat16),
        torch.empty(2, 63, dtype=torch.float32),
    ],
)
def test_unsupported_score_contract_uses_explicit_triton_path(score) -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    view = _view(score=score)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with patch.object(
        MlaSglFa3Provider.__mro__[1], "run", return_value=output
    ) as triton:
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_not_called()
    triton.assert_called_once()


def test_score_capacity_smaller_than_declared_context_uses_triton() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    view = _view(score=torch.empty(2, 64, dtype=torch.float32))
    object.__setattr__(view.meta, "max_context_len", 128)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with patch.object(
        MlaSglFa3Provider.__mro__[1], "run", return_value=output
    ) as triton:
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_not_called()
    triton.assert_called_once()


def test_score_capacity_larger_than_active_slots_uses_triton() -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    view = _view(score=torch.empty(2, 64, dtype=torch.float32))
    object.__setattr__(
        view.meta,
        "active_slots",
        torch.zeros((2, 4), dtype=torch.int32),
    )
    object.__setattr__(view.meta, "max_context_len", 4)
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with patch.object(
        MlaSglFa3Provider.__mro__[1], "run", return_value=output
    ) as triton:
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_not_called()
    triton.assert_called_once()


@pytest.mark.parametrize("noncontiguous", ["active_slots", "attn_score"])
def test_noncontiguous_tilelang_inputs_use_triton(noncontiguous: str) -> None:
    provider, fa3, tilelang = _provider_with_mocks()
    score = torch.empty(2, 128, dtype=torch.float32)[:, ::2]
    view = _view(
        score=(
            score
            if noncontiguous == "attn_score"
            else torch.empty(2, 64, dtype=torch.float32)
        )
    )
    if noncontiguous == "active_slots":
        backing = torch.full((3, 128), -1, dtype=torch.int32)
        backing[2, :6:2] = torch.tensor([5, 2, 7], dtype=torch.int32)
        object.__setattr__(view.meta, "active_slots", backing[:, ::2])
    q_latent = torch.empty(2, 10, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 10, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_latent)

    with patch.object(
        MlaSglFa3Provider.__mro__[1], "run", return_value=output
    ) as triton:
        provider.run(q_latent, q_rope, view, output)

    fa3.assert_not_called()
    tilelang.assert_not_called()
    triton.assert_called_once()
    assert provider.runtime_kernel_stats()["fallback_reasons"] == {
        f"noncontiguous:{noncontiguous}": 1
    }


def test_tilelang_runner_rejects_unaligned_score_capacity_before_import() -> None:
    runner = TileMlaDecodeKernel(device="cpu", softmax_scale=0.0625)
    view = _view(score=torch.empty(2, 63, dtype=torch.float32))
    with pytest.raises(ValueError, match="multiple of 64"):
        runner(
            torch.empty(2, 10, 512, dtype=torch.bfloat16),
            torch.empty(2, 10, 64, dtype=torch.bfloat16),
            view.payload.latent_cache,
            view.payload.rope_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            torch.empty(2, 10, 512, dtype=torch.bfloat16),
            attn_score=view.meta.attn_score,
            max_context_len=63,
        )


def test_tilelang_runner_rejects_unsupported_local_head_count() -> None:
    with pytest.raises(ValueError, match="valid_heads must be one of"):
        TileMlaDecodeKernel(
            device="cpu",
            softmax_scale=0.0625,
            valid_heads=7,
        )
