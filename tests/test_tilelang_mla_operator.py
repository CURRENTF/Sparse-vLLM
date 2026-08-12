from __future__ import annotations

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
    select_tile_mla_config,
    tilelang_mla_support,
)
from sparsevllm.kernels.triton.mla import MlaDecodeWorkspace
from sparsevllm.operators.mla_attention import (
    MLA_ATTENTION_REGISTRY,
    MlaAttentionOpSpec,
    MlaSglFa3Provider,
    MlaTileLangScoreProvider,
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


@pytest.mark.parametrize(
    (
        "heads",
        "batch",
        "context",
        "need_score",
        "expected",
    ),
    [
        (10, 1, 1024, True, (16, 16, "direct")),
        (10, 8, 65536, True, (32, 16, "direct")),
        (5, 32, 32768, True, (8, 16, "direct")),
        (20, 1, 1024, True, (16, 16, "atomic")),
        (20, 1, 4096, True, (32, 16, "atomic")),
        (20, 1, 8192, True, (32, 16, "partial")),
        (20, 8, 16384, True, (16, 32, "direct")),
        (20, 32, 32768, True, (8, 32, "direct")),
        (20, 64, 131072, False, (8, 32, "direct")),
    ],
)
def test_tilelang_config_selection_is_static(
    heads: int,
    batch: int,
    context: int,
    need_score: bool,
    expected: tuple[int, int, str],
) -> None:
    config = select_tile_mla_config(
        batch_size=batch,
        context_capacity=context,
        need_score=need_score,
        local_q_heads=heads,
    )
    assert (config.num_split, config.block_h, config.score_mode) == expected


@pytest.mark.parametrize("tp_size", [1, 2, 4])
def test_tilelang_provider_supports_all_glm_tp_sizes(tp_size: int) -> None:
    with (
        patch(
            "sparsevllm.operators.mla_attention.sgl_fa3_support",
            return_value=(True, "sgl test"),
        ),
        patch(
            "sparsevllm.operators.mla_attention.tilelang_mla_support",
            return_value=(True, "tilelang test"),
        ),
    ):
        assert MlaTileLangScoreProvider.supports(
            _spec(tp_size=tp_size), _h100_caps()
        ).supported


def test_resolver_prefers_tilelang_score_provider_for_tp2() -> None:
    workspace = _cpu_workspace()
    with (
        patch(
            "sparsevllm.operators.mla_attention.sgl_fa3_support",
            return_value=(True, "sgl test"),
        ),
        patch(
            "sparsevllm.operators.mla_attention.tilelang_mla_support",
            return_value=(True, "tilelang test"),
        ),
        patch(
            "sparsevllm.operators.mla_attention.allocate_mla_decode_workspace",
            return_value=workspace,
        ),
        patch("sparsevllm.operators.mla_attention.SglFa3DecodeKernel"),
        patch("sparsevllm.operators.mla_attention.TileMlaDecodeKernel"),
    ):
        resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            _spec(),
            _h100_caps(),
            op_spec=_spec(),
            device="cpu",
            max_batch_size=2,
        )
    assert type(resolved.provider) is MlaTileLangScoreProvider


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


def test_missing_tilelang_keeps_existing_sgl_provider() -> None:
    workspace = _cpu_workspace()
    with (
        patch(
            "sparsevllm.operators.mla_attention.sgl_fa3_support",
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
    assert type(resolved.provider) is MlaSglFa3Provider
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
