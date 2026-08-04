from __future__ import annotations

from unittest.mock import patch

import pytest
import torch

from sparsevllm.engine.cache_manager import (
    AttentionViewMeta,
    DecodeComputeView,
    ExplicitKVPayload,
    MlaLatentPayload,
)
from sparsevllm.operators.mla_attention import (
    MLA_ATTENTION_REGISTRY,
    MlaAttentionOpSpec,
    MlaTritonProvider,
)
from sparsevllm.operators.registry import OpResolver
from sparsevllm.platforms import DeviceCaps, PlatformEnum
from sparsevllm.triton_kernel.mla import MlaDecodeWorkspace


def _spec(**overrides) -> MlaAttentionOpSpec:
    values = {
        "num_q_heads": 20,
        "kv_lora_rank": 512,
        "rope_dim": 64,
        "qk_head_dim": 256,
        "value_head_dim": 256,
        "activation_dtype": torch.bfloat16,
        "cache_dtype": torch.bfloat16,
        "tp_size": 4,
        "cuda_graph": False,
    }
    values.update(overrides)
    return MlaAttentionOpSpec(**values)


def _h100_caps(**overrides) -> DeviceCaps:
    values = {
        "platform": PlatformEnum.CUDA,
        "device_type": "cuda",
        "device_index": 0,
        "device_name": "NVIDIA H100 80GB HBM3",
        "compute_capability": (9, 0),
        "runtime_version": "12.9",
        "supports_graph_capture": True,
        "supports_torch_compile": True,
        "supports_triton": True,
        "supports_pin_memory": True,
        "supports_bfloat16": True,
        "supports_native_fp8": True,
    }
    values.update(overrides)
    return DeviceCaps(**values)


def _cpu_workspace(batch_size: int, head_count: int) -> MlaDecodeWorkspace:
    return MlaDecodeWorkspace(
        block_size=torch.empty(1, dtype=torch.int32),
        batch_start_indices=torch.empty(batch_size, dtype=torch.int32),
        mid_output=torch.empty(head_count, 1, 512, dtype=torch.float32),
        mid_logsumexp=torch.empty(head_count, 1, dtype=torch.float32),
    )


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_q_heads": 0},
        {"kv_lora_rank": 0},
        {"rope_dim": -1},
        {"qk_head_dim": 0},
        {"value_head_dim": 0},
        {"tp_size": 0},
        {"num_q_heads": 20, "tp_size": 3},
    ],
)
def test_mla_attention_spec_rejects_invalid_dimensions(overrides) -> None:
    with pytest.raises(ValueError):
        _spec(**overrides)


def test_mla_attention_scale_uses_qk_head_dimension() -> None:
    spec = _spec()

    assert spec.softmax_scale == pytest.approx(256**-0.5)
    assert spec.softmax_scale != pytest.approx((512 + 64) ** -0.5)


def test_mla_resolver_selects_h100_triton_provider() -> None:
    spec = _spec()
    workspace = _cpu_workspace(batch_size=8, head_count=spec.local_q_heads)

    with patch(
        "sparsevllm.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ) as allocate:
        resolved = OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            spec,
            _h100_caps(),
            op_spec=spec,
            device="cpu",
            max_batch_size=8,
        )

    assert isinstance(resolved.provider, MlaTritonProvider)
    assert resolved.rejected == ()
    allocate.assert_called_once_with(
        batch_size=8,
        head_count=5,
        device=torch.device("cpu"),
        config=resolved.provider.launch_config,
    )


@pytest.mark.parametrize(
    ("spec_overrides", "caps_overrides", "reason"),
    [
        ({}, {"platform": PlatformEnum.CPU}, "requires CUDA SM90"),
        ({}, {"compute_capability": (8, 0)}, "requires CUDA SM90"),
        ({}, {"device_name": "NVIDIA H100 PCIe"}, "validated"),
        ({}, {"supports_triton": False}, "does not support Triton"),
        ({}, {"supports_bfloat16": False}, "does not support BF16"),
        ({"activation_dtype": torch.float16}, {}, "BF16 activations"),
        ({"cache_dtype": torch.float16}, {}, "BF16 cache"),
        ({"kv_lora_rank": 256}, {}, "GLM MLA shape"),
        ({"tp_size": 5}, {}, "tensor parallel size"),
        ({"cuda_graph": True}, {}, "CUDA Graph"),
    ],
)
def test_mla_resolver_rejects_unvalidated_contracts(
    spec_overrides,
    caps_overrides,
    reason,
) -> None:
    with pytest.raises(RuntimeError, match=reason):
        OpResolver(MLA_ATTENTION_REGISTRY).resolve(
            _spec(**spec_overrides),
            _h100_caps(**caps_overrides),
        )


def test_mla_provider_rejects_explicit_kv_before_kernel() -> None:
    spec = _spec(tp_size=1)
    workspace = _cpu_workspace(batch_size=1, head_count=20)
    with patch(
        "sparsevllm.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ):
        provider = MlaTritonProvider(
            op_spec=spec,
            device="cpu",
            max_batch_size=1,
        )
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.tensor([[0]], dtype=torch.int32),
            req_indices=torch.tensor([0], dtype=torch.int32),
            context_lens=torch.tensor([1], dtype=torch.int32),
        ),
        payload=ExplicitKVPayload(
            k_cache=torch.empty(1, 1, 256, dtype=torch.bfloat16),
            v_cache=torch.empty(1, 1, 256, dtype=torch.bfloat16),
        ),
    )

    with (
        patch("sparsevllm.operators.mla_attention.run_mla_decode") as kernel,
        pytest.raises(TypeError, match="MlaLatentPayload"),
    ):
        provider.run(
            torch.empty(1, 20, 512, dtype=torch.bfloat16),
            torch.empty(1, 20, 64, dtype=torch.bfloat16),
            view,
            torch.empty(1, 20, 512, dtype=torch.bfloat16),
        )
    kernel.assert_not_called()


def test_mla_provider_run_does_not_resolve_or_allocate() -> None:
    spec = _spec(tp_size=4)
    workspace = _cpu_workspace(batch_size=2, head_count=5)
    with patch(
        "sparsevllm.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ):
        provider = MlaTritonProvider(
            op_spec=spec,
            device="cpu",
            max_batch_size=2,
        )
    payload = MlaLatentPayload(
        latent_cache=torch.empty(4, 1, 512, dtype=torch.bfloat16),
        rope_cache=torch.empty(4, 1, 64, dtype=torch.bfloat16),
    )
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
            req_indices=torch.tensor([0, -1], dtype=torch.int32),
            context_lens=torch.tensor([2, 0], dtype=torch.int32),
        ),
        payload=payload,
    )
    q_nope_absorbed = torch.empty(2, 5, 512, dtype=torch.bfloat16)
    q_rope = torch.empty(2, 5, 64, dtype=torch.bfloat16)
    output = torch.empty_like(q_nope_absorbed)

    with (
        patch.object(OpResolver, "resolve") as resolve,
        patch(
            "sparsevllm.operators.mla_attention.allocate_mla_decode_workspace"
        ) as allocate,
        patch(
            "sparsevllm.operators.mla_attention.run_mla_decode",
            return_value=output,
        ) as kernel,
    ):
        actual = provider.run(q_nope_absorbed, q_rope, view, output)

    assert actual is output
    resolve.assert_not_called()
    allocate.assert_not_called()
    kernel.assert_called_once_with(
        q_nope_absorbed,
        q_rope,
        payload.latent_cache,
        payload.rope_cache,
        view.meta.active_slots,
        view.meta.req_indices,
        view.meta.context_lens,
        output,
        workspace,
        softmax_scale=spec.softmax_scale,
        config=provider.launch_config,
    )


def test_mla_provider_rejects_batch_larger_than_workspace() -> None:
    spec = _spec(tp_size=4)
    workspace = _cpu_workspace(batch_size=1, head_count=5)
    with patch(
        "sparsevllm.operators.mla_attention.allocate_mla_decode_workspace",
        return_value=workspace,
    ):
        provider = MlaTritonProvider(
            op_spec=spec,
            device="cpu",
            max_batch_size=1,
        )
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.tensor([[0], [1]], dtype=torch.int32),
            req_indices=torch.tensor([0, 1], dtype=torch.int32),
            context_lens=torch.tensor([1, 1], dtype=torch.int32),
        ),
        payload=MlaLatentPayload(
            latent_cache=torch.empty(2, 1, 512, dtype=torch.bfloat16),
            rope_cache=torch.empty(2, 1, 64, dtype=torch.bfloat16),
        ),
    )

    with pytest.raises(ValueError, match="exceeds the bound workspace"):
        provider.run(
            torch.empty(2, 5, 512, dtype=torch.bfloat16),
            torch.empty(2, 5, 64, dtype=torch.bfloat16),
            view,
            torch.empty(2, 5, 512, dtype=torch.bfloat16),
        )


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required for the MLA provider integration test",
)
def test_mla_provider_runs_static_padded_batch() -> None:
    spec = _spec(tp_size=4)
    provider = MlaTritonProvider(
        op_spec=spec,
        device="cuda",
        max_batch_size=2,
    )
    q_nope_absorbed = torch.randn(
        2,
        5,
        512,
        dtype=torch.bfloat16,
        device="cuda",
    )
    q_rope = torch.randn(2, 5, 64, dtype=torch.bfloat16, device="cuda")
    payload = MlaLatentPayload(
        latent_cache=torch.randn(4, 1, 512, dtype=torch.bfloat16, device="cuda"),
        rope_cache=torch.randn(4, 1, 64, dtype=torch.bfloat16, device="cuda"),
    )
    view = DecodeComputeView(
        meta=AttentionViewMeta(
            active_slots=torch.tensor(
                [[0, 2], [1, 3]],
                dtype=torch.int32,
                device="cuda",
            ),
            req_indices=torch.tensor([0, -1], dtype=torch.int32, device="cuda"),
            context_lens=torch.tensor([2, 0], dtype=torch.int32, device="cuda"),
        ),
        payload=payload,
    )
    output = torch.empty_like(q_nope_absorbed)

    provider.run(q_nope_absorbed, q_rope, view, output)
    torch.cuda.synchronize()

    torch.testing.assert_close(output[1], torch.zeros_like(output[1]))
    assert bool(torch.isfinite(output).all().item())
