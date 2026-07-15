import pytest
import torch

from sparsevllm.triton_kernel.moe_config import (
    resolve_moe_gemm_config,
    token_bucket,
)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [(1, 1), (3, 4), (17, 32), (1024, 1024), (1025, 2048), (8192, 2048)],
)
def test_moe_token_bucket(tokens, expected):
    assert token_bucket(tokens) == expected


def test_unsupported_shape_uses_deterministic_heuristic():
    actual = resolve_moe_gemm_config(
        dtype=torch.float16,
        num_tokens=8,
        top_k=2,
        num_local_experts=16,
        hidden_size=64,
        intermediate_size=32,
        stage="w13",
        device_capability=(9, 0),
    )
    assert actual.block_m == 16
    assert actual.block_k == 32


def test_moe_config_rejects_unknown_stage():
    arguments = dict(
        dtype=torch.bfloat16,
        num_tokens=16,
        top_k=8,
        num_local_experts=64,
        hidden_size=2048,
        intermediate_size=768,
        device_capability=(9, 0),
    )
    with pytest.raises(ValueError, match="stage"):
        resolve_moe_gemm_config(**arguments, stage="w3")


def test_h20_qwen3_moe_config_is_shape_and_stage_aware():
    common = dict(
        dtype=torch.bfloat16,
        num_tokens=1,
        top_k=8,
        num_local_experts=64,
        hidden_size=2048,
        intermediate_size=768,
        device_capability=(9, 0),
    )
    w13 = resolve_moe_gemm_config(**common, stage="w13")
    w2 = resolve_moe_gemm_config(**common, stage="w2")
    assert w13.block_k == 32
    assert w2.block_k == 32

    large = resolve_moe_gemm_config(
        **{**common, "num_tokens": 512},
        stage="w13",
    )
    assert large.block_m == 64


def test_fallback_heuristic_uses_logical_assignment_count():
    common = dict(
        dtype=torch.float16,
        num_tokens=8,
        num_local_experts=16,
        hidden_size=64,
        intermediate_size=32,
        stage="w13",
        device_capability=(9, 0),
    )
    assert resolve_moe_gemm_config(**common, top_k=2).block_k == 32
    assert resolve_moe_gemm_config(**common, top_k=8).block_k == 64
