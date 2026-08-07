import pytest
import torch

from sparsevllm.triton_kernel.moe_config import (
    resolve_fp8_routed_gemm_config,
    resolve_moe_gemm_config,
    token_bucket,
)


@pytest.mark.parametrize(
    ("tokens", "expected"),
    [(1, 1), (3, 2), (17, 16), (1024, 1024), (1025, 1024), (8192, 2048)],
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
        device_name="NVIDIA H20",
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
        device_name="NVIDIA H20",
        device_capability=(9, 0),
    )
    with pytest.raises(ValueError, match="stage"):
        resolve_moe_gemm_config(**arguments, stage="w3")


def test_glm_fused_shared_decode_reuses_profiled_tile():
    config = resolve_moe_gemm_config(
        dtype=torch.bfloat16,
        num_tokens=32,
        top_k=5,
        num_local_experts=65,
        hidden_size=2048,
        intermediate_size=768,
        stage="w2",
        device_name="NVIDIA H100 80GB HBM3",
        device_capability=(9, 0),
    )

    assert (
        config.block_m,
        config.block_n,
        config.block_k,
        config.group_m,
    ) == (16, 64, 128, 16)


def test_h100_tp_ep_fused_gate_up_uses_dedicated_profile():
    common = dict(
        dtype=torch.bfloat16,
        top_k=8,
        num_local_experts=64,
        hidden_size=2048,
        intermediate_size=384,
        stage="gate_up_swiglu",
        device_name="NVIDIA H100 80GB HBM3",
        device_capability=(9, 0),
    )
    small = resolve_moe_gemm_config(**common, num_tokens=4)
    large = resolve_moe_gemm_config(**common, num_tokens=1024)

    assert (small.block_n, small.block_k, small.num_stages) == (32, 64, 4)
    assert (large.block_n, large.block_k, large.num_stages) == (128, 64, 3)


def test_fused_gate_up_fallback_is_stage_specific():
    common = dict(
        dtype=torch.bfloat16,
        num_tokens=4,
        top_k=8,
        num_local_experts=16,
        hidden_size=256,
        intermediate_size=64,
        device_name="NVIDIA H100 80GB HBM3",
        device_capability=(9, 0),
    )
    w13 = resolve_moe_gemm_config(**common, stage="w13")
    fused = resolve_moe_gemm_config(**common, stage="gate_up_swiglu")

    assert w13.block_n == 128
    assert fused.block_n == 32


def test_h20_qwen3_moe_config_is_shape_and_stage_aware():
    common = dict(
        dtype=torch.bfloat16,
        num_tokens=1,
        top_k=8,
        num_local_experts=64,
        hidden_size=2048,
        intermediate_size=768,
        device_name="NVIDIA H20",
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


@pytest.mark.parametrize(
    ("device_name", "num_local_experts", "intermediate_size", "expected"),
    [
        (
            "NVIDIA H20",
            256,
            512,
            {
                "w13": ((32, 4, 4), (32, 4, 4), (64, 4, 3), (128, 8, 3)),
                "gate_up_swiglu": ((32, 4, 4), (32, 4, 4), (32, 4, 3), (32, 4, 4)),
                "w2": ((32, 4, 4), (64, 4, 3), (128, 4, 2), (64, 4, 4)),
            },
        ),
        (
            "NVIDIA H20",
            256,
            256,
            {
                "w13": ((32, 4, 4), (32, 4, 4), (32, 4, 4), (64, 4, 3)),
                "gate_up_swiglu": ((32, 4, 4), (32, 4, 4), (32, 4, 4), (32, 4, 3)),
                "w2": ((32, 4, 3), (64, 4, 2), (128, 4, 2), (128, 4, 2)),
            },
        ),
        (
            "NVIDIA H20",
            128,
            512,
            {
                "w13": ((32, 4, 4), (64, 4, 4), (64, 4, 4), (128, 8, 3)),
                "gate_up_swiglu": ((32, 4, 4), (32, 4, 4), (32, 4, 4), (32, 4, 3)),
                "w2": ((64, 4, 4), (32, 4, 3), (64, 4, 3), (64, 4, 4)),
            },
        ),
        (
            "NVIDIA H100 80GB HBM3",
            256,
            512,
            {
                "w13": ((32, 4, 4), (64, 4, 4), (64, 4, 3), (64, 4, 3)),
                "gate_up_swiglu": ((32, 4, 4), (32, 4, 4), (32, 4, 3), (32, 4, 3)),
                "w2": ((32, 4, 4), (32, 4, 4), (64, 4, 3), (32, 4, 3)),
            },
        ),
        (
            "NVIDIA H100 80GB HBM3",
            256,
            256,
            {
                "w13": ((32, 4, 4), (32, 4, 4), (32, 4, 4), (32, 4, 3)),
                "gate_up_swiglu": ((32, 4, 4), (32, 4, 4), (32, 4, 4), (32, 4, 3)),
                "w2": ((32, 4, 4), (64, 4, 4), (64, 4, 2), (32, 4, 3)),
            },
        ),
        (
            "NVIDIA H100 80GB HBM3",
            128,
            512,
            {
                "w13": ((32, 4, 4), (32, 4, 4), (32, 4, 4), (64, 4, 3)),
                "gate_up_swiglu": ((32, 4, 4), (32, 4, 4), (32, 4, 4), (32, 4, 3)),
                "w2": ((32, 4, 4), (32, 4, 4), (32, 4, 3), (32, 4, 3)),
            },
        ),
    ],
)
def test_qwen36_bf16_profiles_cover_decode_buckets(
    device_name, num_local_experts, intermediate_size, expected
):
    for stage, stage_expected in expected.items():
        configs = tuple(
            resolve_moe_gemm_config(
                dtype=torch.bfloat16,
                num_tokens=tokens,
                top_k=8,
                num_local_experts=num_local_experts,
                hidden_size=2048,
                intermediate_size=intermediate_size,
                stage=stage,
                device_name=device_name,
                device_capability=(9, 0),
            )
            for tokens in (1, 2, 4, 8)
        )
        assert tuple(
            (config.block_n, config.num_warps, config.num_stages)
            for config in configs
        ) == stage_expected
        assert all(config.block_m == 16 and config.block_k == 64 for config in configs)


def test_fallback_heuristic_uses_logical_assignment_count():
    common = dict(
        dtype=torch.float16,
        num_tokens=8,
        num_local_experts=16,
        hidden_size=64,
        intermediate_size=32,
        stage="w13",
        device_name="NVIDIA H20",
        device_capability=(9, 0),
    )
    assert resolve_moe_gemm_config(**common, top_k=2).block_k == 32
    assert resolve_moe_gemm_config(**common, top_k=8).block_k == 64


def test_tuned_config_matches_hardware_profile():
    common = dict(
        dtype=torch.bfloat16,
        num_tokens=4,
        top_k=8,
        num_local_experts=128,
        hidden_size=2048,
        intermediate_size=768,
        stage="w13",
        device_capability=(9, 0),
    )
    assert resolve_moe_gemm_config(**common, device_name="NVIDIA H20").block_k == 32
    assert (
        resolve_moe_gemm_config(
            **common,
            device_name="NVIDIA H100 80GB HBM3",
        ).block_k
        == 64
    )
    assert resolve_moe_gemm_config(**common, device_name="NVIDIA H200").block_k == 32


def test_h100_profile_switches_to_large_token_config():
    common = dict(
        dtype=torch.bfloat16,
        top_k=8,
        num_local_experts=32,
        hidden_size=2048,
        intermediate_size=768,
        stage="w13",
        device_name="NVIDIA H100 80GB HBM3",
        device_capability=(9, 0),
    )
    assert resolve_moe_gemm_config(**common, num_tokens=512).block_m == 16
    assert resolve_moe_gemm_config(**common, num_tokens=1024).block_m == 64


@pytest.mark.parametrize(
    ("device_name", "local_experts", "intermediate_size", "expected"),
    [
        (
            "NVIDIA H20",
            256,
            512,
            {
                "w13": ((64, 4), (128, 4), (64, 4), (64, 3)),
                "w2": ((64, 3), (64, 2), (64, 2), (128, 2)),
            },
        ),
        (
            "NVIDIA H20",
            256,
            256,
            {
                "w13": ((64, 4),) * 4,
                "w2": ((64, 3), (64, 2), (64, 2), (128, 2)),
            },
        ),
        (
            "NVIDIA H20",
            128,
            512,
            {
                "w13": ((64, 4),) * 4,
                "w2": ((64, 3), (128, 4), (64, 2), (64, 2)),
            },
        ),
        (
            "NVIDIA H100 80GB HBM3",
            256,
            512,
            {
                "w13": ((64, 5), (64, 4), (64, 3), (64, 4)),
                "w2": ((64, 4), (128, 4), (64, 2), (64, 3)),
            },
        ),
        (
            "NVIDIA H100 80GB HBM3",
            256,
            256,
            {
                "w13": ((64, 4), (64, 5), (64, 4), (64, 3)),
                "w2": ((64, 3), (64, 2), (64, 2), (64, 2)),
            },
        ),
        (
            "NVIDIA H100 80GB HBM3",
            128,
            512,
            {
                "w13": ((64, 4),) * 4,
                "w2": ((64, 4), (64, 3), (64, 2), (64, 2)),
            },
        ),
    ],
)
def test_qwen36_fp8_profiles_cover_decode_buckets(
    device_name, local_experts, intermediate_size, expected
):
    for stage, stage_expected in expected.items():
        configs = tuple(
            resolve_fp8_routed_gemm_config(
                num_tokens=tokens,
                top_k=8,
                num_local_experts=local_experts,
                hidden_size=2048,
                intermediate_size=intermediate_size,
                stage=stage,
                device_name=device_name,
                device_capability=(9, 0),
            )
            for tokens in (1, 2, 4, 8)
        )
        assert tuple(
            (config.block_n, config.num_stages) for config in configs
        ) == stage_expected
        assert all(
            config.block_m == 16
            and config.block_k == 128
            and config.num_warps == 4
            and config.swap_ab
            for config in configs
        )


def test_fp8_routed_unknown_shape_uses_explicit_default():
    config = resolve_fp8_routed_gemm_config(
        num_tokens=1,
        top_k=8,
        num_local_experts=64,
        hidden_size=2048,
        intermediate_size=512,
        stage="w13",
        device_name="NVIDIA H100 80GB HBM3",
        device_capability=(9, 0),
    )

    assert (config.block_n, config.block_k, config.swap_ab) == (128, 128, False)


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (32, (16, 64, 128, 16)),
        (64, (64, 128, 64, 8)),
        (512, (64, 128, 64, 1)),
        (1024, (128, 128, 64, 1)),
        (65536, (128, 128, 64, 1)),
    ],
)
def test_glm_h100_tp2_profile_covers_decode_and_large_prefill(
    num_tokens,
    expected,
):
    common = dict(
        dtype=torch.bfloat16,
        num_tokens=num_tokens,
        top_k=4,
        num_local_experts=64,
        hidden_size=2048,
        intermediate_size=768,
        device_name="NVIDIA H100 80GB HBM3",
        device_capability=(9, 0),
    )
    for stage in ("w13", "w2"):
        config = resolve_moe_gemm_config(**common, stage=stage)
        assert (
            config.block_m,
            config.block_n,
            config.block_k,
            config.group_m,
        ) == expected
