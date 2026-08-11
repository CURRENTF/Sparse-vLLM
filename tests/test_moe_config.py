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
    ("tokens", "expected"),
    [
        (1, ((32, 64, 4, 4), (32, 64, 4, 4), (32, 64, 4, 4))),
        (2, ((32, 64, 4, 4), (32, 64, 4, 4), (64, 64, 4, 3))),
        (4, ((64, 64, 4, 3), (32, 64, 4, 3), (128, 64, 4, 2))),
        (8, ((128, 64, 8, 3), (32, 64, 4, 4), (64, 64, 4, 4))),
    ],
)
def test_h20_qwen36_decode_uses_profiled_bf16_configs(tokens, expected):
    common = dict(
        dtype=torch.bfloat16,
        num_tokens=tokens,
        top_k=8,
        num_local_experts=256,
        hidden_size=2048,
        intermediate_size=512,
        device_name="NVIDIA H20",
        device_capability=(9, 0),
    )
    w13 = resolve_moe_gemm_config(**common, stage="w13")
    fused = resolve_moe_gemm_config(**common, stage="gate_up_swiglu")
    w2 = resolve_moe_gemm_config(**common, stage="w2")

    assert tuple(
        (config.block_n, config.block_k, config.num_warps, config.num_stages)
        for config in (w13, fused, w2)
    ) == expected


@pytest.mark.parametrize(
    ("device_name", "num_local_experts", "intermediate_size", "expected"),
    [
        ("NVIDIA H100 80GB HBM3", 256, 512, ((64, 3), (32, 3), (32, 3))),
        ("NVIDIA H100 80GB HBM3", 256, 256, ((32, 3), (32, 3), (32, 3))),
        ("NVIDIA H100 80GB HBM3", 128, 512, ((64, 3), (32, 3), (32, 3))),
        ("NVIDIA H20", 256, 256, ((64, 3), (32, 3), (128, 2))),
        ("NVIDIA H20", 128, 512, ((128, 3), (32, 3), (64, 4))),
    ],
)
def test_qwen36_parallel_bf16_profiles_cover_bs8(
    device_name, num_local_experts, intermediate_size, expected
):
    common = dict(
        dtype=torch.bfloat16,
        num_tokens=8,
        top_k=8,
        num_local_experts=num_local_experts,
        hidden_size=2048,
        intermediate_size=intermediate_size,
        device_name=device_name,
        device_capability=(9, 0),
    )
    configs = tuple(
        resolve_moe_gemm_config(**common, stage=stage)
        for stage in ("w13", "gate_up_swiglu", "w2")
    )
    assert tuple((config.block_n, config.num_stages) for config in configs) == expected


@pytest.mark.parametrize(
    ("num_local_experts", "intermediate_size", "expected_w2"),
    [(256, 256, (32, 64, 3)), (128, 512, (64, 64, 4))],
)
def test_h20_qwen36_parallel_decode_uses_profiled_bf16_configs(
    num_local_experts,
    intermediate_size,
    expected_w2,
):
    common = dict(
        dtype=torch.bfloat16,
        num_tokens=1,
        top_k=8,
        num_local_experts=num_local_experts,
        hidden_size=2048,
        intermediate_size=intermediate_size,
        device_name="NVIDIA H20",
        device_capability=(9, 0),
    )

    fused = resolve_moe_gemm_config(**common, stage="gate_up_swiglu")
    w2 = resolve_moe_gemm_config(**common, stage="w2")
    assert (fused.block_n, fused.block_k, fused.num_stages) == (32, 64, 4)
    assert (w2.block_n, w2.block_k, w2.num_stages) == expected_w2


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
    ("stage", "tokens", "block_n", "num_stages", "swap_ab"),
    [
        ("w13", 1, 64, 5, True),
        ("w13", 2, 64, 4, True),
        ("w13", 8, 64, 4, True),
        ("w2", 1, 64, 4, True),
        ("w2", 4, 64, 2, True),
        ("w2", 8, 64, 3, True),
    ],
)
def test_h100_qwen36_fp8_routed_config(stage, tokens, block_n, num_stages, swap_ab):
    config = resolve_fp8_routed_gemm_config(
        num_tokens=tokens,
        top_k=8,
        num_local_experts=256,
        hidden_size=2048,
        intermediate_size=512,
        stage=stage,
        device_name="NVIDIA H100 80GB HBM3",
        device_capability=(9, 0),
    )

    assert (config.block_n, config.num_stages, config.swap_ab) == (
        block_n,
        num_stages,
        swap_ab,
    )


@pytest.mark.parametrize(
    ("tokens", "expected_w13", "expected_w2"),
    [
        (1, (64, 4), (64, 4)),
        (2, (64, 4), (64, 3)),
        (4, (64, 4), (64, 2)),
        (8, (64, 4), (64, 2)),
    ],
)
def test_h100_qwen36_fp8_ep2_uses_profiled_configs(
    tokens, expected_w13, expected_w2
):
    common = dict(
        num_tokens=tokens,
        top_k=8,
        num_local_experts=128,
        hidden_size=2048,
        intermediate_size=512,
        device_name="NVIDIA H100 80GB HBM3",
        device_capability=(9, 0),
    )
    w13 = resolve_fp8_routed_gemm_config(**common, stage="w13")
    w2 = resolve_fp8_routed_gemm_config(**common, stage="w2")

    assert (w13.block_n, w13.num_stages) == expected_w13
    assert (w2.block_n, w2.num_stages) == expected_w2
    assert w13.swap_ab and w2.swap_ab


@pytest.mark.parametrize(
    ("local_experts", "intermediate_size", "expected"),
    [
        (
            256,
            512,
            {
                "w13": ((64, 4), (128, 4), (64, 4), (64, 3)),
                "w2": ((64, 3), (64, 2), (64, 2), (128, 2)),
            },
        ),
        (
            256,
            256,
            {
                "w13": ((64, 4),) * 4,
                "w2": ((64, 3), (64, 2), (64, 2), (128, 2)),
            },
        ),
        (
            128,
            512,
            {
                "w13": ((64, 4),) * 4,
                "w2": ((64, 3), (128, 4), (64, 2), (64, 2)),
            },
        ),
    ],
)
def test_h20_qwen36_fp8_profiles_cover_decode_buckets(
    local_experts, intermediate_size, expected
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
                device_name="NVIDIA H20",
                device_capability=(9, 0),
            )
            for tokens in (1, 2, 4, 8)
        )
        assert tuple(
            (config.block_n, config.num_stages) for config in configs
        ) == stage_expected
        assert all(config.swap_ab for config in configs)


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
