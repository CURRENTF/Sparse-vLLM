from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sparsevllm.kernels.external.sgl.moe import (
    sgl_moe_align_block_size,
    sgl_moe_alignment_support,
)
from sparsevllm.kernels.triton.moe import fused_moe, moe_align_block_size
from sparsevllm.kernels.triton.sgl_fused_moe import (
    resolve_sgl_moe_config,
    sgl_fused_moe,
)
from sparsevllm.operators.moe import _sgl_moe_align_block_size


def _torch_local_moe(
    hidden_states,
    w13_weight,
    w2_weight,
    topk_ids,
    topk_weights,
    local_expert_start,
):
    dtype = hidden_states.dtype
    output = torch.zeros(
        hidden_states.shape,
        dtype=torch.float32,
        device=hidden_states.device,
    )
    intermediate_size = int(w13_weight.shape[1]) // 2
    for local_expert_id in range(int(w13_weight.shape[0])):
        global_expert_id = int(local_expert_start) + local_expert_id
        token_ids, route_ids = torch.where(topk_ids == global_expert_id)
        if token_ids.numel() == 0:
            continue
        gate_up = F.linear(
            hidden_states[token_ids].float(),
            w13_weight[local_expert_id].float(),
        ).to(dtype)
        activated = (
            F.silu(gate_up[:, :intermediate_size].float()).to(dtype)
            * gate_up[:, intermediate_size:]
        ).to(dtype)
        routed = F.linear(
            activated.float(),
            w2_weight[local_expert_id].float(),
        )
        routed = (
            routed * topk_weights[token_ids, route_ids].float()[:, None]
        ).to(dtype)
        output.index_add_(0, token_ids, routed.float())
    return output.to(dtype)


@pytest.mark.parametrize(
    ("num_tokens", "expected"),
    [
        (1, (16, 64, 64, 1, 4, 5)),
        (4, (16, 64, 128, 1, 4, 2)),
        (1024, (64, 128, 64, 32, 4, 3)),
        (8192, (128, 256, 64, 16, 8, 4)),
    ],
)
def test_sgl_qwen3_tp2_config_matches_upstream_profile(num_tokens, expected):
    config = resolve_sgl_moe_config(
        num_tokens=num_tokens,
        top_k=8,
        num_local_experts=128,
        intermediate_size=384,
        device_name="NVIDIA H100 80GB HBM3",
    )

    assert tuple(config.values()) == expected


def test_sgl_moe_config_has_generic_shape_fallback():
    assert resolve_sgl_moe_config(
        num_tokens=3,
        top_k=2,
        num_local_experts=64,
        intermediate_size=352,
        device_name="NVIDIA H100 80GB HBM3",
    ) == {
        "BLOCK_SIZE_M": 16,
        "BLOCK_SIZE_N": 128,
        "BLOCK_SIZE_K": 32,
        "GROUP_SIZE_M": 8,
        "num_warps": 4,
        "num_stages": 4,
    }


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_moe_alignment_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(("num_tokens", "local_start"), [(1, 0), (7, 4)])
def test_sgl_fused_moe_matches_independent_torch_oracle(
    dtype,
    num_tokens,
    local_start,
) -> None:
    torch.manual_seed(20260818 + num_tokens + local_start)
    device = torch.device("cuda")
    num_experts = 8
    num_local_experts = 4
    hidden_size = 32
    intermediate_size = 24
    top_k = 2
    hidden_states = (
        torch.randn(num_tokens, hidden_size, device=device, dtype=dtype) * 0.1
    )
    w13_weight = (
        torch.randn(
            num_local_experts,
            2 * intermediate_size,
            hidden_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    w2_weight = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            device=device,
            dtype=dtype,
        )
        * 0.1
    )
    topk_ids = torch.randint(
        0,
        num_experts,
        (num_tokens, top_k),
        dtype=torch.int64,
        device=device,
    )
    topk_ids[0] = torch.tensor(
        [local_start, (local_start + num_local_experts) % num_experts],
        device=device,
    )
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        dtype=torch.float32,
        device=device,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)

    expected = _torch_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        local_start,
    )
    actual = sgl_fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        num_experts=num_experts,
        local_expert_start=local_start,
        alignment_impl=_sgl_moe_align_block_size,
    )
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_moe_alignment_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_fused_moe_cuda_graph_replay() -> None:
    torch.manual_seed(20260819)
    device = torch.device("cuda")
    hidden_states = torch.randn(2, 64, device=device, dtype=torch.bfloat16) * 0.1
    w13_weight = torch.randn(
        8, 64, 64, device=device, dtype=torch.bfloat16
    ) * 0.1
    w2_weight = torch.randn(
        8, 64, 32, device=device, dtype=torch.bfloat16
    ) * 0.1
    topk_ids = torch.randint(0, 8, (2, 2), device=device, dtype=torch.int64)
    topk_weights = torch.rand(2, 2, device=device, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    kwargs = dict(
        num_experts=8,
        local_expert_start=0,
        alignment_impl=_sgl_moe_align_block_size,
    )
    sgl_fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        **kwargs,
    )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = sgl_fused_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            **kwargs,
        )
    hidden_states.copy_(torch.randn_like(hidden_states) * 0.1)
    topk_ids.copy_(torch.randint(0, 8, topk_ids.shape, device=device))
    replay_weights = torch.rand_like(topk_weights)
    replay_weights /= replay_weights.sum(dim=-1, keepdim=True)
    topk_weights.copy_(replay_weights)
    expected = _torch_local_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        0,
    )
    graph.replay()
    torch.cuda.synchronize()

    assert torch.allclose(actual, expected, atol=3e-2, rtol=3e-2)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_moe_alignment_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
@pytest.mark.parametrize("num_tokens", [1, 64])
def test_sgl_fused_moe_matches_qwen3_tp2_shape(num_tokens) -> None:
    torch.manual_seed(20260820 + num_tokens)
    device = torch.device("cuda")
    hidden_states = torch.randn(
        num_tokens, 2048, device=device, dtype=torch.bfloat16
    ) * 0.02
    w13_weight = torch.randn(
        128, 768, 2048, device=device, dtype=torch.bfloat16
    ) * 0.02
    w2_weight = torch.randn(
        128, 2048, 384, device=device, dtype=torch.bfloat16
    ) * 0.02
    topk_ids = torch.topk(
        torch.rand(num_tokens, 128, device=device),
        k=8,
        dim=-1,
    ).indices
    topk_weights = torch.rand(num_tokens, 8, device=device, dtype=torch.float32)
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    kwargs = dict(num_experts=128, local_expert_start=0)

    expected = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        **kwargs,
    )
    actual = sgl_fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        alignment_impl=_sgl_moe_align_block_size,
        **kwargs,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)


def test_sgl_moe_support_rejects_missing_package() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        assert sgl_moe_alignment_support() == (
            False,
            "sglang-kernel is not installed",
        )


@pytest.mark.parametrize("version", ["0.4.4", "0.4.6"])
def test_sgl_moe_support_rejects_outside_declared_range(version: str) -> None:
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value=version),
    ):
        supported, reason = sgl_moe_alignment_support()

    assert not supported
    assert "sglang-kernel>=0.4.5,<0.4.6" in reason


def test_sgl_moe_support_accepts_declared_range() -> None:
    version = "0.4.5"
    module = SimpleNamespace(moe_align_block_size=lambda *_args: None)
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value=version),
        patch("importlib.import_module", return_value=module),
    ):
        supported, reason = sgl_moe_alignment_support()

    assert supported
    assert version in reason


def test_sgl_moe_support_rejects_missing_alignment_api() -> None:
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value="0.4.5"),
        patch("importlib.import_module", return_value=SimpleNamespace()),
    ):
        supported, reason = sgl_moe_alignment_support()

    assert not supported
    assert "moe_align_block_size" in reason


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_moe_alignment_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_moe_alignment_matches_grouped_assignments() -> None:
    torch.manual_seed(20260808)
    topk_ids = torch.randint(
        0,
        64,
        (32, 4),
        dtype=torch.int32,
        device="cuda",
    )
    topk_ids[0].fill_(63)
    reference_sorted, reference_experts, reference_count = moe_align_block_size(
        topk_ids,
        16,
        64,
    )
    actual = sgl_moe_align_block_size(
        topk_ids,
        block_size=16,
        num_experts=64,
    )
    torch.cuda.synchronize()
    max_padded = int(topk_ids.numel()) + 65 * 15
    assert actual.sorted_token_ids.numel() == max_padded
    assert actual.expert_ids.numel() == (max_padded + 15) // 16
    count = int(reference_count.item())
    assert int(actual.num_tokens_post_padded.item()) == count
    assert torch.equal(
        actual.expert_ids[: count // 16],
        reference_experts[: count // 16],
    )
    num_assignments = int(topk_ids.numel())
    for block_index in range(count // 16):
        block = slice(block_index * 16, (block_index + 1) * 16)
        reference_ids = sorted(
            value
            for value in reference_sorted[block].tolist()
            if value < num_assignments
        )
        actual_ids = sorted(
            value
            for value in actual.sorted_token_ids[block].tolist()
            if value < num_assignments
        )
        assert actual_ids == reference_ids


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_moe_alignment_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
@pytest.mark.parametrize(("local_start", "local_end"), [(0, 32), (32, 64)])
def test_sgl_moe_alignment_matches_ep_shard(
    local_start: int,
    local_end: int,
) -> None:
    torch.manual_seed(20260810)
    topk_ids = torch.randint(
        0,
        64,
        (32, 4),
        dtype=torch.int32,
        device="cuda",
    )
    topk_ids[0] = torch.tensor([0, 31, 32, 63], device="cuda")
    reference_sorted, reference_experts, reference_count = moe_align_block_size(
        topk_ids,
        16,
        64,
        local_expert_start=local_start,
        local_expert_end=local_end,
    )
    actual = _sgl_moe_align_block_size(
        topk_ids,
        block_size=16,
        num_experts=64,
        local_expert_start=local_start,
        local_expert_end=local_end,
    )
    torch.cuda.synchronize()
    num_assignments = int(topk_ids.numel())

    def grouped_assignments(sorted_ids, expert_ids, count):
        grouped = {}
        for block_index in range(int(count.item()) // 16):
            expert_id = int(expert_ids[block_index].item())
            if expert_id < 0:
                continue
            block = slice(block_index * 16, (block_index + 1) * 16)
            grouped.setdefault(expert_id, []).extend(
                value
                for value in sorted_ids[block].tolist()
                if value < num_assignments
            )
        return {
            expert_id: sorted(assignment_ids)
            for expert_id, assignment_ids in grouped.items()
        }

    assert grouped_assignments(
        actual.sorted_token_ids,
        actual.expert_ids,
        actual.num_tokens_post_padded,
    ) == grouped_assignments(
        reference_sorted,
        reference_experts,
        reference_count,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_moe_alignment_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
@pytest.mark.parametrize("num_tokens", [1, 2, 4, 5, 64])
@pytest.mark.parametrize("all_remote", [False, True])
@pytest.mark.parametrize("local_start", [0, 4])
def test_sgl_ep_alignment_preserves_full_fused_moe_output(
    num_tokens: int,
    all_remote: bool,
    local_start: int,
) -> None:
    torch.manual_seed(20260810 + num_tokens + int(all_remote))
    device = torch.device("cuda")
    hidden_size = 64
    intermediate_size = 32
    num_experts = 8
    num_local_experts = 4
    top_k = 2
    hidden_states = torch.randn(
        num_tokens,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w13_weight = torch.randn(
        num_local_experts,
        2 * intermediate_size,
        hidden_size,
        dtype=torch.bfloat16,
        device=device,
    )
    w2_weight = torch.randn(
        num_local_experts,
        hidden_size,
        intermediate_size,
        dtype=torch.bfloat16,
        device=device,
    )
    if all_remote:
        remote_start = num_local_experts if local_start == 0 else 0
        topk_ids = torch.randint(
            remote_start,
            remote_start + num_local_experts,
            (num_tokens, top_k),
            dtype=torch.int64,
            device=device,
        )
    else:
        topk_ids = torch.randint(
            0,
            num_experts,
            (num_tokens, top_k),
            dtype=torch.int64,
            device=device,
        )
        remote_id = num_local_experts if local_start == 0 else 0
        topk_ids[0] = torch.tensor(
            [local_start, remote_id],
            device=device,
        )
    topk_weights = torch.rand(
        num_tokens,
        top_k,
        dtype=torch.bfloat16,
        device=device,
    )
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    kwargs = {
        "num_experts": num_experts,
        "local_expert_start": local_start,
    }

    expected = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        **kwargs,
    )
    actual = fused_moe(
        hidden_states,
        w13_weight,
        w2_weight,
        topk_ids,
        topk_weights,
        alignment_impl=_sgl_moe_align_block_size,
        **kwargs,
    )
    torch.cuda.synchronize()

    assert torch.equal(actual, expected)

    if num_tokens == 2:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_actual = fused_moe(
                hidden_states,
                w13_weight,
                w2_weight,
                topk_ids,
                topk_weights,
                alignment_impl=_sgl_moe_align_block_size,
                **kwargs,
            )
        hidden_states.copy_(torch.randn_like(hidden_states))
        if all_remote:
            replay_ids = torch.randint(
                remote_start,
                remote_start + num_local_experts,
                topk_ids.shape,
                dtype=topk_ids.dtype,
                device=device,
            )
        else:
            replay_ids = torch.randint(
                0,
                num_experts,
                topk_ids.shape,
                dtype=topk_ids.dtype,
                device=device,
            )
            remote_id = num_local_experts if local_start == 0 else 0
            replay_ids[0] = torch.tensor(
                [local_start + 1, remote_id],
                device=device,
            )
        topk_ids.copy_(replay_ids)
        replay_weights = torch.rand_like(topk_weights)
        replay_weights /= replay_weights.sum(dim=-1, keepdim=True)
        topk_weights.copy_(replay_weights)
        replay_expected = fused_moe(
            hidden_states,
            w13_weight,
            w2_weight,
            topk_ids,
            topk_weights,
            **kwargs,
        )
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(graph_actual, replay_expected)
