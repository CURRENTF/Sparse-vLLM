from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sparsevllm.kernels.external.support import (
    ExternalKernelContractError,
    ExternalKernelFamilyError,
    KernelFamilyState,
)
from sparsevllm.kernels.external.sgl.fa3 import (
    _FWD_ARGUMENTS,
    SglFa3DecodeKernel,
    sgl_fa3_device_support,
    sgl_fa3_support,
)


def test_sgl_fa3_support_rejects_missing_package() -> None:
    with patch("importlib.util.find_spec", return_value=None):
        with pytest.raises(ExternalKernelFamilyError) as exc_info:
            sgl_fa3_support()

    assert exc_info.value.health.state is KernelFamilyState.ABSENT
    assert "sglang-kernel is not installed" in str(exc_info.value)


@pytest.mark.parametrize("version", ["0.4.4", "0.5.0"])
def test_sgl_fa3_support_rejects_outside_declared_range(version: str) -> None:
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value=version),
    ):
        with pytest.raises(ExternalKernelFamilyError) as exc_info:
            sgl_fa3_support()

    assert exc_info.value.health.state is KernelFamilyState.BROKEN
    assert "sglang-kernel>=0.4.5,<0.5" in str(exc_info.value)


@pytest.mark.parametrize("version", ["0.4.5", "0.4.6.post1"])
def test_sgl_fa3_support_accepts_declared_range(version: str) -> None:
    op = SimpleNamespace(
        _schema=SimpleNamespace(
            arguments=[
                SimpleNamespace(name=name)
                for name in _FWD_ARGUMENTS
            ]
        )
    )
    with (
        patch(
            "sparsevllm.kernels.external.sgl.fa3._sgl_fa3_op",
            return_value=op,
        ),
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value=version),
        patch("importlib.import_module", return_value=object()),
    ):
        supported, reason = sgl_fa3_support()

    assert supported
    assert version in reason


def test_sgl_fa3_support_rejects_binary_load_failure() -> None:
    with (
        patch("importlib.util.find_spec", return_value=object()),
        patch("importlib.metadata.version", return_value="0.4.5"),
        patch(
            "importlib.import_module",
            side_effect=ImportError("undefined symbol: c10_cuda_check"),
        ),
    ):
        with pytest.raises(ExternalKernelFamilyError) as exc_info:
            sgl_fa3_support()

    assert exc_info.value.health.state is KernelFamilyState.BROKEN
    assert "undefined symbol" in str(exc_info.value)


def test_sgl_fa3_support_rejects_missing_op_schema() -> None:
    with (
        patch(
            "sparsevllm.kernels.external.sgl.fa3.sgl_kernel_support",
            return_value=(True, "available"),
        ),
        patch(
            "sparsevllm.kernels.external.sgl.fa3._sgl_fa3_op",
            return_value=object(),
        ),
    ):
        with pytest.raises(ExternalKernelContractError) as exc_info:
            sgl_fa3_support()

    assert "failed to load" in str(exc_info.value)


def test_sgl_fa3_device_support_keeps_package_probe_and_device_probe_separate() -> None:
    with patch(
        "sparsevllm.kernels.external.sgl.fa3.sgl_fa3_support",
        side_effect=RuntimeError("ABI mismatch"),
    ):
        with pytest.raises(RuntimeError, match="ABI mismatch"):
            sgl_fa3_device_support(3)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fa3_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_fa3_decode_matches_torch_and_replays_cuda_graph() -> None:
    torch.manual_seed(20260807)
    device = torch.device("cuda")
    batch_size, heads, width = 3, 10, 8
    slots = 4 * width
    q_rope = torch.randn(
        batch_size, heads, 64, device=device, dtype=torch.bfloat16
    )
    q_latent = torch.randn(
        batch_size, heads, 512, device=device, dtype=torch.bfloat16
    )
    rope_cache = torch.randn(
        slots, 1, 64, device=device, dtype=torch.bfloat16
    )
    latent_cache = torch.randn(
        slots, 1, 512, device=device, dtype=torch.bfloat16
    )
    page_table = torch.arange(
        slots, device=device, dtype=torch.int32
    ).view(4, width)
    request_indices = torch.tensor(
        [2, 0, -1], device=device, dtype=torch.int32
    )
    context_lens = torch.tensor(
        [7, 5, 0], device=device, dtype=torch.int32
    )
    output = torch.empty_like(q_latent)
    kernel = SglFa3DecodeKernel(
        device=device,
        max_batch_size=batch_size,
        softmax_scale=256**-0.5,
    )

    validation_scope = object()
    scheduler_op = kernel._scheduler_op
    scheduler_call_count = 0
    if scheduler_op is not None:

        def counted_scheduler_op(*args, **kwargs):
            nonlocal scheduler_call_count
            scheduler_call_count += 1
            return scheduler_op(*args, **kwargs)

        kernel._scheduler_op = counted_scheduler_op
    actual = kernel(
        q_rope,
        q_latent,
        rope_cache,
        latent_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        validation_scope=validation_scope,
    )
    kernel(
        q_rope,
        q_latent,
        rope_cache,
        latent_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        validation_scope=validation_scope,
    )
    assert scheduler_call_count == int(scheduler_op is not None)
    kernel(
        q_rope,
        q_latent,
        rope_cache,
        latent_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        validation_scope=object(),
    )
    assert scheduler_call_count == 2 * int(scheduler_op is not None)
    expected_rows = []
    for batch_index in range(batch_size):
        length = int(context_lens[batch_index].item())
        if length == 0:
            expected_rows.append(torch.zeros_like(q_latent[batch_index]))
            continue
        row = int(request_indices[batch_index].item())
        active = page_table[row, :length].long()
        logits = q_rope[batch_index].float() @ rope_cache[active, 0].float().T
        logits += q_latent[batch_index].float() @ latent_cache[active, 0].float().T
        probs = torch.softmax(logits * (256**-0.5), dim=-1)
        expected_rows.append(
            (probs @ latent_cache[active, 0].float()).to(torch.bfloat16)
        )
    expected = torch.stack(expected_rows)

    torch.testing.assert_close(actual, expected, rtol=3e-2, atol=3e-2)
    graph = torch.cuda.CUDAGraph()
    graph_output = torch.empty_like(output)
    with torch.cuda.graph(graph):
        kernel(
            q_rope,
            q_latent,
            rope_cache,
            latent_cache,
            page_table,
            request_indices,
            context_lens,
            graph_output,
            validation_scope=object(),
        )
    second_graph = torch.cuda.CUDAGraph()
    second_graph_output = torch.empty_like(output)
    with torch.cuda.graph(second_graph):
        kernel(
            q_rope,
            q_latent,
            rope_cache,
            latent_cache,
            page_table,
            request_indices,
            context_lens,
            second_graph_output,
            validation_scope=object(),
        )
    graph.replay()
    second_graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(graph_output, expected, rtol=3e-2, atol=3e-2)
    torch.testing.assert_close(
        second_graph_output,
        expected,
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fa3_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_fa3_varlen_latent_prefill_matches_causal_torch() -> None:
    torch.manual_seed(20260807)
    device = torch.device("cuda")
    heads, width = 10, 8
    chunk_lens = (3, 2)
    context_lens = torch.tensor([5, 4], device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    query_tokens = int(cu_seqlens_q[-1].item())
    q_rope = torch.randn(
        query_tokens, heads, 64, device=device, dtype=torch.bfloat16
    )
    q_latent = torch.randn(
        query_tokens, heads, 512, device=device, dtype=torch.bfloat16
    )
    rope_cache = torch.randn(
        2 * width, 1, 64, device=device, dtype=torch.bfloat16
    )
    latent_cache = torch.randn(
        2 * width, 1, 512, device=device, dtype=torch.bfloat16
    )
    page_table = torch.arange(
        2 * width, device=device, dtype=torch.int32
    ).view(2, width)
    request_indices = torch.tensor([1, 0], device=device, dtype=torch.int32)
    output = torch.empty_like(q_latent)
    kernel = SglFa3DecodeKernel(
        device=device,
        max_batch_size=2,
        softmax_scale=256**-0.5,
    )

    kernel.run_varlen(
        q_rope,
        q_latent,
        rope_cache,
        latent_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(chunk_lens),
    )
    expected_rows = []
    query_start = 0
    for batch_index, chunk_len in enumerate(chunk_lens):
        context_len = int(context_lens[batch_index].item())
        row = int(request_indices[batch_index].item())
        for query_offset in range(chunk_len):
            visible_len = context_len - chunk_len + query_offset + 1
            active = page_table[row, :visible_len].long()
            query_index = query_start + query_offset
            logits = q_rope[query_index].float() @ rope_cache[active, 0].float().T
            logits += q_latent[query_index].float() @ latent_cache[active, 0].float().T
            probs = torch.softmax(logits * (256**-0.5), dim=-1)
            expected_rows.append(
                (probs @ latent_cache[active, 0].float()).to(torch.bfloat16)
            )
        query_start += chunk_len

    torch.testing.assert_close(
        output,
        torch.stack(expected_rows),
        rtol=3e-2,
        atol=3e-2,
    )


@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fa3_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_fa3_varlen_explicit_prefill_matches_causal_torch() -> None:
    torch.manual_seed(20260807)
    device = torch.device("cuda")
    heads, width, head_dim = 10, 8, 256
    chunk_lens = (3, 2)
    context_lens = torch.tensor([5, 4], device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    query_tokens = int(cu_seqlens_q[-1].item())
    q = torch.randn(
        query_tokens, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    k_cache = torch.randn(
        2 * width, heads, head_dim, device=device, dtype=torch.bfloat16
    )
    v_backing = torch.randn(
        2 * width, heads, 448, device=device, dtype=torch.bfloat16
    )
    v_cache = v_backing[..., 192:]
    assert not v_cache.is_contiguous()
    page_table = torch.arange(
        2 * width, device=device, dtype=torch.int32
    ).view(2, width)
    request_indices = torch.tensor([1, 0], device=device, dtype=torch.int32)
    output = torch.empty_like(q)
    kernel = SglFa3DecodeKernel(
        device=device,
        max_batch_size=2,
        softmax_scale=head_dim**-0.5,
    )
    validation_scope = object()

    kernel.run_explicit_varlen(
        q,
        k_cache,
        v_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(chunk_lens),
        validation_scope=validation_scope,
    )
    packed_indices = torch.cat(
        (
            page_table[1, : int(context_lens[0].item())],
            page_table[0, : int(context_lens[1].item())],
        )
    ).long()
    packed_output = torch.empty_like(q)
    kernel.run_contiguous_explicit_varlen(
        q,
        k_cache[packed_indices],
        v_cache[packed_indices],
        packed_output,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=torch.tensor(
            [0, 5, 9], device=device, dtype=torch.int32
        ),
        max_seqlen_q=max(chunk_lens),
        max_seqlen_k=int(context_lens.max().item()),
    )
    expected_rows = []
    query_start = 0
    for batch_index, chunk_len in enumerate(chunk_lens):
        context_len = int(context_lens[batch_index].item())
        row = int(request_indices[batch_index].item())
        for query_offset in range(chunk_len):
            visible_len = context_len - chunk_len + query_offset + 1
            active = page_table[row, :visible_len].long()
            query_index = query_start + query_offset
            logits = torch.einsum(
                "hd,lhd->hl",
                q[query_index].float(),
                k_cache[active].float(),
            )
            probs = torch.softmax(logits * (head_dim**-0.5), dim=-1)
            expected_rows.append(
                torch.einsum("hl,lhd->hd", probs, v_cache[active].float()).to(
                    torch.bfloat16
                )
            )
        query_start += chunk_len

    torch.testing.assert_close(
        output,
        torch.stack(expected_rows),
        rtol=3e-2,
        atol=3e-2,
    )

    # Scheduler metadata is intentionally reusable across layers, while the
    # raw op must consume each layer's own logical-to-physical page table.
    second_page_table = page_table.flip((0, 1))
    second_output = torch.empty_like(q)
    kernel.run_explicit_varlen(
        q,
        k_cache,
        v_cache,
        second_page_table,
        request_indices,
        context_lens,
        second_output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(chunk_lens),
        validation_scope=validation_scope,
    )
    expected_rows = []
    query_start = 0
    for batch_index, chunk_len in enumerate(chunk_lens):
        context_len = int(context_lens[batch_index].item())
        row = int(request_indices[batch_index].item())
        for query_offset in range(chunk_len):
            visible_len = context_len - chunk_len + query_offset + 1
            active = second_page_table[row, :visible_len].long()
            query_index = query_start + query_offset
            keys = k_cache[active]
            values = v_cache[active]
            logits = torch.einsum("hd,lhd->hl", q[query_index].float(), keys.float())
            probabilities = torch.softmax(logits * (head_dim**-0.5), dim=-1)
            expected_rows.append(
                torch.einsum("hl,lhd->hd", probabilities, values.float()).to(
                    torch.bfloat16
                )
            )
        query_start += chunk_len
    torch.testing.assert_close(
        second_output,
        torch.stack(expected_rows),
        rtol=3e-2,
        atol=3e-2,
    )
    torch.testing.assert_close(packed_output, output)


@pytest.mark.parametrize("q_heads,kv_heads", [(32, 4), (16, 2), (12, 2)])
@pytest.mark.skipif(
    not torch.cuda.is_available() or not sgl_fa3_support()[0],
    reason="CUDA and a validated sglang-kernel are required",
)
def test_sgl_fa3_qwen3_moe_gqa_prefill_matches_torch(
    q_heads: int,
    kv_heads: int,
) -> None:
    torch.manual_seed(20260817 + q_heads)
    device = torch.device("cuda")
    head_dim = 128
    chunk_lens = (3, 2)
    context_lens = torch.tensor([7, 5], device=device, dtype=torch.int32)
    cu_seqlens_q = torch.tensor([0, 3, 5], device=device, dtype=torch.int32)
    q = torch.randn(5, q_heads, head_dim, device=device, dtype=torch.bfloat16)
    k_cache = torch.randn(29, kv_heads, head_dim, device=device, dtype=torch.bfloat16)
    v_cache = torch.randn_like(k_cache)
    physical_slots = torch.randperm(29, device=device)[:12]
    page_table = torch.zeros(2, 7, device=device, dtype=torch.int32)
    page_table[0, :7] = physical_slots[:7].to(torch.int32)
    page_table[1, :5] = physical_slots[7:].to(torch.int32)
    request_indices = torch.tensor([0, 1], device=device, dtype=torch.int32)
    output = torch.empty_like(q)
    kernel = SglFa3DecodeKernel(
        device=device,
        max_batch_size=2,
        softmax_scale=head_dim**-0.5,
    )

    kernel.run_explicit_varlen(
        q,
        k_cache,
        v_cache,
        page_table,
        request_indices,
        context_lens,
        output,
        cu_seqlens_q=cu_seqlens_q,
        max_seqlen_q=max(chunk_lens),
    )

    group_size = q_heads // kv_heads
    expected_rows = []
    query_start = 0
    for batch_index, chunk_len in enumerate(chunk_lens):
        context_len = int(context_lens[batch_index].item())
        row = int(request_indices[batch_index].item())
        for query_offset in range(chunk_len):
            visible_len = context_len - chunk_len + query_offset + 1
            active = page_table[row, :visible_len].long()
            query_index = query_start + query_offset
            keys = k_cache[active].repeat_interleave(group_size, dim=1)
            values = v_cache[active].repeat_interleave(group_size, dim=1)
            logits = torch.einsum("hd,lhd->hl", q[query_index].float(), keys.float())
            probabilities = torch.softmax(logits * (head_dim**-0.5), dim=-1)
            expected_rows.append(
                torch.einsum("hl,lhd->hd", probabilities, values.float()).to(
                    torch.bfloat16
                )
            )
        query_start += chunk_len

    torch.testing.assert_close(
        output,
        torch.stack(expected_rows),
        rtol=3e-2,
        atol=3e-2,
    )
