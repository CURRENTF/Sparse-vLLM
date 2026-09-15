"""Native mixed-format expert semantics against dequantized Torch computation."""

import pytest
import torch
import torch.nn.functional as F

from sparsevllm.kernels.triton.fp8_ue8m0 import quantize_fp8_ue8m0
from sparsevllm.kernels.triton.mxfp4 import mxfp4_gemm
from sparsevllm.kernels.triton.silu_and_mul import weighted_clipped_swiglu
from sparsevllm.operators.mxfp4_moe import Mxfp4MoeSpec, resolve_mxfp4_moe_provider
from sparsevllm.operators.workspace import close_workspace_manager, lock_workspace_manager


def _dequantize_weight(packed, scales):
    # OCP E2M1 codebook, independent of Triton's scaled-dot lowering.
    table = packed.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6,
                               0, -.5, -1, -1.5, -2, -3, -4, -6], dtype=torch.float32)
    codes = torch.stack((packed & 15, packed >> 4), -1).long()
    values = table[codes].flatten(-2)
    return values * torch.exp2(scales.float() - 127).repeat_interleave(32, -1)


def _quantize_dequantize_activation(x):
    grouped = x.float().unflatten(-1, (-1, 128))
    scales = torch.exp2(torch.ceil(torch.log2(grouped.abs().amax(-1).clamp_min(1e-4) / 448)))
    return ((grouped / scales[..., None]).to(torch.float8_e4m3fn).float()
            * scales[..., None]).flatten(-2)


def _oracle(x, ids, weights, logical, local_start, limit):
    output = torch.zeros_like(x, dtype=torch.float32)
    host_ids = ids.cpu()
    for expert in range(logical[0][0].shape[0]):
        tokens, routes = torch.where(host_ids == local_start + expert)
        if not tokens.numel():
            continue
        tokens, routes = tokens.to(x.device), routes.to(x.device)
        w1, w3, w2 = (_dequantize_weight(w[expert], s[expert]) for w, s in logical)
        a = _quantize_dequantize_activation(x[tokens])
        gate = (a @ w1.T).bfloat16().float().clamp(max=limit)
        up = (a @ w3.T).bfloat16().float().clamp(-limit, limit)
        activated = (F.silu(gate) * up * weights[tokens, routes, None]).bfloat16()
        down = (_quantize_dequantize_activation(activated) @ w2.T).bfloat16().float()
        output.index_add_(0, tokens, down)
    return output


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_mxfp4_gemm_preserves_packed_values_and_group_scales():
    torch.manual_seed(31)
    m, n, k = 7, 192, 4096
    x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16)
    x[0].mul_(1e-7)
    weight = torch.randint(0, 256, (n, k//2), device="cuda", dtype=torch.uint8)
    scales = torch.randint(115, 126, (n, k//32), device="cuda", dtype=torch.uint8)
    q, qs = torch.empty_like(x, dtype=torch.float8_e4m3fn), torch.empty((m, k//128), device="cuda")
    quantize_fp8_ue8m0(x, q, qs)
    out = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    mxfp4_gemm(q, qs, weight, scales, out)
    expected = (_quantize_dequantize_activation(x) @ _dequantize_weight(weight, scales).T).bfloat16()
    torch.testing.assert_close(out, expected, rtol=1e-2, atol=3e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_weighted_swiglu_preserves_bf16_rounding_before_requantization():
    # Observed checkpoint case: approximate exp/div crossed a BF16 midpoint,
    # then changed one FP8 activation and hundreds of down-projection outputs.
    packed = torch.tensor([[-3.34375, -1.8046875]], device="cuda", dtype=torch.bfloat16)
    weights = torch.tensor([0.2556769847869873], device="cuda", dtype=torch.float32)
    ids = torch.zeros(1, device="cuda", dtype=torch.int32)
    output = torch.empty((1, 1), device="cuda", dtype=torch.bfloat16)
    weighted_clipped_swiglu(packed, ids, weights, output, local_start=0, local_end=1, limit=10.)
    expected = (F.silu(packed[:, :1].float()) * packed[:, 1:].float() * weights[:, None]).bfloat16()
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("m,local_experts,hidden,intermediate,top_k", [(7, 3, 128, 256, 3), (33, 3, 256, 256, 3), (1, 128, 128, 128, 6)])
def test_routed_mxfp4_matches_weighted_clipped_reference_and_graph(m, local_experts, hidden, intermediate, top_k):
    # Covers remote and padded routes, small/large grouping paths, clipping,
    # per-route activation quantization and changing routing during replay.
    close_workspace_manager()
    try:
        torch.manual_seed(731)
        local_start, total_experts = 1, local_experts + 2
        spec = Mxfp4MoeSpec(hidden, intermediate, total_experts, local_experts, local_start,
                           top_k, m + 5, 10.)
        provider = resolve_mxfp4_moe_provider(spec, device_index=0)
        storage = provider.allocate_weights()
        logical = []
        for name, n, k in (("gate", intermediate, hidden), ("up", intermediate, hidden),
                           ("down", hidden, intermediate)):
            w = torch.randint(0, 256, (local_experts, n, k//2), device="cuda", dtype=torch.uint8)
            s = torch.randint(121, 125, (local_experts, n, k//32), device="cuda", dtype=torch.uint8)
            logical.append((w, s))
            for e in range(local_experts):
                provider.load_projection(storage, e, name, w[e].view(torch.int8), s[e])
        x = torch.randn((m, hidden), device="cuda", dtype=torch.bfloat16) * 3
        ids = torch.rand((m, total_experts), device="cuda").argsort(-1)[:, :top_k].int().contiguous()
        ids[0, 0] = -1
        weights = torch.rand((m, top_k), device="cuda") * 1.5
        original_ids = ids.clone()
        lock_workspace_manager()
        assert provider.run(x[:0], ids[:0], weights[:0], storage).shape == (0, hidden)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            provider.run(x, ids, weights, storage)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = provider.run(x, ids, weights, storage)
        for all_remote in (False, True, False):
            ids.copy_(torch.zeros_like(ids) if all_remote else original_ids)
            graph.replay()
            expected = _oracle(x, ids, weights, logical, local_start, spec.swiglu_limit)
            eager = provider.run(x, ids, weights, storage)
            torch.testing.assert_close(captured, eager, rtol=0, atol=0)
            torch.testing.assert_close(captured, expected, rtol=1e-2, atol=3e-3)
        with pytest.raises(ValueError, match="workspace capacity"):
            provider.run(x.repeat(8, 1), ids.repeat(8, 1), weights.repeat(8, 1), storage)
    finally:
        close_workspace_manager()
