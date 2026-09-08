from types import SimpleNamespace

import pytest
import torch

from sparsevllm.engine.cache_manager.base import (
    AttentionViewMeta,
    MlaLatentPayload,
    PrefillComputeView,
    PrefillScoreRequest,
)
from sparsevllm.operators.mla_attention import MlaAttentionOpSpec, MlaSglFa3Provider
from sparsevllm.operators.mla_prefill import ChunkedMlaPrefill


def make_case(contexts=(73, 29, 41), queries=(17, 29, 9), heads=5):
    torch.manual_seed(129)
    device, dtype = "cuda", torch.bfloat16
    spec = MlaAttentionOpSpec(
        num_q_heads=heads,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=dtype,
        cache_dtype=dtype,
        tp_size=1,
        cuda_graph=False,
    )
    capacity = sum(contexts)
    latent = torch.randn(capacity, 1, 512, device=device, dtype=dtype)
    rope = torch.randn(capacity, 1, 64, device=device, dtype=dtype) * 0.2
    weight = torch.randn(heads, 448, 512, device=device, dtype=dtype) * 0.03
    q = torch.randn(sum(queries), heads, 256, device=device, dtype=dtype) * 0.3
    slots = torch.full(
        (len(contexts), max(contexts)), -1, device=device, dtype=torch.int32
    )
    rows = tuple(reversed(range(len(contexts))))
    permutation = torch.randperm(capacity, device=device).int()
    start = 0
    for row, n in zip(rows, contexts):
        slots[row, :n] = permutation[start : start + n]
        start += n
    view = PrefillComputeView(
        AttentionViewMeta(
            slots,
            torch.tensor(rows, device=device, dtype=torch.int32),
            torch.tensor(contexts, device=device, dtype=torch.int32),
        ),
        MlaLatentPayload(latent, rope),
    )
    cu = torch.tensor(
        [0, *torch.tensor(queries).cumsum(0).tolist()], device=device, dtype=torch.int32
    )

    def project(x):
        return torch.nn.functional.linear(x, weight.flatten(0, 1))

    def absorb(x):
        return torch.bmm(x.transpose(0, 1), weight[:, :192]).transpose(0, 1)

    return spec, q, view, cu, project, absorb


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize(
    "mode,full_normalizer,backend",
    [
        ("logits", False, "fa3"),
        ("probability", False, "fa3"),
        ("probability", True, "triton"),
        ("probability", True, "fa3"),
    ],
)
def test_chunked_attention_and_scores_match_explicit_oracle(
    mode, full_normalizer, backend
):
    # Independent full softmax protects masks, global normalization, physical
    # slot indirection, and score reductions across uneven history/query blocks.
    spec, q, view, cu, project, absorb = make_case()
    provider = (
        MlaSglFa3Provider(op_spec=spec, device="cuda:0", max_batch_size=3)
        if backend == "fa3"
        else SimpleNamespace()
    )
    runner = ChunkedMlaPrefill(spec, provider, 19)
    contexts, starts = view.meta.context_lens.tolist(), cu.tolist()
    ranges = tuple(
        (n - min(7, b - a), n) for n, a, b in zip(contexts, starts, starts[1:])
    )
    request = PrefillScoreRequest(
        ranges, mode, 0 if full_normalizer else 3, 0 if full_normalizer else 4
    )
    actual, lse, scores = runner.run(q, view, cu, object(), project, absorb, request)
    for i, n in enumerate(contexts):
        a, b = starts[i : i + 2]
        row = int(view.meta.req_indices[i])
        indices = view.meta.active_slots[row, :n].long()
        latent = view.payload.latent_cache[indices, 0]
        rope = view.payload.rope_cache[indices, 0]
        expanded = project(latent).view(n, spec.local_q_heads, 448)
        k = torch.cat(
            (expanded[..., :192], rope[:, None].expand(-1, spec.local_q_heads, -1)), -1
        )
        v = expanded[..., 192:]
        raw = torch.einsum("qhd,khd->hqk", q[a:b].float(), k.float())
        qi = torch.arange(n - (b - a), n, device=q.device)
        ki = torch.arange(n, device=q.device)
        mask = qi[:, None] >= ki[None, :]
        z = (raw * spec.softmax_scale).masked_fill(~mask[None], -torch.inf)
        expected = torch.einsum("hqk,khd->qhd", z.softmax(-1), v.float())
        torch.testing.assert_close(actual[a:b].float(), expected, atol=0.006, rtol=0.03)
        torch.testing.assert_close(lse[:, a:b], z.logsumexp(-1), atol=0.003, rtol=0.001)
        observed = raw[:, -(ranges[i][1] - ranges[i][0]) :]
        valid = (
            mask[-observed.shape[1] :]
            & (ki[None] >= request.candidate_start)
            & (ki[None] < n - request.recent_keep_tokens)
        )
        if mode == "logits":
            ref = observed.masked_fill(~valid[None], -torch.inf).amax((0, 1))
            tolerance = 0.001
        else:
            ref = (
                (observed * spec.softmax_scale)
                .masked_fill(~valid[None], -torch.inf)
                .softmax(-1)
                .mean(1)
                .amax(0)
            )
            tolerance = 0.0002
        torch.testing.assert_close(scores[i, :n], ref, atol=tolerance, rtol=0.015)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mode", [None, "logits", "probability"])
def test_chunk_size_preserves_outputs_and_bounds_history_projection(mode):
    spec, q, view, cu, project, absorb = make_case(contexts=(171, 53), queries=(35, 5))
    provider = MlaSglFa3Provider(op_spec=spec, device="cuda:0", max_batch_size=2)
    # Scoring must not re-project historical KV after the attention pass.
    request = PrefillScoreRequest(((166, 171), (48, 53)), mode, 3, 4) if mode else None
    outputs = []
    for size in (17, 64):
        projected = []

        def tracked_project(x, projected=projected):
            projected.append(x.shape[0])
            return project(x)

        runner = ChunkedMlaPrefill(spec, provider, size)
        outputs.append(
            runner.run(q, view, cu, object(), tracked_project, absorb, request)[0]
        )
        assert projected[0] == q.shape[0]
        assert max(projected[1:]) <= size
        assert sum(projected) == sum(view.meta.context_lens.tolist())
    torch.testing.assert_close(outputs[0], outputs[1], atol=0.006, rtol=0.03)


def test_history_budget_is_bounded_and_plan_released():
    # Long contexts previously grew the full-history workspace and retained the
    # temporary startup cache's mapping after runtime retirement.
    import weakref

    from sparsevllm.operators.mla_prefill import estimate_mla_prefill_workspace_bytes

    spec = MlaAttentionOpSpec(
        num_q_heads=20,
        kv_lora_rank=512,
        rope_dim=64,
        qk_head_dim=256,
        value_head_dim=256,
        activation_dtype=torch.bfloat16,
        cache_dtype=torch.bfloat16,
        tp_size=1,
        cuda_graph=False,
    )
    runner = ChunkedMlaPrefill(spec, SimpleNamespace(), 256)
    estimates = []
    for length in (1024, 8192):
        view = PrefillComputeView(
            AttentionViewMeta(
                torch.arange(length, dtype=torch.int32)[None],
                torch.tensor([0], dtype=torch.int32),
                torch.tensor([length], dtype=torch.int32),
            ),
            MlaLatentPayload(torch.empty(0, 1, 512), torch.empty(0, 1, 64)),
        )
        cu = torch.tensor([0, 128], dtype=torch.int32)
        scope = object()
        plan = runner.prepare(view, cu, scope)
        assert runner.prepare(view, cu, scope) is plan
        assert all(n <= runner.chunk_size for _, _, n, _ in plan.history_chunks)
        estimates.append(
            estimate_mla_prefill_workspace_bytes(
                plan=plan,
                spec=spec,
                chunk_size=256,
                hidden_size=2048,
                projection_chunk_size=128,
            )
        )
    # Only tiny packing metadata grows with history; expanded KV does not.
    assert estimates[1] - estimates[0] < 4096
    mapping = weakref.ref(view.meta.active_slots)
    del view, plan
    runner.clear()
    assert mapping() is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_multi_tile_observations_and_empty_candidates():
    # H2O can observe more than one 32-query tile. Entire candidate blocks may
    # be causally masked; they must contribute zero probability, never NaN.
    spec, q, view, cu, project, absorb = make_case((239,), (137,), heads=20)
    provider = MlaSglFa3Provider(op_spec=spec, device="cuda:0", max_batch_size=1)
    runner = ChunkedMlaPrefill(spec, provider, 53)
    req = PrefillScoreRequest(((102, 239),), "probability", 130, 3)
    _, _, scores = runner.run(q, view, cu, object(), project, absorb, req)
    slots = view.meta.active_slots[0, :239].long()
    latent, rope = runner.gather(view.payload, slots)
    expanded = project(latent).view(239, 20, 448)
    keys = torch.cat((expanded[..., :192], rope[:, None].expand(-1, 20, -1)), -1)
    z = torch.einsum("qhd,khd->hqk", q.float(), keys.float()) * spec.softmax_scale
    ki = torch.arange(239, device=q.device)
    valid = (
        (torch.arange(102, 239, device=q.device)[:, None] >= ki)
        & (ki >= 130)
        & (ki < 236)
    )
    p = z.masked_fill(~valid, -torch.inf).softmax(-1).nan_to_num(0)
    ref = p.mean(1).amax(0)
    torch.testing.assert_close(scores[0], ref, atol=2e-4, rtol=0.015)
