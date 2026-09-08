"""Bounded MLA attention partials and online output merging."""

import torch
import triton
import triton.language as tl


@triton.jit
def _attention(
    Q,
    K,
    V,
    O,
    L,
    CQ,
    CK,
    q0: tl.constexpr,
    q1: tl.constexpr,
    k0: tl.constexpr,
    k1: tl.constexpr,
    v0: tl.constexpr,
    v1: tl.constexpr,
    TOTAL_Q,
    H: tl.constexpr,
    D: tl.constexpr,
    SCALE: tl.constexpr,
    CAUSAL: tl.constexpr,
    M: tl.constexpr = 32,
    N: tl.constexpr = 64,
):
    batch, head, block = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    qs, qe = tl.load(CQ + batch), tl.load(CQ + batch + 1)
    ks, ke = tl.load(CK + batch), tl.load(CK + batch + 1)
    qi = block * M + tl.arange(0, M)
    di = tl.arange(0, D)
    q = tl.load(
        Q + (qs + qi[:, None]) * q0 + head * q1 + di[None, :], qi[:, None] < qe - qs, 0
    )
    maximum = tl.full((M,), -float("inf"), tl.float32)
    denominator = tl.zeros((M,), tl.float32)
    acc = tl.zeros((M, D), tl.float32)
    for start in range(0, ke - ks, N):
        ki = start + tl.arange(0, N)
        k = tl.load(
            K + (ks + ki[None, :]) * k0 + head * k1 + di[:, None],
            ki[None, :] < ke - ks,
            0,
        )
        z = tl.dot(q, k) * SCALE
        valid = ki[None, :] < ke - ks
        if CAUSAL:
            valid = valid & (ki[None, :] <= qi[:, None] + (ke - ks) - (qe - qs))
        z = tl.where(valid, z, -float("inf"))
        updated = tl.maximum(maximum, tl.max(z, 1))
        safe = tl.where(updated == -float("inf"), 0.0, updated)
        p = tl.exp(z - safe[:, None])
        alpha = tl.exp(maximum - safe)
        v = tl.load(
            V + (ks + ki[:, None]) * v0 + head * v1 + di[None, :],
            ki[:, None] < ke - ks,
            0,
        )
        acc = acc * alpha[:, None] + tl.dot(p.to(v.dtype), v)
        denominator = denominator * alpha + tl.sum(p, 1)
        maximum = updated
    out = acc / tl.where(denominator > 0, denominator, 1.0)[:, None]
    tl.store(
        O + ((qs + qi[:, None]) * H + head) * D + di[None, :],
        out,
        qi[:, None] < qe - qs,
    )
    tl.store(L + head * TOTAL_Q + qs + qi, maximum + tl.log(denominator), qi < qe - qs)


def attention_partial(q, k, v, cu_q, cu_k, max_q, max_k, *, scale, causal):
    output = torch.empty_like(q)
    lse = torch.empty((q.shape[1], q.shape[0]), device=q.device, dtype=torch.float32)
    _attention[(cu_q.numel() - 1, q.shape[1], triton.cdiv(max_q, 32))](
        q,
        k,
        v,
        output,
        lse,
        cu_q,
        cu_k,
        *q.stride()[:2],
        *k.stride()[:2],
        *v.stride()[:2],
        q.shape[0],
        q.shape[1],
        q.shape[2],
        scale,
        causal,
    )
    return output, lse


@triton.jit
def _merge(
    O,
    L,
    P,
    PL,
    o0: tl.constexpr,
    o1: tl.constexpr,
    l0,
    l1,
    p0: tl.constexpr,
    p1: tl.constexpr,
    pl0,
    pl1,
    QN,
    H: tl.constexpr,
    D: tl.constexpr,
    BD: tl.constexpr,
    ROWS: tl.constexpr = 8,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    qi, head = row // H, row % H
    valid = qi < QN
    a = tl.load(L + head * l0 + qi * l1, valid, -float("inf"))
    b = tl.load(PL + head * pl0 + qi * pl1, valid, -float("inf"))
    m = tl.maximum(a, b)
    safe = tl.where(m == -float("inf"), 0.0, m)
    wa, wb = tl.exp(a - safe), tl.exp(b - safe)
    total = wa + wb
    denom = tl.where(total > 0, total, 1.0)
    d = tl.arange(0, BD)
    old = tl.load(
        O + qi[:, None] * o0 + head[:, None] * o1 + d[None, :],
        valid[:, None] & (d[None, :] < D),
        0,
    ).to(tl.float32)
    partial = tl.load(
        P + qi[:, None] * p0 + head[:, None] * p1 + d[None, :],
        valid[:, None] & (d[None, :] < D),
        0,
    ).to(tl.float32)
    result = old * (wa / denom)[:, None] + partial * (wb / denom)[:, None]
    tl.store(
        O + qi[:, None] * o0 + head[:, None] * o1 + d[None, :],
        result,
        valid[:, None] & (d[None, :] < D),
    )
    tl.store(L + head * l0 + qi * l1, safe + tl.log(total), valid)


def merge_partial(output, lse, partial, partial_lse):
    _merge[(triton.cdiv(output.shape[0] * output.shape[1], 8),)](
        output,
        lse,
        partial,
        partial_lse,
        *output.stride()[:2],
        *lse.stride(),
        *partial.stride()[:2],
        *partial_lse.stride(),
        output.shape[0],
        output.shape[1],
        output.shape[2],
        triton.next_power_of_2(output.shape[2]),
    )
