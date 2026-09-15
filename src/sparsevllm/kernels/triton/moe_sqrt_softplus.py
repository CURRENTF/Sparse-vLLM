"""Native sqrt-softplus routing scores and hash lookup."""

import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _score(value):
    # Launches disable libdevice FTZ: exp(-99) remains nonzero in the
    # reference and normalizes to finite routing weights after sqrt.
    softplus = tl.where(value > 20.0, value, libdevice.log1p(libdevice.exp(value)))
    return libdevice.sqrt(softplus)


@triton.jit
def _biased_scores(Logits, Bias, Out, E: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    expert = tl.arange(0, BLOCK)
    value = tl.load(Logits + row * E + expert, expert < E, 0.0)
    bias = tl.load(Bias + expert, expert < E, 0.0)
    tl.store(Out + row * E + expert, _score(value) + bias, expert < E)


@triton.jit
def _gather(Logits, Ids, Table, Tokens, Weights,
            E: tl.constexpr, K: tl.constexpr, VOCAB: tl.constexpr,
            HASH: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    if HASH:
        token = tl.load(Tokens + row)
        valid = (token >= 0) & (token < VOCAB) & (k < K)
        expert = tl.load(Table + token * K + k, valid, -1)
        tl.store(Ids + row * K + k, expert, k < K)
    else:
        expert = tl.load(Ids + row * K + k, k < K, -1)
    valid = (k < K) & (expert >= 0) & (expert < E)
    value = tl.load(Logits + row * E + expert, valid, 0.0)
    tl.store(Weights + row * K + k, tl.where(valid, _score(value), 0.0), k < K)


@triton.jit
def _normalize(Weights, Norms, Ids, K: tl.constexpr, SCALE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    norm = tl.load(Norms + row)
    weight = tl.load(Weights + row * K + k, k < K, 0.0)
    expert = tl.load(Ids + row * K + k, k < K, -1)
    weight = tl.where(expert >= 0, tl.div_rn(weight, norm) * SCALE, 0.0)
    tl.store(Weights + row * K + k, weight, k < K)


def biased_sqrt_softplus_scores(logits, bias, out):
    _biased_scores[(logits.shape[0],)](logits, bias, out, logits.shape[1],
                                     triton.next_power_of_2(logits.shape[1]), enable_fp_fusion=False, enable_reflect_ftz=False)


def gather_sqrt_softplus_scores(logits, ids, weights, *, table=None, token_ids=None):
    _gather[(logits.shape[0],)](
        logits, ids, table, token_ids, weights, logits.shape[1], ids.shape[1],
        table.shape[0] if table is not None else 0, table is not None,
        triton.next_power_of_2(ids.shape[1]), enable_fp_fusion=False, enable_reflect_ftz=False,
    )


def normalize_sqrt_softplus_weights(weights, norms, ids, scale):
    _normalize[(weights.shape[0],)](weights, norms, ids, weights.shape[1], scale,
                                  triton.next_power_of_2(weights.shape[1]), enable_fp_fusion=False, enable_reflect_ftz=False)
