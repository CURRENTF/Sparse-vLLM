"""Candidate-domain per-head softmax followed by head-max, without a BHL probability tensor."""
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["SB", "SH", "SL", "LS", "CAPACITY", "SPLITS"])
def _partial_stats(
    Scores, Lengths, Partial, SB, SH, SL,
    LS, HEADS: tl.constexpr, CAPACITY,
    SINK: tl.constexpr, RECENT: tl.constexpr, SCALE: tl.constexpr,
    SPLITS, BLOCK: tl.constexpr,
):
    split = tl.program_id(0)
    head = tl.program_id(1)
    batch = tl.program_id(2)
    length = tl.minimum(tl.maximum(tl.load(Lengths + batch * LS) - RECENT - SINK, 0), CAPACITY - SINK)
    base = ((batch * HEADS + head) * SPLITS + split) * 2
    if split * BLOCK < length:
        token = split * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(Scores + batch * SB + head * SH + (token + SINK) * SL,
                        mask=token < length, other=-float('inf')).to(tl.float32) * SCALE
        maximum = tl.max(value, 0)
        total = tl.sum(tl.exp(value - maximum), 0)
        tl.store(Partial + base, maximum)
        tl.store(Partial + base + 1, total)
    else:
        tl.store(Partial + base, -float('inf'))
        tl.store(Partial + base + 1, 0.)


@triton.jit(do_not_specialize=["SPLITS"])
def _merge_stats(Partial, Stats, SPLITS, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    maximum = tl.load(Partial + (row * SPLITS + part) * 2, mask=part < SPLITS, other=-float('inf'))
    total = tl.load(Partial + (row * SPLITS + part) * 2 + 1, mask=part < SPLITS, other=0.)
    overall_max = tl.max(maximum, 0)
    overall_max = tl.where(overall_max == -float('inf'), 0., overall_max)
    overall_sum = tl.sum(total * tl.exp(maximum - overall_max), 0)
    tl.store(Stats + row * 2, overall_max)
    tl.store(Stats + row * 2 + 1, tl.where(overall_sum > 0, 1. / overall_sum, 0.))


@triton.jit(do_not_specialize=["SB", "SH", "SL", "LS", "CAPACITY"])
def _normalize_head_max(
    Scores, Lengths, Stats, Output,
    SB, SH, SL, LS,
    HEADS: tl.constexpr, CAPACITY, SINK: tl.constexpr, RECENT: tl.constexpr,
    SCALE: tl.constexpr, MIN_SCORE: tl.constexpr, BLOCK: tl.constexpr, HEAD_BLOCK: tl.constexpr,
):
    tile = tl.program_id(0)
    batch = tl.program_id(1)
    token = tile * BLOCK + tl.arange(0, BLOCK)
    end = tl.minimum(tl.maximum(tl.load(Lengths + batch * LS) - RECENT, SINK), CAPACITY)
    valid = (token >= SINK) & (token < end)
    score = tl.full((BLOCK,), MIN_SCORE, tl.float32)
    if (tile * BLOCK < end) & ((tile + 1) * BLOCK > SINK):
        head = tl.arange(0, HEAD_BLOCK)
        maximum = tl.load(Stats + (batch * HEADS + head) * 2, mask=head < HEADS, other=0.)
        reciprocal = tl.load(Stats + (batch * HEADS + head) * 2 + 1, mask=head < HEADS, other=0.)
        value = tl.load(Scores + batch * SB + head[:, None] * SH + token[None, :] * SL,
                        mask=(head[:, None] < HEADS) & valid[None, :], other=-float('inf')).to(tl.float32)
        probability = tl.exp(value * SCALE - maximum[:, None]) * reciprocal[:, None]
        score = tl.max(probability, 0)
        score = tl.where(valid, score, MIN_SCORE)
    tl.store(Output + batch * CAPACITY + token, score, mask=token < CAPACITY)


def launch_omnikv_decode_scores(scores, context_lens, partial, stats, output, *,
                                sink, recent, scale, min_score, block=1024, output_block=128):
    batch, heads, capacity = scores.shape
    splits = triton.cdiv(capacity - sink, block)
    _partial_stats[(splits, heads, batch)](
        scores, context_lens, partial, *scores.stride(), context_lens.stride(0),
        heads, capacity, sink, recent, scale, splits, block,
    )
    _merge_stats[(batch * heads,)](partial, stats, splits, triton.next_power_of_2(splits))
    _normalize_head_max[(triton.cdiv(capacity, output_block), batch)](
        scores, context_lens, stats, output, *scores.stride(), context_lens.stride(0),
        heads, capacity, sink, recent, scale, min_score, output_block, triton.next_power_of_2(heads),
        enable_fp_fusion=False,
    )
