"""Native BF16 index scores without a [query, head, context] intermediate."""

import triton
import triton.language as tl


@triton.jit
def _score(Q, Weights, Keys, Slots, Rows, Lengths, Scores,
           H: tl.constexpr, D: tl.constexpr, CAPACITY: tl.constexpr,
           SCORE_STRIDE: tl.constexpr, BH: tl.constexpr, BC: tl.constexpr):
    token = tl.program_id(0)
    request = tl.load(Rows + token)
    length = tl.load(Lengths + token)
    h = tl.arange(0, BH)
    d = tl.arange(0, D)
    if request >= 0 and length > 0:
        query = tl.load(Q + (token * H + h[:, None]) * D + d[None, :], h[:, None] < H, 0)
        weights = tl.load(Weights + token * H + h, h < H, 0).to(tl.float32)
        # A fixed persistent grid follows device lengths at replay, avoiding a
        # max-context grid when a short request occupies a long-context graph.
        for block in range(tl.program_id(1), tl.cdiv(length, BC), tl.num_programs(1)):
            c = block * BC + tl.arange(0, BC)
            slot = tl.load(Slots + request * CAPACITY + c, c < length, 0)
            keys = tl.load(Keys + slot[None, :] * D + d[:, None], c[None, :] < length, 0)
            dot = tl.dot(query, keys).to(tl.bfloat16).to(tl.float32)
            weighted = (tl.maximum(dot, 0.) * weights[:, None]).to(tl.bfloat16).to(tl.float32)
            score = tl.sum(weighted, 0).to(tl.bfloat16)
            tl.store(Scores + token * SCORE_STRIDE + c, score, c < length)


def compressed_index_scores(query, weights, view, scores, *, num_sms):
    if len(query):
        blocks = min(triton.cdiv(view.slots.shape[1], 64), max(1, num_sms // len(query)))
        _score[(len(query), blocks)](
            query, weights, view.keys, view.slots, view.query_rows, view.visible_lengths, scores,
            query.shape[1], query.shape[2], view.slots.shape[1], scores.stride(0),
            max(16, triton.next_power_of_2(query.shape[1])), 64, enable_fp_fusion=False,
        )
