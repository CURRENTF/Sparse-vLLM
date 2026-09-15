"""CPU-side future decode capacity; physical allocation stays method-owned."""
from __future__ import annotations

from dataclasses import dataclass

from sparsevllm.engine.sequence import Sequence


@dataclass
class DecodeReservation:
    sequence: Sequence
    end: int


class DecodeReservations:
    def __init__(self, cache_manager, window: int):
        self.cache_manager = cache_manager
        self.window = window
        self.requests: dict[int, DecodeReservation] = {}

    def release(self, seq_id: int) -> None:
        self.requests.pop(int(seq_id), None)

    def outstanding(self, *, exclude: int | None = None) -> dict[str, int]:
        total: dict[str, int] = {}
        for seq_id, reservation in self.requests.items():
            if seq_id == exclude:
                continue
            seq = reservation.sequence
            remaining = max(0, reservation.end - seq.num_completion_tokens)
            if remaining:
                costs = self.cache_manager.decode_window_costs(seq, remaining)
                for name, cost in costs.items():
                    total[name] = total.get(name, 0) + int(cost)
        return total

    def acquire(self, seq: Sequence, *, allow_short: bool = False,
                prefill_reserve: dict[str, int] | None = None,
                budgets: dict[str, int] | None = None) -> bool:
        reservation = self.requests.get(seq.seq_id)
        if reservation is not None and seq.num_completion_tokens < reservation.end:
            return True
        remaining = seq.max_tokens - seq.num_completion_tokens
        if remaining <= 0 or seq.is_recompute_replay:
            return True
        tokens = min(self.window, remaining)
        if budgets is None:
            budgets = self.cache_manager.decode_window_budgets()
        outstanding = self.outstanding(exclude=seq.seq_id)

        def fits(count: int) -> bool:
            return all(
                cost + outstanding.get(name, 0) + (prefill_reserve or {}).get(name, 0) <= budgets[name]
                for name, cost in self.cache_manager.decode_window_costs(seq, count).items()
            )

        if not fits(tokens):
            if not allow_short or not fits(1):
                return False
            # A sole request must not deadlock merely because a whole window
            # does not fit. Cost hooks are monotone upper bounds over a horizon.
            lo, hi = 1, tokens
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if fits(mid):
                    lo = mid
                else:
                    hi = mid - 1
            tokens = lo
        self.requests[seq.seq_id] = DecodeReservation(seq, seq.num_completion_tokens + tokens)
        return True
