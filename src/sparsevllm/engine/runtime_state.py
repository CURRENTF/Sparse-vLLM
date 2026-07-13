from __future__ import annotations

from collections import deque
from typing import Protocol

import torch

from sparsevllm.config import Config
from sparsevllm.engine.prefix_cache_coordinator import PrefixCacheCoordinator
from sparsevllm.engine.recurrent_state_manager import RecurrentStateManager
from sparsevllm.engine.sequence import Sequence


class MemoryOracle(Protocol):
    """Explicit scheduler-facing memory and admission interface."""

    @property
    def num_free_slots(self) -> int: ...

    def prefill_batched_tokens_margin(self) -> int: ...
    def remaining_prefill_tokens(self, seq: Sequence) -> int: ...
    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], chunk_prefill_size: int) -> int: ...
    def should_schedule_full_prefill(self, seq: Sequence) -> bool: ...
    def requires_full_prefill_step(self, seq: Sequence) -> bool: ...
    def requires_long_prefill_offload(self, seq: Sequence) -> bool: ...
    def prefill_step_free_slots(self) -> int: ...
    def prefill_step_free_slots_for(self, seq: Sequence) -> int: ...
    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int: ...
    def decode_step_free_slots(self) -> int: ...
    def decode_step_free_slots_for(self, seq: Sequence) -> int: ...
    def decode_step_reservation_cost(self, seq: Sequence) -> int: ...
    def prompt_admission_free_slots(self) -> int: ...
    def prompt_admission_budgets(self, waiting_seqs: deque[Sequence], chunk_prefill_size: int) -> dict[str, int]: ...
    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]: ...
    def prompt_logical_reservation_cost(self, seq: Sequence) -> int: ...
    def prompt_admission_failure_action(self) -> str: ...
    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]) -> None: ...
    def refresh_prefix_cache_hit(self, seq: Sequence) -> None: ...
    def clear_prefix_cache_hit(self, seq: Sequence) -> None: ...
    def free_slot_stats(self) -> dict[str, int]: ...
    def debug_live_seq_slots(self) -> dict[int, int]: ...


class RuntimeState:
    """Single lifecycle entrypoint for KV, recurrent state, and mixed prefix cache."""

    def __init__(
        self,
        config: Config,
        cache_manager,
        recurrent_state_manager: RecurrentStateManager | None = None,
        prefix_cache_coordinator: PrefixCacheCoordinator | None = None,
    ):
        self.config = config
        self.cache_manager = cache_manager
        self.recurrent_state_manager = recurrent_state_manager
        self.prefix_cache_coordinator = prefix_cache_coordinator

    @property
    def num_free_slots(self) -> int:
        return int(self.cache_manager.num_free_slots)

    def _step_required_slots(self, seqs: list[Sequence], is_prefill: bool) -> int:
        if is_prefill:
            return int(sum(int(seq.current_chunk_size or 0) for seq in seqs))
        return int(len(seqs))

    def _mixed_prefix_evictable_slots(self) -> int:
        if self.prefix_cache_coordinator is None:
            return 0
        return int(self.prefix_cache_coordinator.evictable_slots())

    def _mixed_prefix_boundary_limit(self, seq: Sequence) -> int:
        if self.prefix_cache_coordinator is None:
            return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        block_size = int(self.prefix_cache_coordinator.block_size)
        if block_size <= 0:
            return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        start = max(
            int(seq.num_prefilled_tokens),
            int(getattr(seq, "prefix_cache_hit_len", 0) or 0),
        )
        remaining = int(seq.num_prompt_tokens) - start
        if remaining <= 0:
            return 0
        to_boundary = block_size - (start % block_size)
        if to_boundary == 0:
            to_boundary = block_size
        return int(min(remaining, to_boundary))

    def _evict_mixed_prefix_for_step(self, seqs: list[Sequence], is_prefill: bool) -> None:
        if self.prefix_cache_coordinator is None:
            return
        needed = self._step_required_slots(seqs, is_prefill)
        if needed <= 0:
            return
        free = int(getattr(self.cache_manager, "num_free_slots"))
        if free < needed:
            self.prefix_cache_coordinator.evict_for_slots(needed - free)

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        if is_prefill and self.prefix_cache_coordinator is not None:
            self.prefix_cache_coordinator.attach_prefix_cache_hits(seqs)
        self._evict_mixed_prefix_for_step(seqs, is_prefill)
        result = self.cache_manager.prepare_step(seqs, is_prefill)
        if self.recurrent_state_manager is not None:
            self.recurrent_state_manager.prepare_step(seqs, is_prefill)
        return result

    def prepare_decode_static(self, seqs: list[Sequence], *args):
        self._evict_mixed_prefix_for_step(seqs, is_prefill=False)
        result = self.cache_manager.prepare_decode_static(seqs, *args)
        if self.recurrent_state_manager is not None:
            if not args or not hasattr(args[0], "shape") or not hasattr(args[0], "device"):
                raise RuntimeError("Static recurrent decode requires the graph input tensor.")
            self.recurrent_state_manager.prepare_decode_static(
                seqs,
                token_batch=int(args[0].shape[0]),
                device=args[0].device,
            )
        return result

    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool) -> None:
        self.cache_manager.on_forward_end(seqs, is_prefill)
        if self.recurrent_state_manager is not None:
            self.recurrent_state_manager.on_forward_end(seqs, is_prefill)
        if self.prefix_cache_coordinator is not None:
            self.prefix_cache_coordinator.record_step_tokens(seqs, is_prefill)
            self.prefix_cache_coordinator.commit_pending_blocks(seqs)

    def free_seq(self, seq_id: int) -> None:
        self.cache_manager.free_seq(seq_id)
        if self.prefix_cache_coordinator is not None:
            self.prefix_cache_coordinator.release_seq(seq_id)
        if self.recurrent_state_manager is not None:
            self.recurrent_state_manager.free_seq(seq_id)

    @torch.inference_mode()
    def reset_after_warmup(self) -> None:
        if self.prefix_cache_coordinator is not None:
            self.prefix_cache_coordinator.reset_after_warmup()
        reset_cache = getattr(self.cache_manager, "reset_after_warmup", None)
        if callable(reset_cache):
            reset_cache()
        else:
            reset_prefix_cache = getattr(self.cache_manager, "reset_prefix_cache", None)
            if callable(reset_prefix_cache):
                reset_prefix_cache()
        if self.recurrent_state_manager is not None:
            self.recurrent_state_manager.reset_after_warmup()

    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        if self.prefix_cache_coordinator is not None:
            self.prefix_cache_coordinator.refresh_prefix_cache_hit(seq)
            return
        self.cache_manager.refresh_prefix_cache_hit(seq)

    def clear_prefix_cache_hit(self, seq: Sequence) -> None:
        self.cache_manager.clear_prefix_cache_hit(seq)

    def prefill_step_free_slots(self) -> int:
        return int(self.cache_manager.prefill_step_free_slots() + self._mixed_prefix_evictable_slots())

    def prefill_batched_tokens_margin(self) -> int:
        return int(self.cache_manager.prefill_batched_tokens_margin())

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        return int(self.cache_manager.remaining_prefill_tokens(seq))

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], chunk_prefill_size: int) -> int:
        return int(self.cache_manager.reserved_prefill_slots(waiting_seqs, chunk_prefill_size))

    def should_schedule_full_prefill(self, seq: Sequence) -> bool:
        return bool(self.cache_manager.should_schedule_full_prefill(seq))

    def requires_full_prefill_step(self, seq: Sequence) -> bool:
        return bool(self.cache_manager.requires_full_prefill_step(seq))

    def requires_long_prefill_offload(self, seq: Sequence) -> bool:
        return bool(self.cache_manager.requires_long_prefill_offload(seq))

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        free_slots = int(self.cache_manager.prefill_step_free_slots_for(seq) + self._mixed_prefix_evictable_slots())
        if self.prefix_cache_coordinator is None:
            return free_slots
        return int(min(free_slots, self._mixed_prefix_boundary_limit(seq)))

    def decode_step_free_slots(self) -> int:
        return int(self.cache_manager.decode_step_free_slots() + self._mixed_prefix_evictable_slots())

    def decode_step_free_slots_for(self, seq: Sequence) -> int:
        return int(self.cache_manager.decode_step_free_slots_for(seq) + self._mixed_prefix_evictable_slots())

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        return int(self.cache_manager.prefill_step_reservation_cost(seq, scheduled_tokens))

    def decode_step_reservation_cost(self, seq: Sequence) -> int:
        return int(self.cache_manager.decode_step_reservation_cost(seq))

    def prompt_admission_free_slots(self) -> int:
        return int(self.cache_manager.prompt_admission_free_slots() + self._mixed_prefix_evictable_slots())

    def prompt_admission_budgets(self, waiting_seqs, chunk_prefill_size: int) -> dict[str, int]:
        budgets = dict(self.cache_manager.prompt_admission_budgets(waiting_seqs, chunk_prefill_size))
        extra = self._mixed_prefix_evictable_slots()
        if extra <= 0:
            return budgets
        if "slots" in budgets:
            budgets["slots"] = int(budgets["slots"]) + extra
        elif len(budgets) == 1:
            key = next(iter(budgets))
            budgets[key] = int(budgets[key]) + extra
        else:
            raise RuntimeError(
                "Mixed prefix admission accounting cannot add evictable slots to "
                f"multi-budget cache manager budgets={budgets}."
            )
        return budgets

    def prompt_admission_cost(self, seq: Sequence) -> int:
        cost = int(self.cache_manager.prompt_admission_cost(seq))
        if self.prefix_cache_coordinator is not None:
            cost += int(self.prefix_cache_coordinator.prefix_hit_evictable_slots(seq))
        return cost

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        cost = int(self.cache_manager.prompt_logical_reservation_cost(seq))
        if self.prefix_cache_coordinator is not None:
            cost += int(self.prefix_cache_coordinator.prefix_hit_evictable_slots(seq))
        return cost

    def prompt_admission_failure_action(self) -> str:
        return str(self.cache_manager.prompt_admission_failure_action())

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]) -> None:
        self.cache_manager.on_prompt_admitted(seq, costs)

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        costs = dict(self.cache_manager.prompt_admission_costs(seq))
        if self.prefix_cache_coordinator is None:
            return costs
        extra = int(self.prefix_cache_coordinator.prefix_hit_evictable_slots(seq))
        if extra <= 0:
            return costs
        if "slots" in costs:
            costs["slots"] = int(costs["slots"]) + extra
        elif len(costs) == 1:
            key = next(iter(costs))
            costs[key] = int(costs[key]) + extra
        else:
            raise RuntimeError(
                "Mixed prefix admission accounting cannot add hit-evictable slots to "
                f"multi-budget cache manager costs={costs}."
            )
        return costs

    def prefix_cache_inspect(self, token_ids: list[int], *, include_subtree: bool = False) -> dict[str, object]:
        if self.prefix_cache_coordinator is not None:
            return self.prefix_cache_coordinator.inspect(token_ids, include_subtree=include_subtree)
        return self.cache_manager.prefix_cache_inspect(token_ids, include_subtree=include_subtree)

    def prefix_cache_match(self, token_ids: list[int]) -> dict[str, object]:
        if self.prefix_cache_coordinator is not None:
            return self.prefix_cache_coordinator.match(token_ids)
        return self.cache_manager.prefix_cache_match(token_ids)

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        if self.prefix_cache_coordinator is not None:
            return self.prefix_cache_coordinator.delete_subtree(token_ids)
        return self.cache_manager.prefix_cache_delete_subtree(token_ids)

    def prefix_cache_set_eviction_priority(self, token_ids: list[int], *, priority: int) -> dict[str, object]:
        if self.prefix_cache_coordinator is not None:
            return self.prefix_cache_coordinator.set_eviction_priority(token_ids, priority=priority)
        return self.cache_manager.prefix_cache_set_eviction_priority(token_ids, priority=priority)

    def free_slot_stats(self) -> dict[str, int]:
        stats = self.cache_manager.free_slot_stats()
        if self.prefix_cache_coordinator is not None:
            stats.update(self.prefix_cache_coordinator.stats())
        return stats

    def debug_live_seq_slots(self) -> dict[int, int]:
        return dict(self.cache_manager.debug_live_seq_slots())
