from __future__ import annotations

from dataclasses import dataclass

import torch

from sparsevllm.engine.prefix_cache import (
    PrefixBlockPayload,
    PrefixCacheBlock,
    RadixPrefixIndex,
    build_prefix_cache_fingerprint,
)
from sparsevllm.engine.sequence import Sequence
from sparsevllm.utils.profiler import profiler


@dataclass
class PrefixRuntimeState:
    parent_block_id: bytes | None
    next_logical_block_idx: int
    pending_tokens: list[int]
    pending_slots: list[torch.Tensor]


@dataclass
class PendingPrefixBlock:
    stable_block_id: bytes
    parent_block_id: bytes | None
    logical_block_idx: int
    payload: PrefixBlockPayload
    slots: torch.Tensor
    token_ids: list[int]


class PrefixCacheMixin:
    """Shared prefix-cache block materialization for cache managers."""

    def _init_prefix_cache_runtime(self) -> None:
        self.seq_id_to_materialized_blocks: dict[int, list[PrefixCacheBlock]] = {}
        self.prefix_runtime_states: dict[int, PrefixRuntimeState] = {}
        self.pending_prefix_blocks: dict[int, list[PendingPrefixBlock]] = {}

    def _prefix_cache_materialization_subject(self) -> str:
        return "Prefix materialization"

    def _prefix_cache_negative_refcount_message(self) -> str:
        return "Prefix cache block ref_count became negative."

    def _prefix_cache_materialize_profile_name(self) -> str:
        return "prefix_cache_materialize"

    def _make_prefix_block_payload(self, slots: torch.Tensor) -> PrefixBlockPayload:
        raise NotImplementedError

    def _mark_materialized_prefix_block(self, seq: Sequence, block: PrefixCacheBlock) -> None:
        raise NotImplementedError

    def _reset_prefix_cache_allocator_after_clear(self) -> None:
        raise NotImplementedError

    def _release_prefix_blocks(self, blocks: list[PrefixCacheBlock]) -> None:
        for block in blocks:
            block.ref_count -= 1
            if block.ref_count < 0:
                raise RuntimeError(self._prefix_cache_negative_refcount_message())

    def reset_prefix_cache(self) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        referenced = [
            block.stable_block_id.hex()
            for block in self.prefix_cache.blocks.values()
            if int(block.ref_count) != 0
        ]
        if referenced:
            raise RuntimeError(
                "Cannot reset prefix cache while blocks are still referenced: "
                f"referenced_blocks={referenced[:5]}."
            )

        self._reset_prefix_cache_allocator_after_clear()
        self.prefix_cache = RadixPrefixIndex(
            block_size=int(self.prefix_cache_block_size),
            fingerprint=build_prefix_cache_fingerprint(self.config, int(self.prefix_cache_block_size)),
            max_blocks=getattr(self.config, "prefix_cache_max_blocks", None),
        )
        self.seq_id_to_prefix_blocks.clear()
        self._init_prefix_cache_runtime()

    def _record_prefix_materialization(
        self,
        seq: Sequence,
        token_ids: list[int],
        slots: torch.Tensor,
    ) -> None:
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        if len(token_ids) != int(slots.numel()):
            raise RuntimeError(
                f"{self._prefix_cache_materialization_subject()} token/slot mismatch: "
                f"seq_id={seq.seq_id} tokens={len(token_ids)} slots={int(slots.numel())}."
            )

        state = self.prefix_runtime_states.get(seq.seq_id)
        if state is None:
            hit_blocks = int(getattr(seq, "prefix_cache_hit_block_count", 0) or 0)
            parent_block_id = getattr(seq, "prefix_cache_hit_last_block_id", None)
            state = PrefixRuntimeState(
                parent_block_id=parent_block_id,
                next_logical_block_idx=hit_blocks,
                pending_tokens=[],
                pending_slots=[],
            )
            self.prefix_runtime_states[seq.seq_id] = state

        pending_blocks = self.pending_prefix_blocks.setdefault(seq.seq_id, [])
        block_size = int(self.prefix_cache_block_size)
        for token_id, slot in zip(token_ids, slots):
            state.pending_tokens.append(int(token_id))
            state.pending_slots.append(slot.detach().clone().reshape(()))
            if len(state.pending_tokens) != block_size:
                continue

            block_tokens = list(state.pending_tokens)
            block_slots = torch.stack(state.pending_slots).to(dtype=torch.int32)
            stable_block_id = self.prefix_cache.stable_block_id(block_tokens, state.parent_block_id)
            pending_blocks.append(
                PendingPrefixBlock(
                    stable_block_id=stable_block_id,
                    parent_block_id=state.parent_block_id,
                    logical_block_idx=state.next_logical_block_idx,
                    payload=self._make_prefix_block_payload(block_slots),
                    slots=block_slots,
                    token_ids=block_tokens,
                )
            )
            state.parent_block_id = stable_block_id
            state.next_logical_block_idx += 1
            state.pending_tokens = []
            state.pending_slots = []

    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool):
        del is_prefill
        if not self.enable_prefix_caching or self.prefix_cache is None:
            return
        with profiler.record(self._prefix_cache_materialize_profile_name()):
            for seq in seqs:
                pending_blocks = self.pending_prefix_blocks.pop(seq.seq_id, [])
                if not pending_blocks:
                    continue
                materialized = self.seq_id_to_materialized_blocks.setdefault(seq.seq_id, [])
                protected: list[PrefixCacheBlock] = []
                protected_block_ids = {
                    block_id
                    for pending in pending_blocks
                    for block_id in (pending.parent_block_id, pending.stable_block_id)
                    if block_id is not None and self.prefix_cache.has_block(block_id)
                }
                for block_id in protected_block_ids:
                    block = self.prefix_cache.get_block(block_id)
                    if block is None:
                        continue
                    block.ref_count += 1
                    protected.append(block)
                try:
                    for pending in pending_blocks:
                        if not self.prefix_cache.has_block(pending.stable_block_id):
                            self._evict_prefix_cache_for_insert(1)
                        block = PrefixCacheBlock(
                            stable_block_id=pending.stable_block_id,
                            parent_block_id=pending.parent_block_id,
                            block_size=int(self.prefix_cache_block_size),
                            logical_block_idx=pending.logical_block_idx,
                            payload=pending.payload,
                            token_ids=tuple(pending.token_ids),
                        )
                        inserted = self.prefix_cache.insert_block(block)
                        if inserted is not block:
                            continue
                        inserted.ref_count = 1
                        materialized.append(inserted)
                        self._mark_materialized_prefix_block(seq, inserted)
                finally:
                    self._release_prefix_blocks(protected)
