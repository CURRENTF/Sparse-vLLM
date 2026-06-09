from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, field
from typing import Any

import torch


def usable_prefix_cache_tokens(prompt_len: int, block_size: int) -> int:
    """Return the largest cache-hit prefix that still leaves logits work."""
    prompt_len = int(prompt_len)
    block_size = int(block_size)
    if block_size <= 0:
        raise ValueError(f"prefix cache block_size must be > 0, got {block_size}.")
    if prompt_len <= 1:
        return 0
    return ((prompt_len - 1) // block_size) * block_size


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, set):
        return sorted(_jsonable(item) for item in value)
    return str(value)


def resolve_prefix_cache_block_size(config: Any) -> int:
    configured = getattr(config, "prefix_cache_block_size", None)
    if configured is not None and (isinstance(configured, bool) or not isinstance(configured, int)):
        raise ValueError(f"prefix_cache_block_size must be a positive integer, got {configured!r}.")
    method = str(getattr(config, "vllm_sparse_method", "") or "")
    if method == "quest":
        quest_chunk_size = int(getattr(config, "quest_chunk_size"))
        if configured is not None and configured != quest_chunk_size:
            raise ValueError(
                "prefix_cache_block_size must equal quest_chunk_size for quest prefix caching: "
                f"prefix_cache_block_size={configured}, quest_chunk_size={quest_chunk_size}."
            )
        return quest_chunk_size

    block_size = 16 if configured is None else configured
    if block_size <= 0:
        raise ValueError(f"prefix_cache_block_size must be > 0, got {block_size}.")
    return block_size


def build_prefix_cache_fingerprint(config: Any, block_size: int) -> bytes:
    hf_config = getattr(config, "hf_config", None)
    payload = {
        "model": getattr(config, "model", None),
        "model_type": getattr(hf_config, "model_type", None),
        "dtype": str(getattr(hf_config, "torch_dtype", None)),
        "tp_size": int(getattr(config, "tensor_parallel_size", 1)),
        "method": str(getattr(config, "vllm_sparse_method", "") or ""),
        "block_size": int(block_size),
        "salt": str(getattr(config, "prefix_cache_salt", "") or ""),
        "chunk_prefill_accel_omnikv": bool(getattr(config, "chunk_prefill_accel_omnikv", False)),
        "num_top_tokens": _jsonable(getattr(config, "num_top_tokens", None)),
        "num_top_tokens_in_prefill": _jsonable(getattr(config, "num_top_tokens_in_prefill", None)),
        "num_sink_tokens": _jsonable(getattr(config, "num_sink_tokens", None)),
        "num_recent_tokens": _jsonable(getattr(config, "num_recent_tokens", None)),
        "full_attn_layers": _jsonable(getattr(config, "full_attn_layers", None)),
        "obs_layer_ids": _jsonable(getattr(config, "obs_layer_ids", None)),
        "quest_chunk_size": _jsonable(getattr(config, "quest_chunk_size", None)),
        "quest_skip_layers": _jsonable(getattr(config, "quest_skip_layers", None)),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).digest()


def _pack_token_ids(token_ids: list[int] | tuple[int, ...]) -> bytes:
    return b"".join(struct.pack("<q", int(token_id)) for token_id in token_ids)


@dataclass
class PrefixCacheBlock:
    key: bytes
    parent_key: bytes | None
    block_size: int
    logical_block_idx: int
    slots: torch.Tensor | None = None
    page_slot: int | None = None
    page_slots: torch.Tensor | None = None
    token_ids: tuple[int, ...] = ()
    ref_count: int = 0
    last_access: int = 0
    child_keys: set[bytes] = field(default_factory=set)


class PrefixCacheIndex:
    def __init__(
        self,
        *,
        block_size: int,
        fingerprint: bytes,
        max_blocks: int | None = None,
    ):
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}.")
        if max_blocks is not None and int(max_blocks) <= 0:
            raise ValueError(f"max_blocks must be > 0 when set, got {max_blocks}.")
        self.fingerprint = bytes(fingerprint)
        self.max_blocks = None if max_blocks is None else int(max_blocks)
        self._blocks: dict[bytes, PrefixCacheBlock] = {}
        self._clock = 0

        self.lookup_requests = 0
        self.hit_requests = 0
        self.hit_tokens = 0
        self.hit_blocks = 0
        self.materialized_blocks = 0
        self.evicted_blocks = 0
        self.duplicate_blocks = 0

    def __len__(self) -> int:
        return len(self._blocks)

    def has_block(self, key: bytes) -> bool:
        return key in self._blocks

    def get_block(self, key: bytes) -> PrefixCacheBlock | None:
        return self._blocks.get(key)

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def hash_block(self, token_ids: list[int] | tuple[int, ...], parent_key: bytes | None) -> bytes:
        if len(token_ids) != self.block_size:
            raise ValueError(
                f"prefix cache blocks must be full: got {len(token_ids)} tokens, block_size={self.block_size}."
            )
        hasher = hashlib.sha256()
        hasher.update(self.fingerprint)
        if parent_key is None:
            hasher.update(b"\x00")
        else:
            hasher.update(b"\x01")
            hasher.update(parent_key)
        hasher.update(_pack_token_ids(token_ids))
        return hasher.digest()

    def lookup_longest_prefix(
        self,
        token_ids: list[int],
        *,
        max_usable_tokens: int,
    ) -> tuple[int, bytes | None, int]:
        self.lookup_requests += 1
        usable_tokens = min(int(max_usable_tokens), (len(token_ids) // self.block_size) * self.block_size)
        parent_key: bytes | None = None
        hit_blocks: list[PrefixCacheBlock] = []
        for start in range(0, usable_tokens, self.block_size):
            block_tokens = token_ids[start: start + self.block_size]
            key = self.hash_block(block_tokens, parent_key)
            block = self._blocks.get(key)
            if block is None:
                break
            hit_blocks.append(block)
            parent_key = key

        if hit_blocks:
            self.hit_requests += 1
            hit_len = len(hit_blocks) * self.block_size
            self.hit_tokens += hit_len
            self.hit_blocks += len(hit_blocks)
            return hit_len, hit_blocks[-1].key, len(hit_blocks)
        return 0, None, 0

    def get_chain(self, last_key: bytes, num_blocks: int) -> list[PrefixCacheBlock]:
        num_blocks = int(num_blocks)
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be > 0, got {num_blocks}.")
        chain: list[PrefixCacheBlock] = []
        key: bytes | None = last_key
        while key is not None and len(chain) < num_blocks:
            block = self._blocks.get(key)
            if block is None:
                short_key = key.hex()[:16]
                raise RuntimeError(
                    "Prefix cache chain is incomplete: "
                    f"missing_key={short_key} recovered_blocks={len(chain)} expected_blocks={num_blocks}."
                )
            chain.append(block)
            key = block.parent_key
        if len(chain) != num_blocks:
            raise RuntimeError(
                "Prefix cache chain is shorter than expected: "
                f"recovered_blocks={len(chain)} expected_blocks={num_blocks}."
            )
        chain.reverse()
        return chain

    def ensure_insert_capacity(self, needed_blocks: int = 1) -> list[PrefixCacheBlock]:
        if self.max_blocks is None:
            return []
        needed_blocks = int(needed_blocks)
        if needed_blocks <= 0:
            raise ValueError(f"needed_blocks must be > 0, got {needed_blocks}.")
        over_capacity = len(self._blocks) + needed_blocks - self.max_blocks
        if over_capacity <= 0:
            return []
        evicted = self.evict_until_freeable(over_capacity)
        if len(evicted) != over_capacity:
            raise RuntimeError(
                "Prefix cache capacity exceeded and not enough blocks are evictable: "
                f"live_blocks={len(self._blocks)} max_blocks={self.max_blocks} "
                f"needed_blocks={needed_blocks} evicted_blocks={len(evicted)} "
                f"evictable_blocks={self.evictable_blocks()}."
            )
        return evicted

    def insert_block(self, block: PrefixCacheBlock) -> PrefixCacheBlock:
        existing = self._blocks.get(block.key)
        if existing is not None:
            self.duplicate_blocks += 1
            existing.last_access = self._tick()
            return existing
        if self.max_blocks is not None and len(self._blocks) >= self.max_blocks:
            raise RuntimeError(
                "Prefix cache capacity exceeded before insert: "
                f"live_blocks={len(self._blocks)} max_blocks={self.max_blocks} "
                f"evictable_blocks={self.evictable_blocks()}."
            )
        if block.parent_key is not None:
            parent = self._blocks.get(block.parent_key)
            if parent is None:
                raise KeyError("Cannot insert prefix cache block because parent is missing.")
            parent.child_keys.add(block.key)
        block.last_access = self._tick()
        self._blocks[block.key] = block
        self.materialized_blocks += 1
        return block

    def touch_chain(self, blocks: list[PrefixCacheBlock]) -> None:
        access = self._tick()
        for block in blocks:
            block.last_access = access

    @staticmethod
    def can_evict(block: PrefixCacheBlock) -> bool:
        return int(block.ref_count) == 0 and not block.child_keys

    def evictable_blocks(self) -> int:
        return sum(1 for block in self._blocks.values() if self.can_evict(block))

    def evict_until_freeable(self, needed_blocks: int) -> list[PrefixCacheBlock]:
        evicted: list[PrefixCacheBlock] = []
        needed_blocks = int(needed_blocks)
        while len(evicted) < needed_blocks:
            candidates = [block for block in self._blocks.values() if self.can_evict(block)]
            if not candidates:
                break
            block = min(candidates, key=lambda candidate: candidate.last_access)
            self.remove(block.key)
            evicted.append(block)
        return evicted

    def remove(self, key: bytes) -> PrefixCacheBlock:
        block = self._blocks.get(key)
        if block is None:
            raise KeyError("Prefix cache block is not present.")
        if int(block.ref_count) != 0:
            raise RuntimeError("Cannot evict a referenced prefix cache block.")
        if block.child_keys:
            raise RuntimeError("Cannot evict a prefix cache block with live children.")
        del self._blocks[key]
        if block.parent_key is not None:
            parent = self._blocks.get(block.parent_key)
            if parent is not None:
                parent.child_keys.discard(key)
        block.child_keys.clear()
        self.evicted_blocks += 1
        return block

    def stats(self) -> dict[str, int]:
        pinned_blocks = sum(1 for block in self._blocks.values() if int(block.ref_count) > 0)
        return {
            "prefix_cache_lookup_requests": int(self.lookup_requests),
            "prefix_cache_hit_requests": int(self.hit_requests),
            "prefix_cache_hit_tokens": int(self.hit_tokens),
            "prefix_cache_hit_blocks": int(self.hit_blocks),
            "prefix_cache_materialized_blocks": int(self.materialized_blocks),
            "prefix_cache_evicted_blocks": int(self.evicted_blocks),
            "prefix_cache_live_blocks": int(len(self._blocks)),
            "prefix_cache_evictable_blocks": int(self.evictable_blocks()),
            "prefix_cache_pinned_blocks": int(pinned_blocks),
            "prefix_cache_duplicate_blocks": int(self.duplicate_blocks),
        }
