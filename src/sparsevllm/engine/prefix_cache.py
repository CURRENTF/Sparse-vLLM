from __future__ import annotations

import hashlib
import heapq
import json
import struct
from dataclasses import dataclass, field
from typing import Any, Protocol


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
        runtime_layout = getattr(config, "runtime_layout", None)
        is_mixed = bool(getattr(runtime_layout, "linear_attention_layer_indices", ()))
        if is_mixed:
            block_size = quest_chunk_size if configured is None else configured
            if block_size <= 0 or block_size % quest_chunk_size != 0:
                raise ValueError(
                    "mixed Quest prefix_cache_block_size must be a positive multiple of quest_chunk_size: "
                    f"prefix_cache_block_size={block_size}, quest_chunk_size={quest_chunk_size}."
                )
            return block_size
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


class PrefixBlockPayload(Protocol):
    """Marker protocol for method-owned prefix block payloads."""


@dataclass
class PrefixCacheBlock:
    stable_block_id: bytes
    parent_block_id: bytes | None
    block_size: int
    logical_block_idx: int
    payload: PrefixBlockPayload
    token_ids: tuple[int, ...] = ()
    ref_count: int = 0
    last_access: int = 0
    eviction_priority: int = 0


@dataclass(frozen=True)
class RadixLookupResult:
    hit_block_count: int
    last_block_id: bytes | None


@dataclass
class RadixTreeNode:
    segment: tuple[bytes, ...] = ()
    parent: RadixTreeNode | None = None
    children: dict[bytes, RadixTreeNode] = field(default_factory=dict)


class RadixTreeBackend:
    """Block-level radix backend.

    Edges store one or more stable block ids, and splits only occur between
    block ids. Every block remains directly addressable through the location
    map so cache managers can attach, inspect, delete, and evict by block id.
    """

    def __init__(self):
        self.root = RadixTreeNode()
        self._locations: dict[bytes, tuple[RadixTreeNode, int]] = {}
        self._leaf_block_ids: set[bytes] = set()

    def _index_segment(self, node: RadixTreeNode) -> None:
        for index, block_id in enumerate(node.segment):
            self._locations[block_id] = (node, index)

    def _unindex_segment(self, segment: tuple[bytes, ...]) -> None:
        for block_id in segment:
            self._locations.pop(block_id, None)

    def _discard_leaf(self, node: RadixTreeNode) -> None:
        if node.segment:
            self._leaf_block_ids.discard(node.segment[-1])

    def _add_leaf(self, node: RadixTreeNode) -> None:
        if node.segment and not node.children:
            self._leaf_block_ids.add(node.segment[-1])

    def _split_after(self, node: RadixTreeNode, index: int) -> RadixTreeNode:
        old_segment = node.segment
        prefix = old_segment[: index + 1]
        suffix = old_segment[index + 1:]
        if not suffix:
            return node
        old_children = node.children
        self._unindex_segment(old_segment)

        suffix_node = RadixTreeNode(segment=suffix, parent=node, children=old_children)
        for child in suffix_node.children.values():
            child.parent = suffix_node
        node.segment = prefix
        node.children = {suffix[0]: suffix_node}
        self._index_segment(node)
        self._index_segment(suffix_node)
        return node

    def insert_child(self, parent_block_id: bytes | None, block_id: bytes) -> None:
        if block_id in self._locations:
            return
        if parent_block_id is None:
            if block_id in self.root.children:
                raise RuntimeError("Radix tree root child exists without a block location.")
            child = RadixTreeNode(segment=(block_id,), parent=self.root)
            self.root.children[block_id] = child
            self._index_segment(child)
            self._add_leaf(child)
            return

        location = self._locations.get(parent_block_id)
        if location is None:
            raise KeyError("Prefix cache parent block id is not present in radix tree.")
        parent_node, index = location
        parent_node = self._split_after(parent_node, index)
        if not parent_node.children:
            self._discard_leaf(parent_node)
        if block_id in parent_node.children:
            raise RuntimeError("Radix tree child exists without a block location.")
        child = RadixTreeNode(segment=(block_id,), parent=parent_node)
        parent_node.children[block_id] = child
        self._index_segment(child)
        self._add_leaf(child)

    def lookup(self, block_ids: list[bytes] | tuple[bytes, ...], max_blocks: int) -> RadixLookupResult:
        node = self.root
        hit_count = 0
        last_block_id: bytes | None = None
        limit = min(int(max_blocks), len(block_ids))
        while hit_count < limit:
            child = node.children.get(block_ids[hit_count])
            if child is None:
                break
            for segment_block_id in child.segment:
                if hit_count >= limit or block_ids[hit_count] != segment_block_id:
                    return RadixLookupResult(hit_block_count=hit_count, last_block_id=last_block_id)
                hit_count += 1
                last_block_id = segment_block_id
            node = child
        return RadixLookupResult(hit_block_count=hit_count, last_block_id=last_block_id)

    def insert(self, block_ids: list[bytes] | tuple[bytes, ...]) -> None:
        block_ids = tuple(block_ids)
        if not block_ids:
            return
        node = self.root
        offset = 0
        while offset < len(block_ids):
            child = node.children.get(block_ids[offset])
            if child is None:
                if not node.children:
                    self._discard_leaf(node)
                child = RadixTreeNode(segment=block_ids[offset:], parent=node)
                node.children[block_ids[offset]] = child
                self._index_segment(child)
                self._add_leaf(child)
                return

            common = 0
            while (
                offset + common < len(block_ids)
                and common < len(child.segment)
                and block_ids[offset + common] == child.segment[common]
            ):
                common += 1

            if common == len(child.segment):
                node = child
                offset += common
                continue

            if common <= 0:
                raise RuntimeError("Radix tree child map is inconsistent with edge segment.")

            old_segment = child.segment
            self._unindex_segment(old_segment)
            prefix = child.segment[:common]
            suffix = child.segment[common:]
            split = RadixTreeNode(segment=prefix, parent=node)
            node.children[prefix[0]] = split
            child.segment = suffix
            child.parent = split
            split.children[suffix[0]] = child

            offset += common
            if offset < len(block_ids):
                new_segment = block_ids[offset:]
                new_child = RadixTreeNode(segment=new_segment, parent=split)
                split.children[new_segment[0]] = new_child
                self._index_segment(new_child)
                self._add_leaf(new_child)
            self._index_segment(split)
            self._index_segment(child)
            return

        return

    def remove_block(self, block_id: bytes) -> None:
        location = self._locations.get(block_id)
        if location is None:
            raise KeyError("Prefix cache block id is not present in radix tree.")
        node, index = location
        if index != len(node.segment) - 1 or node.children:
            raise RuntimeError("Cannot remove a prefix cache tree block with live children.")
        if node.parent is None:
            raise RuntimeError("Cannot remove radix tree root.")
        self._locations.pop(block_id, None)
        self._leaf_block_ids.discard(block_id)
        if len(node.segment) == 1:
            parent = node.parent
            del parent.children[node.segment[0]]
            self._add_leaf(parent)
        else:
            node.segment = node.segment[:-1]
            self._add_leaf(node)

    def path_to_block(self, block_id: bytes) -> tuple[bytes, ...]:
        location = self._locations.get(block_id)
        if location is None:
            raise KeyError("Prefix cache block id is not present in radix tree.")
        node, index = location
        segments: list[tuple[bytes, ...]] = [node.segment[: index + 1]]
        while node.parent is not None:
            node = node.parent
            if node.segment:
                segments.append(node.segment)
        return tuple(block_id for segment in reversed(segments) for block_id in segment)

    def subtree_block_ids(self, root_block_id: bytes) -> tuple[bytes, ...]:
        location = self._locations.get(root_block_id)
        if location is None:
            raise KeyError("Prefix cache subtree root is not present.")
        root, index = location
        result: list[bytes] = []
        result.extend(root.segment[index:])
        stack = list(root.children.values())
        while stack:
            node = stack.pop()
            result.extend(node.segment)
            stack.extend(node.children.values())
        return tuple(result)

    def child_count(self, block_id: bytes) -> int:
        location = self._locations.get(block_id)
        if location is None:
            raise KeyError("Prefix cache block id is not present in radix tree.")
        node, index = location
        if index < len(node.segment) - 1:
            return 1
        return len(node.children)

    def leaf_block_ids(self) -> tuple[bytes, ...]:
        return tuple(self._leaf_block_ids)

    def stats(self) -> dict[str, int]:
        node_count = 0
        edge_count = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            node_count += 1
            edge_count += len(node.children)
            stack.extend(node.children.values())
        return {
            "prefix_cache_tree_nodes": int(node_count),
            "prefix_cache_tree_edges": int(edge_count),
        }

    def _rebuild_locations(self) -> None:
        locations: dict[bytes, tuple[RadixTreeNode, int]] = {}
        leaf_block_ids: set[bytes] = set()
        stack = list(self.root.children.values())
        while stack:
            node = stack.pop()
            for index, block_id in enumerate(node.segment):
                locations[block_id] = (node, index)
            if node.segment and not node.children:
                leaf_block_ids.add(node.segment[-1])
            stack.extend(node.children.values())
        self._locations = locations
        self._leaf_block_ids = leaf_block_ids


@dataclass(frozen=True)
class PrefixCacheBlockedBlock:
    block_id: bytes | None
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "block_id": None if self.block_id is None else self.block_id.hex(),
            "reason": self.reason,
        }


@dataclass
class PrefixCacheDeleteResult:
    deleted_blocks: list[PrefixCacheBlock]
    blocked_blocks: list[PrefixCacheBlockedBlock]

    def to_dict(self) -> dict[str, Any]:
        return {
            "deleted_block_ids": [block.stable_block_id.hex() for block in self.deleted_blocks],
            "deleted_block_count": len(self.deleted_blocks),
            "blocked_blocks": [block.to_dict() for block in self.blocked_blocks],
        }


class RadixPrefixIndex:
    def __init__(
        self,
        *,
        block_size: int,
        fingerprint: bytes,
        max_blocks: int | None = None,
        backend: RadixTreeBackend | None = None,
    ):
        self.block_size = int(block_size)
        if self.block_size <= 0:
            raise ValueError(f"block_size must be > 0, got {block_size}.")
        if max_blocks is not None and int(max_blocks) <= 0:
            raise ValueError(f"max_blocks must be > 0 when set, got {max_blocks}.")
        self.fingerprint = bytes(fingerprint)
        self.max_blocks = None if max_blocks is None else int(max_blocks)
        self.backend = backend or RadixTreeBackend()
        self.blocks: dict[bytes, PrefixCacheBlock] = {}
        self._clock = 0

        self.lookup_requests = 0
        self.hit_requests = 0
        self.hit_tokens = 0
        self.hit_blocks = 0
        self.committed_blocks = 0
        self.evicted_blocks = 0
        self.deleted_blocks = 0
        self.duplicate_commits = 0
        self.control_inspect_requests = 0
        self.control_delete_requests = 0
        self.control_priority_updates = 0

    def __len__(self) -> int:
        return len(self.blocks)

    def has_block(self, stable_block_id: bytes) -> bool:
        return stable_block_id in self.blocks

    def get_block(self, stable_block_id: bytes) -> PrefixCacheBlock | None:
        return self.blocks.get(stable_block_id)

    def _tick(self) -> int:
        self._clock += 1
        return self._clock

    def stable_block_id(
        self,
        token_ids: list[int] | tuple[int, ...],
        parent_block_id: bytes | None,
    ) -> bytes:
        if len(token_ids) != self.block_size:
            raise ValueError(
                f"prefix cache blocks must be full: got {len(token_ids)} tokens, block_size={self.block_size}."
            )
        hasher = hashlib.sha256()
        hasher.update(self.fingerprint)
        if parent_block_id is None:
            hasher.update(b"\x00")
        else:
            hasher.update(b"\x01")
            hasher.update(parent_block_id)
        hasher.update(_pack_token_ids(token_ids))
        return hasher.digest()

    def block_ids_for_tokens(
        self,
        token_ids: list[int],
        *,
        max_tokens: int | None = None,
    ) -> list[bytes]:
        token_limit = len(token_ids) if max_tokens is None else min(int(max_tokens), len(token_ids))
        token_limit = (token_limit // self.block_size) * self.block_size
        parent_block_id: bytes | None = None
        block_ids: list[bytes] = []
        for start in range(0, token_limit, self.block_size):
            block_tokens = token_ids[start: start + self.block_size]
            block_id = self.stable_block_id(block_tokens, parent_block_id)
            block_ids.append(block_id)
            parent_block_id = block_id
        return block_ids

    def match_longest_prefix(
        self,
        token_ids: list[int],
        *,
        max_usable_tokens: int,
    ) -> tuple[int, bytes | None, int]:
        block_ids = self.block_ids_for_tokens(token_ids, max_tokens=max_usable_tokens)
        result = self.backend.lookup(block_ids, len(block_ids))
        if result.hit_block_count <= 0:
            return 0, None, 0
        hit_len = result.hit_block_count * self.block_size
        return hit_len, result.last_block_id, result.hit_block_count

    def lookup_longest_prefix(
        self,
        token_ids: list[int],
        *,
        max_usable_tokens: int,
    ) -> tuple[int, bytes | None, int]:
        self.lookup_requests += 1
        hit_len, last_block_id, hit_blocks = self.match_longest_prefix(
            token_ids,
            max_usable_tokens=max_usable_tokens,
        )
        if hit_blocks <= 0:
            return 0, None, 0
        self.hit_requests += 1
        self.hit_tokens += hit_len
        self.hit_blocks += hit_blocks
        return hit_len, last_block_id, hit_blocks

    def get_chain(self, last_block_id: bytes, block_count: int) -> list[PrefixCacheBlock]:
        block_count = int(block_count)
        if block_count <= 0:
            raise ValueError(f"block_count must be > 0, got {block_count}.")
        path = self.backend.path_to_block(last_block_id)
        if len(path) < block_count:
            raise RuntimeError(
                "Prefix cache chain is shorter than expected: "
                f"recovered_blocks={len(path)} expected_blocks={block_count}."
            )
        chain_ids = path[-block_count:]
        chain: list[PrefixCacheBlock] = []
        for block_id in chain_ids:
            block = self.blocks.get(block_id)
            if block is None:
                short_key = block_id.hex()[:16]
                raise RuntimeError(
                    "Prefix cache chain is incomplete: "
                    f"missing_block_id={short_key} recovered_blocks={len(chain)} expected_blocks={block_count}."
                )
            chain.append(block)
        return chain

    def ensure_insert_capacity(self, needed_blocks: int = 1) -> list[PrefixCacheBlock]:
        if self.max_blocks is None:
            return []
        needed_blocks = int(needed_blocks)
        if needed_blocks <= 0:
            raise ValueError(f"needed_blocks must be > 0, got {needed_blocks}.")
        over_capacity = len(self.blocks) + needed_blocks - self.max_blocks
        if over_capacity <= 0:
            return []
        evicted = self.evict_until_freeable(over_capacity)
        if len(evicted) != over_capacity:
            raise RuntimeError(
                "Prefix cache capacity exceeded and not enough blocks are evictable: "
                f"live_blocks={len(self.blocks)} max_blocks={self.max_blocks} "
                f"needed_blocks={needed_blocks} evicted_blocks={len(evicted)} "
                f"evictable_blocks={self.evictable_blocks()}."
            )
        return evicted

    def insert_block(self, block: PrefixCacheBlock) -> PrefixCacheBlock:
        existing = self.blocks.get(block.stable_block_id)
        if existing is not None:
            self.duplicate_commits += 1
            existing.last_access = self._tick()
            return existing
        if self.max_blocks is not None and len(self.blocks) >= self.max_blocks:
            raise RuntimeError(
                "Prefix cache capacity exceeded before insert: "
                f"live_blocks={len(self.blocks)} max_blocks={self.max_blocks} "
                f"evictable_blocks={self.evictable_blocks()}."
            )
        if block.parent_block_id is not None:
            if block.parent_block_id not in self.blocks:
                raise KeyError("Cannot insert prefix cache block because parent is missing.")
        block.last_access = self._tick()
        self.blocks[block.stable_block_id] = block
        self.backend.insert_child(block.parent_block_id, block.stable_block_id)
        self.committed_blocks += 1
        return block

    def touch_chain(self, blocks: list[PrefixCacheBlock]) -> None:
        access = self._tick()
        for block in blocks:
            block.last_access = access

    def child_count(self, stable_block_id: bytes) -> int:
        return self.backend.child_count(stable_block_id)

    def can_evict(self, block: PrefixCacheBlock) -> bool:
        if int(block.ref_count) != 0 or int(block.eviction_priority) < 0:
            return False
        return self.child_count(block.stable_block_id) == 0

    def evictable_blocks(self) -> int:
        return sum(
            1
            for block_id in self.backend.leaf_block_ids()
            if (block := self.blocks.get(block_id)) is not None
            and int(block.ref_count) == 0
            and int(block.eviction_priority) >= 0
        )

    def freeable_block_ids(self) -> set[bytes]:
        """Return blocks removable by repeated leaf eviction without mutating the tree."""
        freeable: set[bytes] = set()
        subtree_freeable: dict[int, bool] = {}
        stack: list[tuple[RadixTreeNode, bool]] = [(self.backend.root, False)]

        while stack:
            node, visited = stack.pop()
            node_key = id(node)
            if not visited:
                stack.append((node, True))
                for child in node.children.values():
                    stack.append((child, False))
                continue

            children_freeable = True
            for child in node.children.values():
                if not subtree_freeable.pop(id(child), False):
                    children_freeable = False

            suffix_freeable = children_freeable
            for block_id in reversed(node.segment):
                block = self.blocks.get(block_id)
                locally_freeable = (
                    block is not None
                    and int(block.ref_count) == 0
                    and int(block.eviction_priority) >= 0
                )
                if locally_freeable and suffix_freeable:
                    freeable.add(block_id)
                    suffix_freeable = True
                else:
                    suffix_freeable = False

            subtree_freeable[node_key] = suffix_freeable

        return freeable

    def freeable_blocks(self) -> int:
        return len(self.freeable_block_ids())

    def _remove_block_from_index(self, stable_block_id: bytes) -> PrefixCacheBlock:
        block = self.blocks.get(stable_block_id)
        if block is None:
            raise KeyError("Prefix cache block is not present.")
        if int(block.ref_count) != 0:
            raise RuntimeError("Cannot remove a referenced prefix cache block.")
        if int(block.eviction_priority) < 0:
            raise RuntimeError("Cannot remove a negative-priority prefix cache block.")
        if self.child_count(stable_block_id) != 0:
            raise RuntimeError("Cannot remove a prefix cache block with live children.")
        self.backend.remove_block(stable_block_id)
        del self.blocks[stable_block_id]
        return block

    def evict_until_freeable(self, needed_blocks: int) -> list[PrefixCacheBlock]:
        evicted: list[PrefixCacheBlock] = []
        needed_blocks = int(needed_blocks)
        candidate_heap: list[tuple[int, int, bytes]] = []
        queued: set[bytes] = set()

        def queue_if_evictable(block_id: bytes | None) -> None:
            if block_id is None or block_id in queued:
                return
            block = self.blocks.get(block_id)
            if block is None or not self.can_evict(block):
                return
            heapq.heappush(
                candidate_heap,
                (-int(block.eviction_priority), int(block.last_access), block_id),
            )
            queued.add(block_id)

        for block_id in self.backend.leaf_block_ids():
            queue_if_evictable(block_id)

        while len(evicted) < needed_blocks:
            while candidate_heap:
                _, _, block_id = heapq.heappop(candidate_heap)
                queued.discard(block_id)
                block = self.blocks.get(block_id)
                if block is not None and self.can_evict(block):
                    break
            else:
                break
            parent_block_id = block.parent_block_id
            evicted.append(self._remove_block_from_index(block.stable_block_id))
            self.evicted_blocks += 1
            queue_if_evictable(parent_block_id)
        return evicted

    def inspect_prefix(
        self,
        token_ids: list[int],
        *,
        include_subtree: bool = False,
    ) -> dict[str, Any]:
        self.control_inspect_requests += 1
        block_ids = self.block_ids_for_tokens(token_ids)
        if not block_ids:
            return {
                "matched": False,
                "reason": "prefix_shorter_than_block_size",
                "selector_block_count": 0,
                "hit_block_count": 0,
                "hit_len": 0,
                "last_block_id": None,
                "path_blocks": [],
            }
        result = self.backend.lookup(block_ids, len(block_ids))
        path_ids = block_ids[: result.hit_block_count]
        path_blocks = [self._block_status_dict(block_id) for block_id in path_ids if block_id in self.blocks]
        response: dict[str, Any] = {
            "matched": result.hit_block_count > 0,
            "selector_block_count": len(block_ids),
            "hit_block_count": int(result.hit_block_count),
            "hit_len": int(result.hit_block_count * self.block_size),
            "last_block_id": None if result.last_block_id is None else result.last_block_id.hex(),
            "path_blocks": path_blocks,
        }
        if include_subtree and result.last_block_id is not None:
            subtree_ids = self.backend.subtree_block_ids(result.last_block_id)
            response["subtree"] = self._subtree_summary(subtree_ids)
        return response

    def _block_status_dict(self, block_id: bytes) -> dict[str, Any]:
        block = self.blocks[block_id]
        return {
            "block_id": block_id.hex(),
            "logical_block_idx": int(block.logical_block_idx),
            "ref_count": int(block.ref_count),
            "eviction_priority": int(block.eviction_priority),
            "child_count": int(self.child_count(block_id)),
            "last_access": int(block.last_access),
        }

    def _subtree_summary(self, block_ids: tuple[bytes, ...]) -> dict[str, int]:
        existing = [self.blocks[block_id] for block_id in block_ids if block_id in self.blocks]
        return {
            "block_count": int(len(existing)),
            "referenced_block_count": int(sum(1 for block in existing if int(block.ref_count) > 0)),
            "negative_priority_block_count": int(sum(1 for block in existing if int(block.eviction_priority) < 0)),
            "leaf_block_count": int(sum(1 for block in existing if self.child_count(block.stable_block_id) == 0)),
        }

    def _exact_subtree_ids_for_tokens(self, token_ids: list[int]) -> tuple[bytes, ...] | None:
        block_ids = self.block_ids_for_tokens(token_ids)
        if not block_ids:
            return None
        result = self.backend.lookup(block_ids, len(block_ids))
        if result.hit_block_count != len(block_ids) or result.last_block_id is None:
            return None
        return self.backend.subtree_block_ids(result.last_block_id)

    def safe_delete_subtree(self, token_ids: list[int]) -> PrefixCacheDeleteResult:
        self.control_delete_requests += 1
        subtree_ids = self._exact_subtree_ids_for_tokens(token_ids)
        if subtree_ids is None:
            return PrefixCacheDeleteResult(
                deleted_blocks=[],
                blocked_blocks=[PrefixCacheBlockedBlock(block_id=None, reason="not_found")],
            )
        sorted_ids = sorted(
            subtree_ids,
            key=lambda block_id: len(self.backend.path_to_block(block_id)),
            reverse=True,
        )
        deleted: list[PrefixCacheBlock] = []
        blocked: list[PrefixCacheBlockedBlock] = []
        for block_id in sorted_ids:
            block = self.blocks.get(block_id)
            if block is None:
                continue
            reason = None
            if int(block.ref_count) > 0:
                reason = "referenced"
            elif int(block.eviction_priority) < 0:
                reason = "negative_priority"
            elif self.child_count(block_id) > 0:
                reason = "has_children"
            if reason is not None:
                blocked.append(PrefixCacheBlockedBlock(block_id=block_id, reason=reason))
                continue
            deleted.append(self._remove_block_from_index(block_id))
            self.deleted_blocks += 1
        return PrefixCacheDeleteResult(deleted_blocks=deleted, blocked_blocks=blocked)

    def set_subtree_eviction_priority(self, token_ids: list[int], priority: int) -> dict[str, Any]:
        self.control_priority_updates += 1
        block_ids = self.block_ids_for_tokens(token_ids)
        if not block_ids:
            return {
                "matched": False,
                "reason": "prefix_shorter_than_block_size",
                "root_block_id": None,
                "updated_block_count": 0,
                "eviction_priority": int(priority),
            }
        result = self.backend.lookup(block_ids, len(block_ids))
        if result.hit_block_count != len(block_ids) or result.last_block_id is None:
            return {
                "matched": False,
                "reason": "not_found",
                "root_block_id": None,
                "updated_block_count": 0,
                "eviction_priority": int(priority),
            }
        subtree_ids = self.backend.subtree_block_ids(result.last_block_id)
        for block_id in subtree_ids:
            block = self.blocks.get(block_id)
            if block is not None:
                block.eviction_priority = int(priority)
        return {
            "matched": True,
            "root_block_id": result.last_block_id.hex(),
            "updated_block_count": int(sum(1 for block_id in subtree_ids if block_id in self.blocks)),
            "eviction_priority": int(priority),
        }

    def stats(self) -> dict[str, int]:
        tree_stats = self.backend.stats()
        stats = {
            "prefix_cache_lookup_requests": int(self.lookup_requests),
            "prefix_cache_hit_requests": int(self.hit_requests),
            "prefix_cache_hit_tokens": int(self.hit_tokens),
            "prefix_cache_hit_blocks": int(self.hit_blocks),
            "prefix_cache_committed_blocks": int(self.committed_blocks),
            "prefix_cache_duplicate_commits": int(self.duplicate_commits),
            "prefix_cache_deleted_blocks": int(self.deleted_blocks),
            "prefix_cache_evicted_blocks": int(self.evicted_blocks),
            "prefix_cache_live_blocks": int(len(self.blocks)),
            "prefix_cache_referenced_blocks": int(sum(1 for block in self.blocks.values() if int(block.ref_count) > 0)),
            "prefix_cache_negative_priority_blocks": int(
                sum(1 for block in self.blocks.values() if int(block.eviction_priority) < 0)
            ),
            "prefix_cache_leaf_blocks": int(len(self.backend.leaf_block_ids())),
            "prefix_cache_freeable_blocks": int(self.freeable_blocks()),
            "prefix_cache_control_inspect_requests": int(self.control_inspect_requests),
            "prefix_cache_control_delete_requests": int(self.control_delete_requests),
            "prefix_cache_control_priority_updates": int(self.control_priority_updates),
        }
        stats.update(tree_stats)
        return stats
