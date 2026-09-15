"""Radix lifecycle for cache-owned native window/carry snapshots."""

from dataclasses import dataclass

from sparsevllm.engine.prefix_cache import PrefixCacheBlock, RadixPrefixIndex, usable_prefix_cache_tokens


@dataclass
class NativePrefixPlan:
    entries: tuple
    parents: tuple[PrefixCacheBlock, ...]

    @property
    def snapshot_ends(self):
        result = {}
        for seq_id, end, *_ in self.entries:
            result.setdefault(seq_id, []).append(end)
        return result


class NativePrefixCache:
    def __init__(self, pool, *, block_size, fingerprint):
        self.pool = pool
        self.index = RadixPrefixIndex(block_size=block_size, fingerprint=fingerprint,
                                      max_blocks=len(pool.free_snapshot_rows))
        self.paths = {}
        self.hits = {}

    def _path(self, seq, end):
        path = self.paths.setdefault(seq.seq_id, [])
        block_size = self.index.block_size
        for block in range(len(path), end // block_size):
            start = block * block_size
            path.append(self.index.stable_block_id(seq.token_ids[start:start + block_size], path[-1] if path else None))
        return path

    def clear_hit(self, seq):
        self._release_hit(seq.seq_id)
        seq.clear_prefix_cache_hit()

    def _release_hit(self, seq_id):
        if block := self.hits.pop(seq_id, None):
            self.index.release_block_ref(block)

    def refresh_hit(self, seq):
        self.clear_hit(seq)
        usable = usable_prefix_cache_tokens(seq.num_prompt_tokens, self.index.block_size)
        length, last, count = self.index.lookup_longest_block_ids(self._path(seq, usable)[:usable // self.index.block_size])
        seq.prefix_cache_enabled = True
        seq.prefix_cache_block_size = self.index.block_size
        seq.prefix_cache_method = "deepseek_v4"
        seq.prefix_cache_hit_len = length
        seq.prefix_cache_hit_block_count = count
        seq.prefix_cache_hit_last_block_id = last
        if last is not None:
            block = self.index.get_block(last)
            self.index.acquire_block_ref(block)
            self.hits[seq.seq_id] = block

    def attach_hit(self, seq):
        if not seq.prefix_cache_hit_len:
            return
        block = self.hits.get(seq.seq_id)
        if block is None or block.stable_block_id != seq.prefix_cache_hit_last_block_id:
            raise RuntimeError("Native prefix attachment requires the pinned scheduler lookup.")
        self.pool.attach(seq.seq_id, block.payload)
        self._release_hit(seq.seq_id)

    def evict_for_capacity(self, compressed_costs, *, snapshot_rows=0):
        def fits():
            return (len(self.pool.free_snapshot_rows) >= snapshot_rows
                    and all(self.pool.slots[ratio].free_count >= need for ratio, need in compressed_costs.items()))

        if fits():
            return True

        def release(block):
            # Release each physical payload while walking the radix leaves,
            # so shared-slot refcounts determine the next eviction's benefit.
            self.pool.release_snapshot(block.payload)
            return int(fits())

        self.index.evict_until_weight(1, release)
        return fits()

    def plan(self, seqs, requests):
        by_id = {seq.seq_id: seq for seq in seqs}
        entries, held, planned = [], {}, set()
        block_size = self.index.block_size
        costs = dict.fromkeys(self.pool.slots, 0)
        try:
            for seq_id, start, end in requests:
                seq = by_id[seq_id]
                for ratio, need in self.pool.append_costs(start, end).items():
                    costs[ratio] += need
                path = self._path(seq, end)
                for block_idx in range(start // block_size, end // block_size):
                    key = path[block_idx]
                    if self.index.has_block(key) or key in planned:
                        continue
                    parent_id = path[block_idx - 1] if block_idx else None
                    parent = self.index.get_block(parent_id) if parent_id is not None else None
                    if parent_id is not None and parent is None and parent_id not in planned:
                        break
                    if parent is not None and parent_id not in held:
                        self.index.acquire_block_ref(parent)
                        held[parent_id] = parent
                    boundary = (block_idx + 1) * block_size
                    entries.append((seq_id, boundary, key, parent_id,
                                    tuple(seq.token_ids[boundary-block_size:boundary])))
                    planned.add(key)
            self.evict_for_capacity(costs, snapshot_rows=min(len(entries), self.index.max_blocks))
            entries = entries[:len(self.pool.free_snapshot_rows)]
            return NativePrefixPlan(tuple(entries), tuple(held.values()))
        except Exception:
            for parent in held.values():
                self.index.release_block_ref(parent)
            raise

    def finish_plan(self, plan):
        for parent in plan.parents:
            self.index.release_block_ref(parent)
        plan.parents = ()

    def publish(self, plan, step):
        if len(plan.entries) != len(step.snapshots) or any(not snapshot.ready for snapshot in step.snapshots):
            raise RuntimeError("Native radix publication requires the committed physical snapshot plan.")
        try:
            for entry, snapshot in zip(plan.entries, step.snapshots):
                _, end, key, parent_id, tokens = entry
                block = PrefixCacheBlock(key, parent_id, self.index.block_size,
                                         end // self.index.block_size - 1, snapshot, tokens)
                actual = self.index.insert_block(block)
                if actual is not block:
                    self.pool.release_snapshot(snapshot)
        finally:
            self.finish_plan(plan)

    def release_sequence(self, seq_id):
        self._release_hit(seq_id)
        self.paths.pop(seq_id, None)
