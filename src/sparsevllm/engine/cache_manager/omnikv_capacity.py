"""Shared physical-pool budget for startup profiling and production admission."""

from dataclasses import dataclass

from .omnikv_lru import OmniKVLRU


@dataclass(frozen=True)
class OmniKVPoolPlan:
    full_layers: tuple[int, ...]
    layer_groups: dict[int, int]
    selected_capacity: int
    cache_capacity: int
    fixed_bytes: int
    slot_bytes: int

    def budget(self, slots):
        return self.fixed_bytes + slots * self.slot_bytes


def plan_omnikv_pools(config, full_layers, num_layers, rows, per_layer):
    full = set(full_layers)
    # GPU-only OmniKV consumes full history before its first observer too.
    if full:
        full.update(range(min(full)))
    sparse = num_layers - len(full)
    if not full or sparse <= 0:
        raise ValueError(
            "OmniKV offload requires both full and sparse attention layers."
        )
    selected = min(
        config.max_model_len,
        config.sink_keep_tokens + config.decode_keep_tokens + config.recent_keep_tokens,
    )
    if selected == 0:
        raise ValueError(
            "OmniKV offload requires a positive total selected-token budget."
        )
    cache = min(getattr(config, "omnikv_offload_cache_tokens", 0), config.max_model_len)
    if cache and cache < selected:
        raise ValueError(
            "omnikv_offload_cache_tokens must cover the full selected-token budget."
        )
    layer_groups = {}
    for layer in range(num_layers):
        if layer in full:
            group = layer
        else:
            layer_groups[layer] = group
    groups = len(set(layer_groups.values()))
    # Logical slot table and stable compute-view table; full-layer storage plus
    # one shared full-history prefill pool and the allocator/host-map vectors.
    fixed = sparse * rows * selected * per_layer + 2 * rows * config.max_model_len * 4
    slot_bytes = (len(full) + 1) * per_layer + 8
    if cache:
        fixed += sparse * rows * cache * per_layer
        fixed += OmniKVLRU.metadata_bytes(rows, 0, cache, selected, groups)
        slot_bytes += groups * rows * 4
    return OmniKVPoolPlan(
        tuple(sorted(full)), layer_groups, selected, cache, fixed, slot_bytes
    )
