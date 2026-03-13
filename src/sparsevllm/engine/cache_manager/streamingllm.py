from __future__ import annotations

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence

from .snapkv import SnapKVCacheManager


class StreamingLLMCacheManager(SnapKVCacheManager):
    """Attention-sink / StreamingLLM cache manager.

    Reuse the standard per-layer physical slot bookkeeping from SnapKV, but keep
    scheduling headroom aligned with the fixed recent-window policy instead of
    score-based top-k eviction.
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)

    def prefill_batched_tokens_margin(self) -> int:
        return int(getattr(self.config, "num_recent_tokens", 0) or 0)

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        recent = int(getattr(self.config, "num_recent_tokens", 0) or 0)
        if recent > 0 and remaining > recent:
            return remaining - recent
        return remaining
