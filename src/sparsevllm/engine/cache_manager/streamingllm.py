from __future__ import annotations

from sparsevllm.config import Config
from sparsevllm.distributed import ParallelContext
from sparsevllm.engine.sequence import Sequence

from .snapkv import SnapKVCacheManager


class StreamingLLMCacheManager(SnapKVCacheManager):
    """Attention-sink / StreamingLLM cache manager.

    Reuse the standard per-layer physical slot bookkeeping from SnapKV, but keep
    scheduling headroom aligned with the fixed recent-window policy instead of
    score-based top-k eviction.
    """

    def __init__(self, config: Config, parallel_context: ParallelContext):
        super().__init__(config, parallel_context)
        # StreamingLLM applies the same prefix/recent window on every layer, so
        # decode metadata stays layer-uniform even after compaction.
        self._uniform_decode_metadata = True

    def prefill_batched_tokens_margin(self) -> int:
        return int(self.config.recent_keep_tokens)

    def _decode_graph_metadata_always_uniform(self) -> bool:
        return True

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        recent = int(self.config.recent_keep_tokens)
        if recent > 0 and remaining > recent:
            return remaining - recent
        return remaining

    def free_prefix_recent_slots_batch_layers(
        self,
        layer_indices: list[int],
        seqs: list[Sequence],
        *,
        kv_len: int,
        sink_keep_tokens: int,
        recent_keep_tokens: int,
    ):
        super().free_prefix_recent_slots_batch_layers(
            layer_indices,
            seqs,
            kv_len=kv_len,
            sink_keep_tokens=sink_keep_tokens,
            recent_keep_tokens=recent_keep_tokens,
        )
        if layer_indices and len(layer_indices) == self.num_layers:
            self._uniform_decode_metadata = True
