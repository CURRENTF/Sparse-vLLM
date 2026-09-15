"""Native query-dependent compressed-key selection and graph workspace lifetime."""

from .base import SparseMethodRuntime
from .native_index import NativeIndexSelection


class DeepseekV4Runtime(SparseMethodRuntime):
    def __init__(self, config, cache_manager):
        super().__init__(config, cache_manager)
        hf = config.hf_config
        max_index_tokens = max(1, config.max_model_len // 4)
        # Batch prefill queries while bounding the temporary BF16 score matrix.
        query_chunk_size = min(config.max_num_batched_tokens,
                               max(1, (8 * 1024**2) // (2 * max_index_tokens)))
        self.selection = NativeIndexSelection(
            num_heads=hf.index_n_heads // cache_manager.parallel_context.attn_tp_size,
            head_dim=hf.index_head_dim, max_index_tokens=max_index_tokens,
            query_chunk_size=query_chunk_size, top_k=hf.index_topk,
            parallel_context=cache_manager.parallel_context,
            device_index=cache_manager.device.index,
        )

    def needs_attention_score(self, layer_idx, step):
        return False

    def build_prefill_selection(self, request):
        raise TypeError("Native attention uses the typed compressed-index selection contract.")

    def build_decode_selection(self, request):
        raise TypeError("Native attention uses the typed compressed-index selection contract.")

    def select_compressed_index(self, query, weights, view, *, out):
        return self.selection.select_compressed_index(query, weights, view, out=out)

    def decode_graph_keepalive_tensors(self):
        return self.selection.graph_keepalive_tensors()
