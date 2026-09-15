"""Bounded index selection owned by the native sparse runtime."""

from dataclasses import replace

import torch

from sparsevllm.operators.compressed_index import CompressedIndexSpec, prepare_compressed_index


class NativeIndexSelection:
    def __init__(self, *, num_heads, head_dim, max_index_tokens, query_chunk_size, top_k, parallel_context, device_index):
        self.parallel_context = parallel_context
        self.chunk_size = query_chunk_size
        self.provider = prepare_compressed_index(
            CompressedIndexSpec(num_heads, head_dim, max_index_tokens, query_chunk_size, top_k),
            device_index=device_index,
        )
        self._columns = None
        if parallel_context.attn_tp_size > 1:
            self._columns = torch.arange(max_index_tokens, device=self.provider.device)

    def graph_keepalive_tensors(self):
        tensors = list(self.provider.graph_keepalive_tensors())
        if self._columns is not None:
            tensors.append(self._columns)
        return tensors

    def select_compressed_index(self, query, weights, view, *, out):
        for start in range(0, len(query), self.chunk_size):
            end = min(len(query), start + self.chunk_size)
            chunk = replace(view, query_rows=view.query_rows[start:end], visible_lengths=view.visible_lengths[start:end])
            scores = self.provider.score(query[start:end], weights[start:end], chunk)
            if self._columns is not None:
                scores.masked_fill_(self._columns[None] >= chunk.visible_lengths[:, None], 0.)
                scores = self.parallel_context.attn_tp.all_reduce(scores)
            self.provider.select(scores, chunk, out=out[start:end])
        return out
