from __future__ import annotations

from typing import Optional

import torch

from deltakv.modeling.kv_cache import CompressedKVCache, ClusterCompressedKVCache


class OriginResidualQuantCompressedKVCache(CompressedKVCache):
    """For full-attention layers, store token-space residuals (token - ref) directly.

    Sparse layers continue to use the original DeltaKV compressor path.
    """

    def _build_chunk_bases(self, kv_states: torch.Tensor) -> torch.Tensor:
        bs, seq_len, kv_dim = kv_states.shape
        kv_chunks = kv_states.view(bs, -1, self.config.seq_chunk_size, kv_dim)
        if self.config.ref_mode == "avg":
            bases = kv_chunks.mean(dim=2, keepdim=True)
        elif self.config.ref_mode == "first":
            bases = kv_chunks[:, :, :1]
        else:
            raise ValueError(f"Unsupported ref_mode for origin_residual_quant: {self.config.ref_mode}")
        return bases.reshape(bs, -1, kv_dim)

    def _reconstruct_full_history(self, layer_idx: int, *, k_dim: int):
        if self.config.kv_quant_bits == 4:
            residual = self._dequantize_kv_tokens(
                self.comp_kv_cache[layer_idx],
                self.comp_kv_scales[layer_idx],
                self.comp_kv_mins[layer_idx],
                2 * k_dim,
            )
        else:
            residual = self.comp_kv_cache[layer_idx]

        bases = self.bases_cache[layer_idx]
        bases = bases.repeat_interleave(self.config.seq_chunk_size, dim=1)[:, : residual.shape[1]]
        recon_kv = (residual + bases).view(residual.shape[0], -1, 2, k_dim)
        return recon_kv[:, :, 0], recon_kv[:, :, 1]

    def _build_full_layer_response(
        self,
        *,
        layer_idx: int,
        bs: int,
        k_dim: int,
        sink_idx: torch.Tensor,
        buffer_idx: torch.Tensor,
    ):
        if layer_idx in self.comp_kv_cache:
            history_k, history_v = self._reconstruct_full_history(layer_idx, k_dim=k_dim)
            num_hist = history_k.shape[1]
            hist_idx = (
                torch.arange(num_hist, device=buffer_idx.device, dtype=buffer_idx.dtype)
                .unsqueeze(0)
                .expand(bs, -1)
                + self.sink_size
            )
            full_idx = torch.cat([sink_idx, hist_idx, buffer_idx], dim=1)
            return (
                torch.cat([self.sink_key_cache[layer_idx], history_k, self.buffer_key_cache[layer_idx]], dim=1),
                torch.cat([self.sink_value_cache[layer_idx], history_v, self.buffer_value_cache[layer_idx]], dim=1),
                full_idx,
            )

        full_idx = torch.cat([sink_idx, buffer_idx], dim=1)
        return (
            torch.cat([self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]], dim=1),
            torch.cat([self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]], dim=1),
            full_idx,
        )

    def _flush_full_layer_history(self, *, layer_idx: int) -> None:
        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        if buffer_len < self.tail_token_size * 2:
            return

        compress_len = (buffer_len - self.tail_token_size) // self.tail_token_size * self.tail_token_size
        if compress_len <= 0:
            return
        if compress_len % self.config.seq_chunk_size != 0:
            raise ValueError(
                f"compress_len={compress_len} must be divisible by seq_chunk_size={self.config.seq_chunk_size}"
            )

        key_hist, self.buffer_key_cache[layer_idx] = (
            self.buffer_key_cache[layer_idx][:, :compress_len],
            self.buffer_key_cache[layer_idx][:, compress_len:],
        )
        value_hist, self.buffer_value_cache[layer_idx] = (
            self.buffer_value_cache[layer_idx][:, :compress_len],
            self.buffer_value_cache[layer_idx][:, compress_len:],
        )

        kv_hist = torch.cat([key_hist, value_hist], dim=-1)
        bases = self._build_chunk_bases(kv_hist)
        bases_per_token = bases.repeat_interleave(self.config.seq_chunk_size, dim=1)[:, : kv_hist.shape[1]]
        residual = kv_hist - bases_per_token

        if self.config.kv_quant_bits == 4:
            packed, scale, mn = self._quantize_kv_tokens(residual)
            if layer_idx not in self.comp_kv_cache:
                self.comp_kv_cache[layer_idx] = packed
                self.comp_kv_scales[layer_idx] = scale
                self.comp_kv_mins[layer_idx] = mn
            else:
                self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], packed], dim=1)
                self.comp_kv_scales[layer_idx] = torch.cat([self.comp_kv_scales[layer_idx], scale], dim=1)
                self.comp_kv_mins[layer_idx] = torch.cat([self.comp_kv_mins[layer_idx], mn], dim=1)
        else:
            if layer_idx not in self.comp_kv_cache:
                self.comp_kv_cache[layer_idx] = residual
            else:
                self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], residual], dim=1)

        if layer_idx not in self.bases_cache:
            self.bases_cache[layer_idx] = bases
        else:
            self.bases_cache[layer_idx] = torch.cat([self.bases_cache[layer_idx], bases], dim=1)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
        compressor_down=None,
        compressor_up=None,
    ):
        if layer_idx not in self.full_attn_layers:
            return super().update(
                key_states,
                value_states,
                layer_idx,
                cache_kwargs=cache_kwargs,
                compressor_down=compressor_down,
                compressor_up=compressor_up,
            )

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if key_states is None:
            raise NotImplementedError

        bs, _, k_dim = key_states.shape
        if layer_idx not in self.buffer_key_cache:
            self.sink_size = min(self.sink_size, key_states.shape[1])
            self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx] = (
                key_states[:, :self.sink_size],
                key_states[:, self.sink_size:],
            )
            self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx] = (
                value_states[:, :self.sink_size],
                value_states[:, self.sink_size:],
            )
        else:
            self.buffer_key_cache[layer_idx] = torch.cat([self.buffer_key_cache[layer_idx], key_states], dim=1)
            self.buffer_value_cache[layer_idx] = torch.cat([self.buffer_value_cache[layer_idx], value_states], dim=1)

        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        seen_tokens = self._seen_tokens
        sink_idx = torch.arange(0, self.sink_size, device=key_states.device).unsqueeze(0).expand(bs, -1)
        buffer_idx = torch.arange(
            seen_tokens - buffer_len, seen_tokens, device=key_states.device
        ).unsqueeze(0).expand(bs, -1)

        response = self._build_full_layer_response(
            layer_idx=layer_idx,
            bs=bs,
            k_dim=k_dim,
            sink_idx=sink_idx,
            buffer_idx=buffer_idx,
        )
        self._flush_full_layer_history(layer_idx=layer_idx)
        return response


class OriginResidualQuantClusterCompressedKVCache(ClusterCompressedKVCache):
    """Clustered variant storing 4-bit token-space residuals against full-precision prototypes."""

    def _compress_origin_residual(
        self,
        kv_states: torch.Tensor,
        existing_centers: Optional[torch.Tensor] = None,
        *,
        abs_start_pos: Optional[int] = None,
    ):
        bs, seq_len, kv_dim = kv_states.shape
        cluster_step = max(1, int(1 / self.config.cluster_ratio))
        stride_alpha = float(getattr(self.config, "stride_alpha", 0.0) or 0.0)
        if abs_start_pos is None:
            abs_start_pos = 0

        if stride_alpha <= 0.0:
            new_centers = kv_states[:, ::cluster_step, :].contiguous()
            center_rel = None
        else:
            center_rel = self._get_dynamic_center_rel(
                abs_start_pos=int(abs_start_pos),
                seq_len=int(seq_len),
                base_step=int(cluster_step),
                device=kv_states.device,
            )
            new_centers = kv_states[:, center_rel.to(torch.long), :].contiguous()
        all_centers = torch.cat([existing_centers, new_centers], dim=1) if existing_centers is not None else new_centers

        num_existing = existing_centers.shape[1] if existing_centers is not None else 0
        num_new = new_centers.shape[1]

        metric_type = self.config.cluster_metric
        use_kv = self.config.cluster_on_kv
        if metric_type == "l2":
            scores = self._metric_l2(kv_states, all_centers, use_kv=use_kv)
        elif metric_type == "dot":
            scores = self._metric_dot(kv_states, all_centers, use_kv=use_kv)
        elif metric_type == "cosine":
            scores = self._metric_cosine(kv_states, all_centers, use_kv=use_kv)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        rows = torch.arange(seq_len, device=kv_states.device).view(-1, 1)
        if center_rel is None:
            cols = torch.arange(num_new, device=kv_states.device).view(1, -1) * cluster_step
        else:
            cols = center_rel.to(torch.long).view(1, -1)
        mask_new = cols <= rows
        mask_existing = torch.ones((seq_len, num_existing), device=kv_states.device, dtype=torch.bool)
        full_mask = torch.cat([mask_existing, mask_new], dim=1)
        scores = scores.masked_fill(~full_mask.unsqueeze(0), float("-inf"))

        k = max(1, self.config.seq_chunk_size)
        _, topk_indices = torch.topk(scores, k=min(k, all_centers.shape[1]), dim=-1)

        indices = topk_indices.view(bs, -1)[:, :, None].expand(-1, -1, kv_dim)
        gathered_fathers = all_centers.gather(1, indices).view(bs, seq_len, -1, kv_dim).mean(dim=2)
        residual = kv_states - gathered_fathers

        if self.config.kv_quant_bits == 4:
            packed, scale, mn = self._quantize_kv_tokens(residual)
            return packed, all_centers, topk_indices, scale, mn
        return residual, all_centers, topk_indices, None, None

    def _reconstruct_full_history(self, layer_idx: int, *, k_dim: int):
        if self.config.kv_quant_bits == 4:
            residual = self._dequantize_kv_tokens(
                self.comp_kv_cache[layer_idx],
                self.comp_kv_scales[layer_idx],
                self.comp_kv_mins[layer_idx],
                2 * k_dim,
            )
        else:
            residual = self.comp_kv_cache[layer_idx]

        bases = self.bases_cache[layer_idx]
        father_idx = self.token_father_idx[layer_idx]
        k = father_idx.shape[-1]
        flat_idx = father_idx.view(residual.shape[0], -1)[:, :, None].expand(-1, -1, bases.shape[-1])
        gathered = bases.gather(1, flat_idx).view(residual.shape[0], residual.shape[1], k, -1).mean(dim=2)
        recon_kv = (residual + gathered).view(residual.shape[0], -1, 2, k_dim)
        return recon_kv[:, :, 0], recon_kv[:, :, 1]

    def _build_full_layer_response(
        self,
        *,
        layer_idx: int,
        bs: int,
        k_dim: int,
        sink_idx: torch.Tensor,
        buffer_idx: torch.Tensor,
    ):
        if layer_idx in self.comp_kv_cache:
            history_k, history_v = self._reconstruct_full_history(layer_idx, k_dim=k_dim)
            num_hist = history_k.shape[1]
            hist_idx = (
                torch.arange(num_hist, device=buffer_idx.device, dtype=buffer_idx.dtype)
                .unsqueeze(0)
                .expand(bs, -1)
                + self.sink_size
            )
            full_idx = torch.cat([sink_idx, hist_idx, buffer_idx], dim=1)
            return (
                torch.cat([self.sink_key_cache[layer_idx], history_k, self.buffer_key_cache[layer_idx]], dim=1),
                torch.cat([self.sink_value_cache[layer_idx], history_v, self.buffer_value_cache[layer_idx]], dim=1),
                full_idx,
            )

        full_idx = torch.cat([sink_idx, buffer_idx], dim=1)
        return (
            torch.cat([self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]], dim=1),
            torch.cat([self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]], dim=1),
            full_idx,
        )

    def _flush_full_layer_history(self, *, layer_idx: int) -> None:
        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        if buffer_len < self.tail_token_size * 2:
            return

        compress_len = (buffer_len - self.tail_token_size) // self.tail_token_size * self.tail_token_size
        if compress_len <= 0:
            return

        key_hist, self.buffer_key_cache[layer_idx] = (
            self.buffer_key_cache[layer_idx][:, :compress_len],
            self.buffer_key_cache[layer_idx][:, compress_len:],
        )
        value_hist, self.buffer_value_cache[layer_idx] = (
            self.buffer_value_cache[layer_idx][:, :compress_len],
            self.buffer_value_cache[layer_idx][:, compress_len:],
        )

        kv_hist = torch.cat([key_hist, value_hist], dim=-1)
        existing_centers = self.bases_cache.get(layer_idx, None)
        hist_len = int(self.comp_kv_cache[layer_idx].shape[1]) if layer_idx in self.comp_kv_cache else 0
        abs_start_pos = int(self.sink_size + hist_len)
        residual, all_centers, father_idx, scale, mn = self._compress_origin_residual(
            kv_hist,
            existing_centers,
            abs_start_pos=abs_start_pos,
        )
        self.bases_cache[layer_idx] = all_centers

        if layer_idx not in self.comp_kv_cache:
            self.comp_kv_cache[layer_idx] = residual
            self.token_father_idx[layer_idx] = father_idx
            if scale is not None:
                self.comp_kv_scales[layer_idx] = scale
                self.comp_kv_mins[layer_idx] = mn
            return

        self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], residual], dim=1)
        self.token_father_idx[layer_idx] = torch.cat([self.token_father_idx[layer_idx], father_idx], dim=1)
        if scale is not None:
            self.comp_kv_scales[layer_idx] = torch.cat([self.comp_kv_scales[layer_idx], scale], dim=1)
            self.comp_kv_mins[layer_idx] = torch.cat([self.comp_kv_mins[layer_idx], mn], dim=1)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
        compressor_down=None,
        compressor_up=None,
    ):
        if layer_idx not in self.full_attn_layers:
            return super().update(
                key_states,
                value_states,
                layer_idx,
                cache_kwargs=cache_kwargs,
                compressor_down=compressor_down,
                compressor_up=compressor_up,
            )

        assert self.config.use_cluster

        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if key_states is None:
            raise NotImplementedError

        bs, _, k_dim = key_states.shape
        if layer_idx not in self.buffer_key_cache:
            self.sink_size = min(self.sink_size, key_states.shape[1])
            self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx] = (
                key_states[:, :self.sink_size],
                key_states[:, self.sink_size:],
            )
            self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx] = (
                value_states[:, :self.sink_size],
                value_states[:, self.sink_size:],
            )
            self.bases_cache[layer_idx] = torch.cat(
                [self.sink_key_cache[layer_idx], self.sink_value_cache[layer_idx]], dim=-1
            )
            if self._cluster_next_center_abs_pos is None:
                self._cluster_next_center_abs_pos = int(self.sink_size)
        else:
            self.buffer_key_cache[layer_idx] = torch.cat([self.buffer_key_cache[layer_idx], key_states], dim=1)
            self.buffer_value_cache[layer_idx] = torch.cat([self.buffer_value_cache[layer_idx], value_states], dim=1)

        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        seen_tokens = self._seen_tokens
        sink_idx = torch.arange(0, self.sink_size, device=key_states.device).unsqueeze(0).expand(bs, -1)
        buffer_idx = torch.arange(
            seen_tokens - buffer_len, seen_tokens, device=key_states.device
        ).unsqueeze(0).expand(bs, -1)

        response = self._build_full_layer_response(
            layer_idx=layer_idx,
            bs=bs,
            k_dim=k_dim,
            sink_idx=sink_idx,
            buffer_idx=buffer_idx,
        )
        self._flush_full_layer_history(layer_idx=layer_idx)
        return response
