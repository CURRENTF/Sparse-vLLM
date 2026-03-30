from __future__ import annotations

from typing import Optional

import torch

from deltakv.modeling.origin_residual_quant_cache import (
    OriginResidualQuantClusterCompressedKVCache,
)


class AllOriginResidualQuantClusterCompressedKVCache(OriginResidualQuantClusterCompressedKVCache):
    """Store token-space residuals (token - ref) for every layer.

    This variant only supports `use_cluster=True`. Full-attention layers reconstruct all
    compressed history, while sparse layers reconstruct either selected or all history
    tokens depending on whether external token selection is available.
    """

    def _reconstruct_selected_cluster_tokens(
        self,
        *,
        layer_idx: int,
        token_idx: torch.Tensor,
        bs: int,
        k_dim: int,
    ):
        comp_kv = self.comp_kv_cache[layer_idx]
        bases = self.bases_cache[layer_idx]
        topk_father_idx = self.token_father_idx[layer_idx]
        k = topk_father_idx.shape[-1]

        imp_comp_kv = comp_kv.gather(1, token_idx[:, :, None].expand(-1, -1, comp_kv.shape[-1]))

        if self.config.kv_quant_bits == 4:
            scales = self.comp_kv_scales[layer_idx]
            mins = self.comp_kv_mins[layer_idx]
            imp_scales = scales.gather(1, token_idx[:, :, None].expand(-1, -1, scales.shape[-1]))
            imp_mins = mins.gather(1, token_idx[:, :, None].expand(-1, -1, mins.shape[-1]))
            imp_comp_kv = self._dequantize_kv_tokens(
                imp_comp_kv,
                imp_scales,
                imp_mins,
                2 * k_dim,
            )

        imp_topk_idx = topk_father_idx.gather(1, token_idx[:, :, None].expand(-1, -1, k))
        flat_idx = imp_topk_idx.view(bs, -1)[:, :, None].expand(-1, -1, bases.shape[-1])
        imp_bases = bases.gather(1, flat_idx).view(bs, token_idx.shape[1], k, -1).mean(dim=2)

        recon_kv = (imp_comp_kv + imp_bases).view(bs, -1, 2, k_dim)
        return recon_kv[:, :, 0], recon_kv[:, :, 1]

    def _reconstruct_all_cluster_tokens(
        self,
        *,
        layer_idx: int,
        bs: int,
        k_dim: int,
    ):
        if self.config.kv_quant_bits == 4:
            comp_kv = self._dequantize_kv_tokens(
                self.comp_kv_cache[layer_idx],
                self.comp_kv_scales[layer_idx],
                self.comp_kv_mins[layer_idx],
                2 * k_dim,
            )
        else:
            comp_kv = self.comp_kv_cache[layer_idx]

        bases = self.bases_cache[layer_idx]
        topk_father_idx = self.token_father_idx[layer_idx]
        num_hist = comp_kv.shape[1]
        k = topk_father_idx.shape[-1]

        flat_idx = topk_father_idx.view(bs, -1)[:, :, None].expand(-1, -1, bases.shape[-1])
        all_bases = bases.gather(1, flat_idx).view(bs, num_hist, k, -1).mean(dim=2)

        recon_kv = (comp_kv + all_bases).view(bs, -1, 2, k_dim)
        return recon_kv[:, :, 0], recon_kv[:, :, 1]

    def _flush_layer_history(self, *, layer_idx: int) -> None:
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
        del cache_kwargs, compressor_down, compressor_up
        assert self.config.use_cluster, "AllOriginResidualQuant only supports use_cluster=True"

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
        buffer_idx = torch.arange(seen_tokens - buffer_len, seen_tokens, device=key_states.device).unsqueeze(0).expand(bs, -1)

        if layer_idx in self.full_attn_layers:
            response = self._build_full_layer_response(
                layer_idx=layer_idx,
                bs=bs,
                k_dim=k_dim,
                sink_idx=sink_idx,
                buffer_idx=buffer_idx,
            )
        elif layer_idx in self.comp_kv_cache:
            if self.layer_to_full_layer_idx[layer_idx] in self.top_token_idx:
                token_idx = self.top_token_idx[self.layer_to_full_layer_idx[layer_idx]]
                recon_k, recon_v = self._reconstruct_selected_cluster_tokens(
                    layer_idx=layer_idx,
                    token_idx=token_idx,
                    bs=bs,
                    k_dim=k_dim,
                )
                full_idx = torch.cat([sink_idx, token_idx + self.sink_size, buffer_idx], dim=1)
                response = (
                    torch.cat([self.sink_key_cache[layer_idx], recon_k, self.buffer_key_cache[layer_idx]], dim=1),
                    torch.cat([self.sink_value_cache[layer_idx], recon_v, self.buffer_value_cache[layer_idx]], dim=1),
                    full_idx,
                )
            else:
                recon_k, recon_v = self._reconstruct_all_cluster_tokens(
                    layer_idx=layer_idx,
                    bs=bs,
                    k_dim=k_dim,
                )
                num_hist = recon_k.shape[1]
                hist_idx = torch.arange(num_hist, device=key_states.device).unsqueeze(0).expand(bs, -1) + self.sink_size
                full_idx = torch.cat([sink_idx, hist_idx, buffer_idx], dim=1)
                response = (
                    torch.cat([self.sink_key_cache[layer_idx], recon_k, self.buffer_key_cache[layer_idx]], dim=1),
                    torch.cat([self.sink_value_cache[layer_idx], recon_v, self.buffer_value_cache[layer_idx]], dim=1),
                    full_idx,
                )
        else:
            full_idx = torch.cat([sink_idx, buffer_idx], dim=1)
            response = (
                torch.cat([self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]], dim=1),
                torch.cat([self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]], dim=1),
                full_idx,
            )

        self._flush_layer_history(layer_idx=layer_idx)
        return response
