from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from deltakv.modeling.kv_cache import CompressedKVCache, ClusterCompressedKVCache


class FullCompressedKVCache(CompressedKVCache):
    """DeltaKV cache where full-attention layers also flush old history through the
    compressor path, but always reconstruct all compressed history on read.
    """

    def _build_full_layer_response(
        self,
        *,
        layer_idx: int,
        bs: int,
        k_dim: int,
        sink_idx: torch.Tensor,
        buffer_idx: torch.Tensor,
        compressor_up: Optional[nn.Module],
    ):
        if layer_idx in self.comp_kv_cache:
            history_k, history_v = self._decompress_all_history_kv(
                layer_idx=layer_idx,
                compressor_up=compressor_up,
                bs=bs,
                k_dim=k_dim,
            )
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

    def _flush_full_layer_history(
        self,
        *,
        layer_idx: int,
        compressor_down: Optional[nn.Module],
    ) -> None:
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

        if self.config.use_compression:
            to_be_compress = torch.cat([key_hist, value_hist], dim=-1)
            comp_kv, bases = self.compress(to_be_compress, compressor_down)
            if layer_idx not in self.comp_kv_cache:
                self.comp_kv_cache[layer_idx], self.bases_cache[layer_idx] = comp_kv, bases
            else:
                self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], comp_kv], dim=1)
                self.bases_cache[layer_idx] = torch.cat([self.bases_cache[layer_idx], bases], dim=1)
            return

        if self.config.kv_quant_bits == 4:
            packed_kv, scale, mn = self._quantize_kv_tokens(torch.cat([key_hist, value_hist], dim=-1))
            if layer_idx not in self.comp_kv_cache:
                self.comp_kv_cache[layer_idx] = packed_kv
                self.comp_kv_scales[layer_idx] = scale
                self.comp_kv_mins[layer_idx] = mn
            else:
                self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], packed_kv], dim=1)
                self.comp_kv_scales[layer_idx] = torch.cat([self.comp_kv_scales[layer_idx], scale], dim=1)
                self.comp_kv_mins[layer_idx] = torch.cat([self.comp_kv_mins[layer_idx], mn], dim=1)
            return

        if layer_idx not in self.comp_kv_cache:
            self.comp_kv_cache[layer_idx], self.bases_cache[layer_idx] = key_hist, value_hist
        else:
            self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], key_hist], dim=1)
            self.bases_cache[layer_idx] = torch.cat([self.bases_cache[layer_idx], value_hist], dim=1)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict] = None,
        compressor_down: Optional[nn.Module] = None,
        compressor_up: Optional[nn.Module] = None,
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
            compressor_up=compressor_up,
        )
        self._flush_full_layer_history(layer_idx=layer_idx, compressor_down=compressor_down)
        return response


class FullClusterCompressedKVCache(ClusterCompressedKVCache):
    """Clustered DeltaKV cache where full-attention layers reconstruct the entire
    compressed history but still store old tokens via the clustered compressor path.
    """

    def _build_full_layer_response(
        self,
        *,
        layer_idx: int,
        bs: int,
        k_dim: int,
        sink_idx: torch.Tensor,
        buffer_idx: torch.Tensor,
        compressor_up: Optional[nn.Module],
    ):
        if layer_idx in self.comp_kv_cache:
            if not self.config.use_compression:
                raise NotImplementedError("Cluster without compression is not implemented")
            history_k, history_v = self._reconstruct_all_cluster_tokens(
                layer_idx=layer_idx,
                compressor_up=compressor_up,
                bs=bs,
                k_dim=k_dim,
            )
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

    def _flush_full_layer_history(
        self,
        *,
        layer_idx: int,
        compressor_down: Optional[nn.Module],
    ) -> None:
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

        to_be_compress = torch.cat([key_hist, value_hist], dim=-1)
        existing_centers = self.bases_cache.get(layer_idx, None)
        hist_len = int(self.comp_kv_cache[layer_idx].shape[1]) if layer_idx in self.comp_kv_cache else 0
        abs_start_pos = int(self.sink_size + hist_len)
        comp_kv, all_centers, father_idx, scale, mn = self.compress(
            to_be_compress,
            compressor_down,
            existing_centers,
            abs_start_pos=abs_start_pos,
        )
        self.bases_cache[layer_idx] = all_centers

        if layer_idx not in self.comp_kv_cache:
            self.comp_kv_cache[layer_idx] = comp_kv
            self.token_father_idx[layer_idx] = father_idx
            if scale is not None:
                self.comp_kv_scales[layer_idx] = scale
                self.comp_kv_mins[layer_idx] = mn
            return

        self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], comp_kv], dim=1)
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
        compressor_down: Optional[nn.Module] = None,
        compressor_up: Optional[nn.Module] = None,
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
            compressor_up=compressor_up,
        )
        self._flush_full_layer_history(layer_idx=layer_idx, compressor_down=compressor_down)
        return response
