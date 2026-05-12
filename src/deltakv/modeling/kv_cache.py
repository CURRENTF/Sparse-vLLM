import os
from typing import Optional, Any

import torch
from torch import nn
import torch.nn.functional as F
from transformers import DynamicCache

from deltakv.configs.model_config_cls import KVQwen2Config, parse_full_attn_layers
from sparsevllm.triton_kernel.quant import triton_quantize_and_pack_along_last_dim, unpack_4bit_to_16bit
from sparsevllm.utils.log import log_once


class BaseCache(DynamicCache):
    def __init__(self, config: KVQwen2Config) -> None:
        super().__init__()
        self.tail_token_size = config.tail_token_size
        self.sink_size = config.num_sink_tokens
        self.top_token_idx = {}
        self.token_scores = {}
        self.num_prompt_tokens = None
        self.visual_token_prune_only = bool(getattr(config, "visual_token_prune_only", False))
        self.visual_compress_only = self.visual_token_prune_only
        self.invalid_token_position = torch.iinfo(torch.long).max // 4

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self._seen_tokens

    @staticmethod
    def _default_cache_position(key_states: torch.Tensor, q_len: int) -> torch.Tensor:
        return torch.arange(q_len, device=key_states.device)

    def _extract_current_metadata(
        self,
        key_states: torch.Tensor,
        cache_kwargs: Optional[dict[str, Any]],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs, q_len = key_states.shape[:2]
        cache_kwargs = cache_kwargs or {}
        cache_position = cache_kwargs.get("cache_position")
        if cache_position is None:
            cache_position = self._default_cache_position(key_states, q_len)
        cache_position = cache_position.to(device=key_states.device, dtype=torch.long)
        if cache_position.dim() == 1:
            token_pos = cache_position.unsqueeze(0).expand(bs, -1)
        else:
            token_pos = cache_position.to(device=key_states.device, dtype=torch.long)

        visual_mask = cache_kwargs.get("deltakv_visual_token_mask")
        if visual_mask is None:
            visual_mask = torch.zeros((bs, q_len), device=key_states.device, dtype=torch.bool)
        else:
            visual_mask = visual_mask.to(device=key_states.device, dtype=torch.bool)
            if visual_mask.dim() == 1:
                visual_mask = visual_mask.unsqueeze(0)
            if visual_mask.shape[0] == 1 and bs > 1:
                visual_mask = visual_mask.expand(bs, -1)

        attention_mask = cache_kwargs.get("deltakv_attention_mask")
        if attention_mask is None:
            token_valid_mask = torch.ones((bs, q_len), device=key_states.device, dtype=torch.bool)
        else:
            token_valid_mask = attention_mask.to(device=key_states.device, dtype=torch.bool)
            if token_valid_mask.dim() == 1:
                token_valid_mask = token_valid_mask.unsqueeze(0)
            if token_valid_mask.shape[0] == 1 and bs > 1:
                token_valid_mask = token_valid_mask.expand(bs, -1)
            token_valid_mask = token_valid_mask[:, -q_len:]
        return token_pos, visual_mask, token_valid_mask

    def _invalid_pos_like(self, token_pos: torch.Tensor) -> torch.Tensor:
        return torch.full_like(token_pos, self.invalid_token_position)

    def _valid_pos_mask(self, token_pos: torch.Tensor) -> torch.Tensor:
        return token_pos < self.invalid_token_position

    def _compressible_lengths_from_buffer(self, layer_idx: int) -> torch.Tensor:
        valid_counts = self._buffer_valid_mask(layer_idx).sum(dim=1)
        compress_lens = torch.div(
            (valid_counts - self.tail_token_size).clamp_min(0),
            self.tail_token_size,
            rounding_mode="floor",
        ) * self.tail_token_size
        return compress_lens.to(dtype=torch.long)

    def get_buffer_compressible_lengths(self, layer_idx: int, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if not hasattr(self, "buffer_pos_cache") or layer_idx not in self.buffer_pos_cache:
            return None
        lengths = self._compressible_lengths_from_buffer(layer_idx)
        return lengths.to(device=device) if device is not None else lengths

    def get_buffer_valid_lengths(self, layer_idx: int, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        if not hasattr(self, "buffer_pos_cache") or layer_idx not in self.buffer_pos_cache:
            return None
        lengths = self._buffer_valid_mask(layer_idx).sum(dim=1).to(dtype=torch.long)
        return lengths.to(device=device) if device is not None else lengths

    def _buffer_valid_mask(self, layer_idx: int) -> torch.Tensor:
        if hasattr(self, "buffer_token_valid_cache") and layer_idx in self.buffer_token_valid_cache:
            return self.buffer_token_valid_cache[layer_idx] & self._valid_pos_mask(self.buffer_pos_cache[layer_idx])
        return self._valid_pos_mask(self.buffer_pos_cache[layer_idx])

    def _dummy_positions_for_rows(self, layer_idx: int, *, pos_like: torch.Tensor) -> torch.Tensor:
        pad_pos_cache = getattr(self, "row_pad_pos_cache", {})
        if layer_idx not in pad_pos_cache:
            return pos_like.new_full((pos_like.shape[0],), self.invalid_token_position)
        return pad_pos_cache[layer_idx].to(device=pos_like.device, dtype=pos_like.dtype)

    def _first_valid_buffer_mask(self, layer_idx: int, lengths: torch.Tensor) -> torch.Tensor:
        valid = self._buffer_valid_mask(layer_idx)
        if valid.numel() == 0:
            return valid
        lengths = lengths.to(device=valid.device, dtype=torch.long)
        valid_rank = valid.to(torch.long).cumsum(dim=1) - 1
        return valid & (valid_rank < lengths.unsqueeze(1))

    def _pack_buffer_by_mask(
        self,
        layer_idx: int,
        select_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = self.buffer_key_cache[layer_idx]
        value = self.buffer_value_cache[layer_idx]
        pos = self.buffer_pos_cache[layer_idx]
        bs, _, k_dim = key.shape
        counts = select_mask.sum(dim=1)
        max_count = int(counts.max().item()) if counts.numel() else 0
        dummy_pos = self._dummy_positions_for_rows(layer_idx, pos_like=pos)
        packed_key = key.new_zeros((bs, max_count, k_dim))
        packed_value = value.new_zeros((bs, max_count, k_dim))
        packed_pos = dummy_pos[:, None].expand(-1, max_count).clone()
        for b in range(bs):
            idx = torch.nonzero(select_mask[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            n = int(idx.numel())
            packed_key[b, :n] = key[b].index_select(0, idx)
            packed_value[b, :n] = value[b].index_select(0, idx)
            packed_pos[b, :n] = pos[b].index_select(0, idx)
        return packed_key, packed_value, packed_pos

    def get_buffer_candidate_positions(
        self,
        layer_idx: int,
        lengths: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        if not hasattr(self, "buffer_pos_cache") or layer_idx not in self.buffer_pos_cache:
            return None
        select_mask = self._first_valid_buffer_mask(layer_idx, lengths)
        pos = self.buffer_pos_cache[layer_idx]
        counts = select_mask.sum(dim=1)
        max_count = int(counts.max().item()) if counts.numel() else 0
        dummy_pos = self._dummy_positions_for_rows(layer_idx, pos_like=pos)
        packed_pos = dummy_pos[:, None].expand(-1, max_count).clone()
        packed_valid = select_mask.new_zeros((pos.shape[0], max_count))
        for b in range(pos.shape[0]):
            idx = torch.nonzero(select_mask[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            n = int(idx.numel())
            packed_pos[b, :n] = pos[b].index_select(0, idx)
            packed_valid[b, :n] = True
        if device is not None:
            packed_pos = packed_pos.to(device=device)
            packed_valid = packed_valid.to(device=device)
        return packed_pos, packed_valid

    def _filter_buffer_by_mask(self, layer_idx: int, keep_mask: torch.Tensor) -> None:
        key = self.buffer_key_cache[layer_idx]
        value = self.buffer_value_cache[layer_idx]
        pos = self.buffer_pos_cache[layer_idx]
        visual = self.buffer_visual_mask_cache[layer_idx]
        bs, _, k_dim = key.shape
        keep_counts = keep_mask.sum(dim=1)
        max_keep = int(keep_counts.max().item()) if keep_counts.numel() else 0
        dummy_pos = self._dummy_positions_for_rows(layer_idx, pos_like=pos)
        new_key = key.new_zeros((bs, max_keep, k_dim))
        new_value = value.new_zeros((bs, max_keep, k_dim))
        new_pos = dummy_pos[:, None].expand(-1, max_keep).clone()
        new_visual = visual.new_zeros((bs, max_keep))
        valid_cache = getattr(self, "buffer_token_valid_cache", {}).get(layer_idx)
        new_valid = keep_mask.new_zeros((bs, max_keep))
        for b in range(bs):
            idx = torch.nonzero(keep_mask[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            n = int(idx.numel())
            new_key[b, :n] = key[b].index_select(0, idx)
            new_value[b, :n] = value[b].index_select(0, idx)
            new_pos[b, :n] = pos[b].index_select(0, idx)
            new_visual[b, :n] = visual[b].index_select(0, idx)
            if valid_cache is not None:
                new_valid[b, :n] = valid_cache[b].index_select(0, idx)
        self.buffer_key_cache[layer_idx] = new_key
        self.buffer_value_cache[layer_idx] = new_value
        self.buffer_pos_cache[layer_idx] = new_pos
        self.buffer_visual_mask_cache[layer_idx] = new_visual
        if valid_cache is not None:
            self.buffer_token_valid_cache[layer_idx] = new_valid

    def _init_pad_aware_layer_cache(
        self,
        *,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        token_pos: torch.Tensor,
        visual_mask: torch.Tensor,
    ) -> None:
        bs, _, k_dim = key_states.shape
        sink_len = int(self.sink_size)
        self.sink_key_cache[layer_idx] = key_states.new_zeros((bs, sink_len, k_dim))
        self.sink_value_cache[layer_idx] = value_states.new_zeros((bs, sink_len, k_dim))
        self.sink_pos_cache[layer_idx] = torch.full(
            (bs, sink_len),
            self.invalid_token_position,
            device=key_states.device,
            dtype=token_pos.dtype,
        )
        self.sink_filled_count[layer_idx] = torch.zeros((bs,), device=key_states.device, dtype=torch.long)
        self.row_pad_pos_cache[layer_idx] = torch.full(
            (bs,),
            self.invalid_token_position,
            device=key_states.device,
            dtype=token_pos.dtype,
        )
        self.buffer_key_cache[layer_idx] = key_states.new_zeros((bs, 0, k_dim))
        self.buffer_value_cache[layer_idx] = value_states.new_zeros((bs, 0, k_dim))
        self.buffer_pos_cache[layer_idx] = token_pos.new_full((bs, 0), self.invalid_token_position)
        self.buffer_visual_mask_cache[layer_idx] = visual_mask.new_zeros((bs, 0))
        if hasattr(self, "buffer_token_valid_cache"):
            self.buffer_token_valid_cache[layer_idx] = visual_mask.new_zeros((bs, 0))

    def _fill_sink_and_append_buffer(
        self,
        *,
        layer_idx: int,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        token_pos: torch.Tensor,
        visual_mask: torch.Tensor,
        token_valid_mask: torch.Tensor,
    ) -> None:
        sink_take_mask = torch.zeros_like(token_valid_mask, dtype=torch.bool)
        sink_len = int(self.sink_size)
        if hasattr(self, "row_pad_pos_cache") and layer_idx in self.row_pad_pos_cache:
            for b in range(key_states.shape[0]):
                if self.row_pad_pos_cache[layer_idx][b] < self.invalid_token_position:
                    continue
                pad_idx = torch.nonzero(~token_valid_mask[b], as_tuple=False).flatten()
                if pad_idx.numel() > 0:
                    self.row_pad_pos_cache[layer_idx][b] = token_pos[b, pad_idx[0]]

        if sink_len > 0:
            for b in range(key_states.shape[0]):
                remaining = sink_len - int(self.sink_filled_count[layer_idx][b].item())
                if remaining <= 0:
                    continue
                valid_idx = torch.nonzero(token_valid_mask[b], as_tuple=False).flatten()
                if valid_idx.numel() == 0:
                    continue
                take_idx = valid_idx[:remaining]
                start = int(self.sink_filled_count[layer_idx][b].item())
                end = start + int(take_idx.numel())
                self.sink_key_cache[layer_idx][b, start:end] = key_states[b].index_select(0, take_idx)
                self.sink_value_cache[layer_idx][b, start:end] = value_states[b].index_select(0, take_idx)
                self.sink_pos_cache[layer_idx][b, start:end] = token_pos[b].index_select(0, take_idx)
                sink_take_mask[b, take_idx] = True
                self.sink_filled_count[layer_idx][b] = end

        keep_mask = ~sink_take_mask
        keep_counts = keep_mask.sum(dim=1)
        max_keep = int(keep_counts.max().item()) if keep_counts.numel() else 0
        bs, _, k_dim = key_states.shape
        buffer_key = key_states.new_zeros((bs, max_keep, k_dim))
        buffer_value = value_states.new_zeros((bs, max_keep, k_dim))
        buffer_pos = token_pos.new_full((bs, max_keep), self.invalid_token_position)
        buffer_visual = visual_mask.new_zeros((bs, max_keep))
        buffer_valid = token_valid_mask.new_zeros((bs, max_keep))
        for b in range(bs):
            idx = torch.nonzero(keep_mask[b], as_tuple=False).flatten()
            if idx.numel() == 0:
                continue
            n = int(idx.numel())
            buffer_key[b, :n] = key_states[b].index_select(0, idx)
            buffer_value[b, :n] = value_states[b].index_select(0, idx)
            buffer_pos[b, :n] = token_pos[b].index_select(0, idx)
            buffer_visual[b, :n] = visual_mask[b].index_select(0, idx)
            buffer_valid[b, :n] = token_valid_mask[b].index_select(0, idx)

        self.buffer_key_cache[layer_idx] = torch.cat([self.buffer_key_cache[layer_idx], buffer_key], dim=1)
        self.buffer_value_cache[layer_idx] = torch.cat([self.buffer_value_cache[layer_idx], buffer_value], dim=1)
        self.buffer_pos_cache[layer_idx] = torch.cat([self.buffer_pos_cache[layer_idx], buffer_pos], dim=1)
        self.buffer_visual_mask_cache[layer_idx] = torch.cat([self.buffer_visual_mask_cache[layer_idx], buffer_visual], dim=1)
        if hasattr(self, "buffer_token_valid_cache"):
            self.buffer_token_valid_cache[layer_idx] = torch.cat([self.buffer_token_valid_cache[layer_idx], buffer_valid], dim=1)

    @staticmethod
    def _sort_cache_view(
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        token_pos: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        order = torch.argsort(token_pos, dim=1)
        gather_idx = order.unsqueeze(-1)
        sorted_key = key_states.gather(1, gather_idx.expand(-1, -1, key_states.shape[-1]))
        sorted_value = value_states.gather(1, gather_idx.expand(-1, -1, value_states.shape[-1]))
        sorted_pos = token_pos.gather(1, order)
        return sorted_key, sorted_value, sorted_pos

    def _build_cache_view(
        self,
        key_parts: list[torch.Tensor],
        value_parts: list[torch.Tensor],
        pos_parts: list[torch.Tensor],
        *,
        layer_idx: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key_states = torch.cat(key_parts, dim=1)
        value_states = torch.cat(value_parts, dim=1)
        token_pos = torch.cat(pos_parts, dim=1)
        if layer_idx is not None:
            invalid = ~self._valid_pos_mask(token_pos)
            if invalid.any():
                dummy_pos = self._dummy_positions_for_rows(layer_idx, pos_like=token_pos)
                token_pos = torch.where(invalid, dummy_pos[:, None], token_pos)
        return self._sort_cache_view(key_states, value_states, token_pos)

    def get_compressed_positions(self, layer_idx: int, device: Optional[torch.device] = None) -> Optional[torch.Tensor]:
        comp_pos_cache = getattr(self, "comp_pos_cache", None)
        if not comp_pos_cache or layer_idx not in comp_pos_cache:
            return None
        pos = comp_pos_cache[layer_idx]
        return pos.to(device=device) if device is not None else pos

    def get_compressed_length(self, layer_idx: int) -> int:
        comp_pos = self.get_compressed_positions(layer_idx)
        return 0 if comp_pos is None else int(comp_pos.shape[1])

    def _select_visual_store_and_drop_indices(
        self,
        *,
        layer_idx: int,
        candidate_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        visual_idx = torch.nonzero(
            self.buffer_visual_mask_cache[layer_idx][0, :candidate_len],
            as_tuple=False,
        ).flatten()
        if visual_idx.numel() == 0:
            return visual_idx, visual_idx

        keep_ratio = float(getattr(self.config, "visual_token_keep_ratio", 1.0) or 0.0)
        keep_ratio = max(0.0, min(1.0, keep_ratio))
        if keep_ratio >= 1.0:
            return visual_idx, visual_idx
        if keep_ratio <= 0.0:
            return torch.empty((0,), device=device, dtype=torch.long), visual_idx

        keep_count = max(1, int(round(float(visual_idx.numel()) * keep_ratio)))
        keep_count = min(int(visual_idx.numel()), keep_count)
        sample_pos = torch.div(
            torch.arange(keep_count, device=device, dtype=torch.long) * int(visual_idx.numel()),
            keep_count,
            rounding_mode="floor",
        )
        return visual_idx.index_select(0, sample_pos), visual_idx


class SnapKVCache(BaseCache):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        self.is_pruned = False
        self.num_layers = config.num_hidden_layers

    @property
    def is_last_chunk(self):
        return self._seen_tokens == self.num_prompt_tokens and not self.is_pruned

    def delete_tokens(self, layer_idx, top_token_idx):
        if layer_idx == self.num_layers - 1:
            self.is_pruned = True

        # top_token_idx shape: (bs, num_kv_heads, num_top) or (bs, num_top)
        if top_token_idx.dim() == 2:
            top_token_idx = top_token_idx.unsqueeze(1) # (bs, 1, num_top)

        bs, num_kv_heads_idx, num_top = top_token_idx.shape
        kv_len = self.key_cache[layer_idx].shape[2]
        head_dim = self.key_cache[layer_idx].shape[3]
        num_kv_heads = self.key_cache[layer_idx].shape[1]

        if num_kv_heads_idx == 1 and num_kv_heads > 1:
            top_token_idx = top_token_idx.expand(-1, num_kv_heads, -1)
            num_kv_heads_idx = num_kv_heads

        token_idx = top_token_idx + self.sink_size  # (bs, num_kv_heads, num_top)

        device = token_idx.device
        sink_idx = torch.arange(self.sink_size, device=device)[None, None, :].expand(bs, num_kv_heads_idx, -1)
        recent_idx = torch.arange(kv_len - self.tail_token_size, kv_len, device=device)[None, None, :].expand(bs, num_kv_heads_idx, -1)

        # final_idx shape: (bs, num_kv_heads, num_selected)
        final_idx = torch.cat([sink_idx, token_idx, recent_idx], dim=2)

        # Prepare index for 4D gather: (bs, num_kv_heads, num_selected, head_dim)
        gather_idx = final_idx.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        self.key_cache[layer_idx] = self.key_cache[layer_idx].gather(index=gather_idx, dim=2)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].gather(index=gather_idx, dim=2)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[dict[str, Any]] = None,
            compressor_down: Optional[nn.Module] = None,
            compressor_up: Optional[nn.Module] = None,
    ):
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]

        # Update the cache
        if key_states is not None:
            if len(self.key_cache) <= layer_idx:
                # There may be skipped layers, fill them with empty lists
                for _ in range(len(self.key_cache), layer_idx):
                    self.key_cache.append(torch.tensor([]))
                    self.value_cache.append(torch.tensor([]))
                self.key_cache.append(key_states)
                self.value_cache.append(value_states)
            elif (
                    not self.key_cache[layer_idx].numel()  # prefers not t.numel() to len(t) == 0 to export the model
            ):  # fills previously skipped layers; checking for tensor causes errors
                self.key_cache[layer_idx] = key_states
                self.value_cache[layer_idx] = value_states
            else:
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class CompressedKVCache(BaseCache):
    def __init__(self, config: KVQwen2Config) -> None:
        super().__init__(config)
        log_once("💡💡💡 CompressedKVCache", 'INFO')
        self._seen_tokens = 0  # Used in `generate` to keep tally of how many tokens the cache has seen
        self.comp_kv_cache = {}
        self.bases_cache = {}
        self.comp_kv_scales = {}
        self.comp_kv_mins = {}
        self.buffer_key_cache = {}
        self.buffer_value_cache = {}
        self.buffer_pos_cache = {}
        self.buffer_visual_mask_cache = {}
        self.buffer_token_valid_cache = {}
        self.sink_key_cache = {}
        self.sink_value_cache = {}
        self.sink_pos_cache = {}
        self.sink_filled_count = {}
        self.row_pad_pos_cache = {}
        self.comp_pos_cache = {}

        self.full_attn_layers = parse_full_attn_layers(config.full_attn_layers)
        config.full_attn_layers = self.full_attn_layers

        self.layer_to_full_layer_idx = {}
        _last = None
        for l in range(config.num_hidden_layers):
            if l in self.full_attn_layers:
                _last = l
            self.layer_to_full_layer_idx[l] = _last

        self.config = config
        self.current_compress_len = 0
        self.cos = None
        self.sin = None

        if config.use_compression:
            assert config.recon_mode == 'delta_in_latent', "TODO: 完成delta_in_origin的压缩（或者不需要？）"
            assert config.layer_chunk_size == 1, "TODO: 完成层压缩"
            if not getattr(config, "use_cluster", False):
                assert config.ref_mode == 'avg'

    def __len__(self):
        """
        Support for backwards-compatible `past_key_value` length, e.g. `len(past_key_value)`. This value corresponds
        to the number of layers in the model.
        """
        return len(self.buffer_key_cache)

    def compress(self, kv_states, compressor_down):
        bs, seq_len, kv_dim = kv_states.shape
        # kv_states shape: (bs, seq_len, kv_dim)
        # Reshape to chunks: (bs, num_chunks, chunk_size, kv_dim)
        kv_chunks = kv_states.view(bs, -1, self.config.compressor_token_group_size, kv_dim)

        # ref_mode='avg': use the mean of each chunk as base
        # bases_states shape: (bs, num_chunks, 1, kv_dim)
        bases_states = kv_chunks.mean(dim=2, keepdim=True)
        if os.getenv('REMOVE_COMP'):
            comp_kv = torch.zeros((bs, kv_chunks.shape[1], kv_chunks.shape[2], self.config.kv_compressed_size), device=kv_states.device, dtype=kv_states.dtype)
        elif os.getenv('REMOVE_REF'):
            comp_kv = compressor_down(kv_chunks)
        else:
            comp_kv = compressor_down(kv_chunks) - compressor_down(bases_states)

        return comp_kv.reshape(bs, seq_len, -1), bases_states.reshape(bs, -1, kv_dim)

    def decompress(self, comp_states, bases_states, compressor_up):
        # comp_states shape: (bs, num_chunks, chunk_size, comp_dim)
        # bases_states shape: (bs, num_chunks, 1, kv_dim)
        return compressor_up(comp_states) + bases_states

    def _quantize_kv_tokens(self, kv_states: torch.Tensor):
        packed, scale, mn = triton_quantize_and_pack_along_last_dim(
            kv_states.unsqueeze(1), kv_states.shape[-1], 4
        )
        return packed.squeeze(1), scale.squeeze(1), mn.squeeze(1)

    def _dequantize_kv_tokens(self, packed: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, kv_dim: int):
        return unpack_4bit_to_16bit(
            packed.unsqueeze(1),
            scale.unsqueeze(1),
            mn.unsqueeze(1),
            kv_dim,
        ).squeeze(1)

    def _decompress_all_history_kv(
        self,
        *,
        layer_idx: int,
        compressor_up: Optional[nn.Module],
        bs: int,
        k_dim: int,
    ):
        if self.config.use_compression:
            if compressor_up is None:
                raise ValueError("compressor_up is required to reconstruct compressed history.")
            comp_kv = self.comp_kv_cache[layer_idx]
            bases = self.bases_cache[layer_idx]
            # `comp_kv` is stored per token while `bases` is stored per chunk.
            # Expand chunk-level bases back to token resolution before reconstruction.
            bases = bases.repeat_interleave(self.config.compressor_token_group_size, dim=1)[:, : comp_kv.shape[1]]
            recon_kv = (compressor_up(comp_kv) + bases).view(bs, -1, 2, k_dim)
            return recon_kv[:, :, 0], recon_kv[:, :, 1]

        if self.config.kv_quant_bits == 4:
            history_kv = self._dequantize_kv_tokens(
                self.comp_kv_cache[layer_idx],
                self.comp_kv_scales[layer_idx],
                self.comp_kv_mins[layer_idx],
                2 * k_dim,
            )
            return history_kv.split(k_dim, dim=-1)

        history_k = self.comp_kv_cache[layer_idx]
        history_v = self.bases_cache[layer_idx]
        return history_k, history_v

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[dict[str, Any]] = None,
            compressor_down: Optional[nn.Module] = None,
            compressor_up: Optional[nn.Module] = None,
    ):
        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is None:
            raise NotImplementedError

        # key shape -> bs, seq_len, k_dim
        bs, q_len, k_dim = key_states.shape
        token_pos, visual_mask, token_valid_mask = self._extract_current_metadata(key_states, cache_kwargs)

        if layer_idx not in self.buffer_key_cache:
            self._init_pad_aware_layer_cache(
                layer_idx=layer_idx,
                key_states=key_states,
                value_states=value_states,
                token_pos=token_pos,
                visual_mask=visual_mask,
            )
        self._fill_sink_and_append_buffer(
            layer_idx=layer_idx,
            key_states=key_states,
            value_states=value_states,
            token_pos=token_pos,
            visual_mask=visual_mask,
            token_valid_mask=token_valid_mask,
        )

        # 初始化一些index
        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        sink_idx = self.sink_pos_cache[layer_idx]
        buffer_idx = self.buffer_pos_cache[layer_idx]
        
        if layer_idx in self.full_attn_layers:
            # 做full attn的layer不需要压缩kv cache
            return self._build_cache_view(
                [self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]],
                [self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]],
                [sink_idx, buffer_idx],
                layer_idx=layer_idx,
            )

        # 下面是 sparse attn layers :
        # 当前传入的 key value states 是准确的，没必要传回一个重建的。而且token idx也确实不会包含。所以先拿到返回值，再做压缩。
        this_response = None
        # 根据token idx重建所需kv cache
        if layer_idx in self.comp_kv_cache:
            # decode阶段，或是 chunk prefill 的加速开启
            if self.layer_to_full_layer_idx[layer_idx] in self.top_token_idx:
                rel_token_idx = self.top_token_idx[self.layer_to_full_layer_idx[layer_idx]]  # bs, num_imp_tokens
                if self.config.use_compression:
                    comp_kv, bases = self.comp_kv_cache[layer_idx], self.bases_cache[layer_idx]

                    imp_kv = comp_kv.gather(1, rel_token_idx[:, :, None].expand(-1, -1, comp_kv.shape[-1]))
                    bases_idx = rel_token_idx // self.config.compressor_token_group_size
                    imp_bases = bases.gather(1, bases_idx[:, :, None].expand(-1, -1, bases.shape[-1]))
                    if os.getenv('REMOVE_COMP'):
                        recon_kv = imp_bases.view(bs, -1, 2, k_dim)
                    elif os.getenv('REMOVE_REF'):
                        recon_kv = compressor_up(imp_kv).view(bs, -1, 2, k_dim)
                    else:
                        recon_kv = (compressor_up(imp_kv) + imp_bases).view(bs, -1, 2, k_dim)
                    recon_k, recon_v = recon_kv[:, :, 0], recon_kv[:, :, 1]
                else:
                    if self.config.kv_quant_bits == 4:
                        packed_kv = self.comp_kv_cache[layer_idx]
                        scales = self.comp_kv_scales[layer_idx]
                        mins = self.comp_kv_mins[layer_idx]
                        imp_packed_kv = packed_kv.gather(
                            1, rel_token_idx[:, :, None].expand(-1, -1, packed_kv.shape[-1])
                        )
                        imp_scales = scales.gather(
                            1, rel_token_idx[:, :, None].expand(-1, -1, scales.shape[-1])
                        )
                        imp_mins = mins.gather(
                            1, rel_token_idx[:, :, None].expand(-1, -1, mins.shape[-1])
                        )
                        imp_kv = self._dequantize_kv_tokens(imp_packed_kv, imp_scales, imp_mins, 2 * k_dim)
                        recon_k, recon_v = imp_kv.split(k_dim, dim=-1)
                    else:
                        # tensor的定义略有变化
                        history_k, history_v = self.comp_kv_cache[layer_idx], self.bases_cache[layer_idx]
                        recon_k = history_k.gather(1, rel_token_idx[:, :, None].expand(-1, -1, history_k.shape[-1]))
                        recon_v = history_v.gather(1, rel_token_idx[:, :, None].expand(-1, -1, history_v.shape[-1]))

                comp_pos = self.comp_pos_cache[layer_idx].gather(1, rel_token_idx)
                this_response = self._build_cache_view(
                    [self.sink_key_cache[layer_idx], recon_k, self.buffer_key_cache[layer_idx]],
                    [self.sink_value_cache[layer_idx], recon_v, self.buffer_value_cache[layer_idx]],
                    [sink_idx, comp_pos, buffer_idx],
                    layer_idx=layer_idx,
                )
            else:
                # Standalone DeltaKV: if no external selector provided token ids, reconstruct all compressed history.
                history_k, history_v = self._decompress_all_history_kv(
                    layer_idx=layer_idx,
                    compressor_up=compressor_up,
                    bs=bs,
                    k_dim=k_dim,
                )
                this_response = self._build_cache_view(
                    [self.sink_key_cache[layer_idx], history_k, self.buffer_key_cache[layer_idx]],
                    [self.sink_value_cache[layer_idx], history_v, self.buffer_value_cache[layer_idx]],
                    [sink_idx, self.comp_pos_cache[layer_idx], buffer_idx],
                    layer_idx=layer_idx,
                )
        else:
            this_response = self._build_cache_view(
                [self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]],
                [self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]],
                [sink_idx, buffer_idx],
                layer_idx=layer_idx,
            )

        # 如果buffer内长度超过2倍size，做压缩，只剩下 tail_token_size个token
        if buffer_len >= self.tail_token_size * 2:
            if self.visual_token_prune_only:
                if bs != 1:
                    raise NotImplementedError("visual_token_prune_only currently supports batch_size=1.")
                valid_buffer_len = int(self._buffer_valid_mask(layer_idx).sum(dim=1).max().item())
                candidate_len = max(0, valid_buffer_len - self.tail_token_size)
                compress_idx, drop_idx = self._select_visual_store_and_drop_indices(
                    layer_idx=layer_idx,
                    candidate_len=candidate_len,
                    device=key_states.device,
                )
                if self.config.use_compression:
                    usable_len = (compress_idx.numel() // self.config.compressor_token_group_size) * self.config.compressor_token_group_size
                    compress_idx = compress_idx[:usable_len]
                if drop_idx.numel() == 0:
                    return this_response
                keep_mask = self._buffer_valid_mask(layer_idx)
                keep_mask[:, drop_idx] = False
                _key = self.buffer_key_cache[layer_idx].index_select(1, compress_idx)
                _val = self.buffer_value_cache[layer_idx].index_select(1, compress_idx)
                _pos = self.buffer_pos_cache[layer_idx].index_select(1, compress_idx)
                self._filter_buffer_by_mask(layer_idx, keep_mask)
            else:
                compress_lens = self._compressible_lengths_from_buffer(layer_idx)
                if self.config.use_compression:
                    group_size = max(1, int(self.config.compressor_token_group_size))
                    compress_lens = torch.div(compress_lens, group_size, rounding_mode="floor") * group_size
                max_compress_len = int(compress_lens.max().item()) if compress_lens.numel() else 0
                if max_compress_len == 0:
                    return this_response
                compress_mask = self._first_valid_buffer_mask(layer_idx, compress_lens)
                _key, _val, _pos = self._pack_buffer_by_mask(layer_idx, compress_mask)

                keep_mask = self._buffer_valid_mask(layer_idx) & ~compress_mask
                self._filter_buffer_by_mask(layer_idx, keep_mask)

            if _pos.numel() > 0:

                bs, _len, k_dim = _key.shape

                if self.config.use_compression:
                    to_be_compress = torch.cat([_key, _val], dim=-1)
                    comp_kv, bases = self.compress(to_be_compress, compressor_down)

                    if layer_idx not in self.comp_kv_cache:
                        self.comp_kv_cache[layer_idx], self.bases_cache[layer_idx] = comp_kv, bases
                        self.comp_pos_cache[layer_idx] = _pos
                    else:
                        self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], comp_kv], dim=1)
                        self.bases_cache[layer_idx] = torch.cat([self.bases_cache[layer_idx], bases], dim=1)
                        self.comp_pos_cache[layer_idx] = torch.cat([self.comp_pos_cache[layer_idx], _pos], dim=1)
                else:
                    if self.config.kv_quant_bits == 4:
                        packed_kv, scale, mn = self._quantize_kv_tokens(torch.cat([_key, _val], dim=-1))
                        if layer_idx not in self.comp_kv_cache:
                            self.comp_kv_cache[layer_idx] = packed_kv
                            self.comp_kv_scales[layer_idx] = scale
                            self.comp_kv_mins[layer_idx] = mn
                            self.comp_pos_cache[layer_idx] = _pos
                        else:
                            self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], packed_kv], dim=1)
                            self.comp_kv_scales[layer_idx] = torch.cat([self.comp_kv_scales[layer_idx], scale], dim=1)
                            self.comp_kv_mins[layer_idx] = torch.cat([self.comp_kv_mins[layer_idx], mn], dim=1)
                            self.comp_pos_cache[layer_idx] = torch.cat([self.comp_pos_cache[layer_idx], _pos], dim=1)
                    else:
                        if layer_idx not in self.comp_kv_cache:
                            self.comp_kv_cache[layer_idx], self.bases_cache[layer_idx] = _key, _val
                            self.comp_pos_cache[layer_idx] = _pos
                        else:
                            self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], _key], dim=1)
                            self.bases_cache[layer_idx] = torch.cat([self.bases_cache[layer_idx], _val], dim=1)
                            self.comp_pos_cache[layer_idx] = torch.cat([self.comp_pos_cache[layer_idx], _pos], dim=1)

        return this_response

    @staticmethod
    def from_legacy_cache(cache):
        raise NotImplementedError


class ClusterCompressedKVCache(CompressedKVCache):
    def __init__(self, config: KVQwen2Config) -> None:
        super().__init__(config)
        log_once("💡💡💡 ClusterCompressedKVCache", 'INFO')
        self.token_father_idx = {}
        # 为量化准备
        self.comp_kv_scales = {}
        self.comp_kv_mins = {}
        # Dynamic prototype sampling: keep a per-sequence cursor so stride can grow with
        # absolute position without "resetting" at chunk boundaries.
        self.bases_valid_cache = {}
        self._cluster_next_center_abs_pos = None  # type: Optional[int]
        self._cluster_center_plan_cache_key = None  # type: Optional[tuple[int, int]]
        self._cluster_center_plan_cache_val = None  # type: Optional[torch.Tensor]

    def _get_dynamic_center_rel(
        self,
        *,
        abs_start_pos: int,
        seq_len: int,
        base_step: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Return center indices (relative to this block) for dynamic stride sampling.

        For stride_alpha <= 0, returns the legacy fixed-stride centers and does not
        touch the cursor state.

        For stride_alpha > 0, we maintain a cursor `self._cluster_next_center_abs_pos`
        so the sampling is continuous across blocks (sublinear growth in #centers).
        The computed plan is cached per (abs_start_pos, seq_len) so multiple layers
        sharing the same block reuse the same centers without advancing the cursor.
        """
        stride_alpha = float(getattr(self.config, "stride_alpha", 0.0) or 0.0)
        if stride_alpha <= 0.0:
            center_rel = torch.arange(0, seq_len, base_step, device=device, dtype=torch.long)
            if os.getenv("DEBUG"):
                print(
                    "[DeltaKV DEBUG] center_plan "
                    f"alpha={stride_alpha} abs_start={abs_start_pos} seq_len={seq_len} "
                    f"base_step={base_step} ref_count={center_rel.numel()} sink={self.sink_size}",
                    flush=True,
                )
            return center_rel

        if self._cluster_next_center_abs_pos is None:
            self._cluster_next_center_abs_pos = int(self.sink_size)

        cache_key = (int(abs_start_pos), int(seq_len))
        if (
            self._cluster_center_plan_cache_key == cache_key
            and self._cluster_center_plan_cache_val is not None
        ):
            return self._cluster_center_plan_cache_val

        sink = int(self.sink_size)
        end = int(abs_start_pos) + int(seq_len)
        pos = int(self._cluster_next_center_abs_pos)
        if pos < int(abs_start_pos):
            pos = int(abs_start_pos)

        rel_idx: list[int] = []
        while pos < end:
            rel_idx.append(pos - int(abs_start_pos))
            t = max(0, pos - sink)
            step = int(base_step) + int(stride_alpha * float(t))
            if step < 1:
                step = 1
            pos += step

        self._cluster_next_center_abs_pos = pos
        center_rel = (
            torch.tensor(rel_idx, device=device, dtype=torch.long)
            if rel_idx
            else torch.empty((0,), device=device, dtype=torch.long)
        )
        if os.getenv("DEBUG"):
            print(
                "[DeltaKV DEBUG] center_plan "
                f"alpha={stride_alpha} abs_start={abs_start_pos} seq_len={seq_len} "
                f"base_step={base_step} ref_count={center_rel.numel()} sink={self.sink_size} "
                f"next_center_abs_pos={self._cluster_next_center_abs_pos}",
                flush=True,
            )
        self._cluster_center_plan_cache_key = cache_key
        self._cluster_center_plan_cache_val = center_rel
        return center_rel

    @staticmethod
    def _metric_l2(kv_states, all_centers, use_kv=False):
        if not use_kv:
            k_dim = kv_states.shape[-1] // 2
            kv_states = kv_states[..., :k_dim]
            all_centers = all_centers[..., :k_dim]
        if kv_states.numel() == 0 or all_centers.numel() == 0:
            return kv_states.new_empty(kv_states.shape[:-1] + (all_centers.shape[-2],))

        # Match Sparse-vLLM runtime: for fixed a, argmin ||a-b||^2 is equivalent
        # to argmax (2 * a·b - ||b||^2). We only need ranking for top-k selection.
        dot = torch.matmul(kv_states, all_centers.transpose(-1, -2))
        center_norm = (all_centers * all_centers).sum(dim=-1, dtype=torch.float32).to(dot.dtype)
        return dot.mul(2.0).sub_(center_norm.unsqueeze(-2))

    @staticmethod
    def _metric_dot(kv_states, all_centers, use_kv=False):
        if not use_kv:
            k_dim = kv_states.shape[-1] // 2
            kv_states = kv_states[..., :k_dim]
            all_centers = all_centers[..., :k_dim]
        return torch.matmul(kv_states, all_centers.transpose(-1, -2))

    @staticmethod
    def _metric_cosine(kv_states, all_centers, use_kv=False):
        if not use_kv:
            k_dim = kv_states.shape[-1] // 2
            kv_states = kv_states[..., :k_dim]
            all_centers = all_centers[..., :k_dim]
        kv_norm = F.normalize(kv_states, p=2, dim=-1)
        center_norm = F.normalize(all_centers, p=2, dim=-1)
        return torch.matmul(kv_norm, center_norm.transpose(-1, -2))

    def compress(self, kv_states, compressor_down, existing_centers=None, *, abs_start_pos: Optional[int] = None):
        bs, seq_len, kv_dim = kv_states.shape
        cluster_step = max(1, int(1 / self.config.cluster_ratio))
        stride_alpha = float(getattr(self.config, "stride_alpha", 0.0) or 0.0)
        if abs_start_pos is None:
            abs_start_pos = 0

        # 1. 选择聚类中心 (Prototypes)
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

        # 2. 计算分配评分
        metric_type = self.config.cluster_metric
        use_kv = self.config.cluster_on_kv
        
        if metric_type == 'l2':
            scores = self._metric_l2(kv_states, all_centers, use_kv=use_kv)
        elif metric_type == 'dot':
            scores = self._metric_dot(kv_states, all_centers, use_kv=use_kv)
        elif metric_type == 'cosine':
            scores = self._metric_cosine(kv_states, all_centers, use_kv=use_kv)
        else:
            raise ValueError(f"Unknown metric type: {metric_type}")

        # 3. 应用因果掩码 (防止匹配到当前块内尚未采样出的中心)
        # rows: 当前 token 在 kv_states 中的索引 (0 ~ seq_len-1)
        # cols: new_centers 在 kv_states 中采样出来的相对索引 (dynamic 或 0, step, 2*step, ...)
        rows = torch.arange(seq_len, device=kv_states.device).view(-1, 1)
        if center_rel is None:
            cols = torch.arange(num_new, device=kv_states.device).view(1, -1) * cluster_step
        else:
            cols = center_rel.to(torch.long).view(1, -1)
        mask_new = (cols <= rows) # (seq_len, num_new)
        
        # 现有的中心 (existing_centers) 都在过去，全部可见
        mask_existing = torch.ones((seq_len, num_existing), device=kv_states.device, dtype=torch.bool)
        full_mask = torch.cat([mask_existing, mask_new], dim=1) # (seq_len, num_all)
        
        scores = scores.masked_fill(~full_mask.unsqueeze(0), float('-inf'))

        # 4. 选取 Top-K 中心
        k = self.config.get_cluster_neighbor_count()
        topk_scores, topk_indices = torch.topk(scores, k=min(k, all_centers.shape[1]), dim=-1)

        # 5. 计算参考基 (均值)
        # indices shape: (bs, seq_len * k, kv_dim)
        indices = topk_indices.view(bs, -1)[:, :, None].expand(-1, -1, kv_dim)
        gathered_fathers = all_centers.gather(1, indices).view(bs, seq_len, -1, kv_dim).mean(dim=2)

        # 6. 计算残差并压缩
        if os.getenv('REMOVE_COMP'):
            comp_kv = torch.zeros((bs, seq_len, self.config.kv_compressed_size), device=kv_states.device, dtype=kv_states.dtype)
        elif os.getenv('REMOVE_REF'):
            comp_kv = compressor_down(kv_states)
        else:
            comp_kv = compressor_down(kv_states) - compressor_down(gathered_fathers)

        # 7. 量化存储 (PTQ)
        if self.config.kv_quant_bits == 4:
            # comp_kv shape: (bs, seq_len, dim)
            # triton_quantize_and_pack_along_last_dim expects 4D (B, nh, D, T)
            comp_kv, scale, mn = triton_quantize_and_pack_along_last_dim(comp_kv.unsqueeze(1), comp_kv.shape[-1], 4)
            # result shapes: (bs, 1, seq_len, dim//8), (bs, 1, seq_len, 1), (bs, 1, seq_len, 1)
            return comp_kv.squeeze(1), all_centers, topk_indices, scale.squeeze(1), mn.squeeze(1)

        return comp_kv, all_centers, topk_indices, None, None

    def _reconstruct_selected_cluster_tokens(
        self,
        *,
        layer_idx: int,
        token_idx: torch.Tensor,
        compressor_up: Optional[nn.Module],
        bs: int,
        k_dim: int,
    ):
        if compressor_up is None:
            raise ValueError("compressor_up is required to reconstruct clustered compressed history.")

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
            imp_comp_kv = unpack_4bit_to_16bit(
                imp_comp_kv.unsqueeze(1),
                imp_scales.unsqueeze(1),
                imp_mins.unsqueeze(1),
                self.config.kv_compressed_size,
            ).squeeze(1)

        imp_topk_idx = topk_father_idx.gather(1, token_idx[:, :, None].expand(-1, -1, k))
        flat_idx = imp_topk_idx.view(bs, -1)[:, :, None].expand(-1, -1, bases.shape[-1])
        imp_bases = bases.gather(1, flat_idx).view(bs, token_idx.shape[1], k, -1).mean(dim=2)

        if os.getenv('REMOVE_COMP'):
            recon_kv = imp_bases.view(bs, -1, 2, k_dim)
        elif os.getenv('REMOVE_REF'):
            recon_kv = compressor_up(imp_comp_kv).view(bs, -1, 2, k_dim)
        else:
            recon_kv = (compressor_up(imp_comp_kv) + imp_bases).view(bs, -1, 2, k_dim)
        return recon_kv[:, :, 0], recon_kv[:, :, 1]

    def _reconstruct_selected_cluster_delta_quant_tokens(
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

        residual = comp_kv.gather(1, token_idx[:, :, None].expand(-1, -1, comp_kv.shape[-1]))
        if self.config.kv_quant_bits == 4:
            scales = self.comp_kv_scales[layer_idx]
            mins = self.comp_kv_mins[layer_idx]
            residual_scales = scales.gather(1, token_idx[:, :, None].expand(-1, -1, scales.shape[-1]))
            residual_mins = mins.gather(1, token_idx[:, :, None].expand(-1, -1, mins.shape[-1]))
            residual = self._dequantize_kv_tokens(residual, residual_scales, residual_mins, 2 * k_dim)

        father_idx = topk_father_idx.gather(1, token_idx[:, :, None].expand(-1, -1, k))
        flat_idx = father_idx.reshape(bs, -1)[:, :, None].expand(-1, -1, bases.shape[-1])
        refs = bases.gather(1, flat_idx).view(bs, token_idx.shape[1], k, -1).mean(dim=2)
        recon_kv = (residual + refs).view(bs, -1, 2, k_dim)
        return recon_kv[:, :, 0], recon_kv[:, :, 1]

    def _reconstruct_all_cluster_tokens(
        self,
        *,
        layer_idx: int,
        compressor_up: Optional[nn.Module],
        bs: int,
        k_dim: int,
    ):
        if compressor_up is None:
            raise ValueError("compressor_up is required to reconstruct clustered compressed history.")

        comp_kv = self.comp_kv_cache[layer_idx]
        bases = self.bases_cache[layer_idx]
        topk_father_idx = self.token_father_idx[layer_idx]
        num_hist = comp_kv.shape[1]
        k = topk_father_idx.shape[-1]

        if self.config.kv_quant_bits == 4:
            comp_kv = unpack_4bit_to_16bit(
                comp_kv.unsqueeze(1),
                self.comp_kv_scales[layer_idx].unsqueeze(1),
                self.comp_kv_mins[layer_idx].unsqueeze(1),
                self.config.kv_compressed_size,
            ).squeeze(1)

        flat_idx = topk_father_idx.view(bs, -1)[:, :, None].expand(-1, -1, bases.shape[-1])
        all_bases = bases.gather(1, flat_idx).view(bs, num_hist, k, -1).mean(dim=2)

        if os.getenv('REMOVE_COMP'):
            recon_kv = all_bases.view(bs, -1, 2, k_dim)
        elif os.getenv('REMOVE_REF'):
            recon_kv = compressor_up(comp_kv).view(bs, -1, 2, k_dim)
        else:
            recon_kv = (compressor_up(comp_kv) + all_bases).view(bs, -1, 2, k_dim)
        return recon_kv[:, :, 0], recon_kv[:, :, 1]

    def _reconstruct_all_cluster_delta_quant_tokens(
        self,
        *,
        layer_idx: int,
        bs: int,
        k_dim: int,
    ):
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
        topk_father_idx = self.token_father_idx[layer_idx]
        num_hist = residual.shape[1]
        k = topk_father_idx.shape[-1]
        flat_idx = topk_father_idx.reshape(bs, -1)[:, :, None].expand(-1, -1, bases.shape[-1])
        refs = bases.gather(1, flat_idx).view(bs, num_hist, k, -1).mean(dim=2)
        recon_kv = (residual + refs).view(bs, -1, 2, k_dim)
        return recon_kv[:, :, 0], recon_kv[:, :, 1]

    def _compress_cluster_delta_quant(
        self,
        kv_states: torch.Tensor,
        valid_mask: torch.Tensor,
        existing_centers: Optional[torch.Tensor] = None,
        existing_center_valid: Optional[torch.Tensor] = None,
        *,
        abs_start_pos: Optional[int] = None,
    ):
        bs, seq_len, kv_dim = kv_states.shape
        cluster_step = max(1, int(1 / self.config.cluster_ratio))
        stride_alpha = float(getattr(self.config, "stride_alpha", 0.0) or 0.0)
        if abs_start_pos is None:
            abs_start_pos = 0

        if stride_alpha <= 0.0:
            center_rel = torch.arange(0, seq_len, cluster_step, device=kv_states.device, dtype=torch.long)
        else:
            center_rel = self._get_dynamic_center_rel(
                abs_start_pos=int(abs_start_pos),
                seq_len=int(seq_len),
                base_step=int(cluster_step),
                device=kv_states.device,
            )
        new_centers = kv_states.index_select(1, center_rel).contiguous()
        new_center_valid = valid_mask.index_select(1, center_rel) if center_rel.numel() else valid_mask.new_zeros((bs, 0))

        if existing_centers is None:
            all_centers = new_centers
            all_center_valid = new_center_valid
            num_existing = 0
        else:
            all_centers = torch.cat([existing_centers, new_centers], dim=1)
            if existing_center_valid is None:
                existing_center_valid = torch.ones(
                    existing_centers.shape[:2],
                    device=existing_centers.device,
                    dtype=torch.bool,
                )
            all_center_valid = torch.cat([existing_center_valid.to(new_center_valid.device), new_center_valid], dim=1)
            num_existing = existing_centers.shape[1]

        if all_centers.shape[1] == 0:
            raise RuntimeError("Delta-quant clustering has no reference centers.")

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

        num_new = new_centers.shape[1]
        rows = torch.arange(seq_len, device=kv_states.device).view(-1, 1)
        cols = center_rel.to(torch.long).view(1, -1)
        mask_new = cols <= rows if num_new else torch.empty((seq_len, 0), device=kv_states.device, dtype=torch.bool)
        mask_existing = torch.ones((seq_len, num_existing), device=kv_states.device, dtype=torch.bool)
        causal_center_mask = torch.cat([mask_existing, mask_new], dim=1)
        scores = scores.masked_fill(~causal_center_mask.unsqueeze(0), float("-inf"))
        scores = scores.masked_fill(~all_center_valid[:, None, :], float("-inf"))

        k = min(self.config.get_cluster_neighbor_count(), all_centers.shape[1])
        safe_scores = scores.masked_fill(~valid_mask[:, :, None], 0.0)
        topk_indices = torch.topk(safe_scores, k=k, dim=-1).indices

        indices = topk_indices.reshape(bs, -1)[:, :, None].expand(-1, -1, kv_dim)
        refs = all_centers.gather(1, indices).view(bs, seq_len, k, kv_dim).mean(dim=2)
        residual = kv_states - refs
        residual = residual.masked_fill(~valid_mask[:, :, None], 0.0)

        if self.config.kv_quant_bits == 4:
            packed, scale, mn = self._quantize_kv_tokens(residual)
            return packed, all_centers, all_center_valid, topk_indices, scale, mn
        return residual, all_centers, all_center_valid, topk_indices, None, None

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[dict[str, Any]] = None,
            compressor_down: Optional[nn.Module] = None,
            compressor_up: Optional[nn.Module] = None,
    ):
        assert self.config.use_cluster

        # Update the number of seen tokens
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        # Update the cache
        if key_states is None:
            raise NotImplementedError

        # key shape -> bs, seq_len, k_dim
        bs, q_len, k_dim = key_states.shape
        token_pos, visual_mask, token_valid_mask = self._extract_current_metadata(key_states, cache_kwargs)

        is_new_layer = layer_idx not in self.buffer_key_cache
        if is_new_layer:
            self._init_pad_aware_layer_cache(
                layer_idx=layer_idx,
                key_states=key_states,
                value_states=value_states,
                token_pos=token_pos,
                visual_mask=visual_mask,
            )
            # Initialize dynamic stride cursor after sink size is finalized.
            if self._cluster_next_center_abs_pos is None:
                self._cluster_next_center_abs_pos = int(self.sink_size)
        self._fill_sink_and_append_buffer(
            layer_idx=layer_idx,
            key_states=key_states,
            value_states=value_states,
            token_pos=token_pos,
            visual_mask=visual_mask,
            token_valid_mask=token_valid_mask,
        )
        if is_new_layer:
            # 把 sink tokens 也作为初始聚类中心
            self.bases_cache[layer_idx] = torch.cat([self.sink_key_cache[layer_idx], self.sink_value_cache[layer_idx]], dim=-1)
            self.bases_valid_cache[layer_idx] = self._valid_pos_mask(self.sink_pos_cache[layer_idx])

        # 初始化一些index
        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        sink_idx = self.sink_pos_cache[layer_idx]
        buffer_idx = self.buffer_pos_cache[layer_idx]
        
        if layer_idx in self.full_attn_layers:
            # 做full attn的layer不需要压缩kv cache
            return self._build_cache_view(
                [self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]],
                [self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]],
                [sink_idx, buffer_idx],
                layer_idx=layer_idx,
            )

        # 下面是 sparse attn layers :
        # 根据token idx重建所需kv cache
        if layer_idx in self.comp_kv_cache:
            if self.layer_to_full_layer_idx[layer_idx] in self.top_token_idx:
                token_idx = self.top_token_idx[self.layer_to_full_layer_idx[layer_idx]]  # bs, num_imp_tokens

                if self.config.use_compression:
                    recon_k, recon_v = self._reconstruct_selected_cluster_tokens(
                        layer_idx=layer_idx,
                        token_idx=token_idx,
                        compressor_up=compressor_up,
                        bs=bs,
                        k_dim=k_dim,
                    )
                else:
                    recon_k, recon_v = self._reconstruct_selected_cluster_delta_quant_tokens(
                        layer_idx=layer_idx,
                        token_idx=token_idx,
                        bs=bs,
                        k_dim=k_dim,
                    )

                comp_pos = self.comp_pos_cache[layer_idx].gather(1, token_idx)
                this_response = self._build_cache_view(
                    [self.sink_key_cache[layer_idx], recon_k, self.buffer_key_cache[layer_idx]],
                    [self.sink_value_cache[layer_idx], recon_v, self.buffer_value_cache[layer_idx]],
                    [sink_idx, comp_pos, buffer_idx],
                    layer_idx=layer_idx,
                )
            else:
                if self.config.use_compression:
                    recon_k, recon_v = self._reconstruct_all_cluster_tokens(
                        layer_idx=layer_idx,
                        compressor_up=compressor_up,
                        bs=bs,
                        k_dim=k_dim,
                    )
                else:
                    recon_k, recon_v = self._reconstruct_all_cluster_delta_quant_tokens(
                        layer_idx=layer_idx,
                        bs=bs,
                        k_dim=k_dim,
                    )
                this_response = self._build_cache_view(
                    [self.sink_key_cache[layer_idx], recon_k, self.buffer_key_cache[layer_idx]],
                    [self.sink_value_cache[layer_idx], recon_v, self.buffer_value_cache[layer_idx]],
                    [sink_idx, self.comp_pos_cache[layer_idx], buffer_idx],
                    layer_idx=layer_idx,
                )
        else:
            this_response = self._build_cache_view(
                [self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]],
                [self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]],
                [sink_idx, buffer_idx],
                layer_idx=layer_idx,
            )

        # 如果buffer内长度超过2倍size，做压缩
        if buffer_len >= self.tail_token_size * 2:
            if self.visual_token_prune_only:
                if bs != 1:
                    raise NotImplementedError("visual_token_prune_only currently supports batch_size=1.")
                valid_buffer_len = int(self._buffer_valid_mask(layer_idx).sum(dim=1).max().item())
                candidate_len = max(0, valid_buffer_len - self.tail_token_size)
                compress_idx, drop_idx = self._select_visual_store_and_drop_indices(
                    layer_idx=layer_idx,
                    candidate_len=candidate_len,
                    device=key_states.device,
                )
                if self.config.use_compression:
                    usable_len = (compress_idx.numel() // self.config.compressor_token_group_size) * self.config.compressor_token_group_size
                    compress_idx = compress_idx[:usable_len]
                if drop_idx.numel() == 0:
                    return this_response
                keep_mask = self._buffer_valid_mask(layer_idx)
                keep_mask[:, drop_idx] = False
                _key = self.buffer_key_cache[layer_idx].index_select(1, compress_idx)
                _val = self.buffer_value_cache[layer_idx].index_select(1, compress_idx)
                _pos = self.buffer_pos_cache[layer_idx].index_select(1, compress_idx)
                selected_valid_mask = self._valid_pos_mask(_pos)
                self._filter_buffer_by_mask(layer_idx, keep_mask)
            else:
                compress_lens = self._compressible_lengths_from_buffer(layer_idx)
                group_size = max(1, int(self.config.compressor_token_group_size))
                compress_lens = torch.div(compress_lens, group_size, rounding_mode="floor") * group_size
                max_compress_len = int(compress_lens.max().item()) if compress_lens.numel() else 0
                if max_compress_len == 0:
                    return this_response
                compress_mask = self._first_valid_buffer_mask(layer_idx, compress_lens)
                _key, _val, _pos = self._pack_buffer_by_mask(layer_idx, compress_mask)
                keep_mask = self._buffer_valid_mask(layer_idx) & ~compress_mask
                selected_valid_mask = (
                    torch.arange(_pos.shape[1], device=_pos.device).unsqueeze(0)
                    < compress_lens.to(device=_pos.device).unsqueeze(1)
                )
                self._filter_buffer_by_mask(layer_idx, keep_mask)

            if _pos.numel() > 0:
                bs, _len, k_dim = _key.shape
                to_be_compress = torch.cat([_key, _val], dim=-1)

                existing_centers = self.bases_cache.get(layer_idx, None)
                valid_pos = _pos[self._valid_pos_mask(_pos)]
                abs_start_pos = int(valid_pos.min().item()) if valid_pos.numel() else int(self.sink_size)
                if self.config.use_compression:
                    # compress 内部会处理量化逻辑
                    comp_kv, all_centers, father_idx, scale, mn = self.compress(
                        to_be_compress,
                        compressor_down,
                        existing_centers,
                        abs_start_pos=abs_start_pos,
                    )
                    self.bases_cache[layer_idx] = all_centers
                else:
                    comp_kv, all_centers, all_center_valid, father_idx, scale, mn = self._compress_cluster_delta_quant(
                        to_be_compress,
                        selected_valid_mask,
                        existing_centers,
                        self.bases_valid_cache.get(layer_idx),
                        abs_start_pos=abs_start_pos,
                    )
                    self.bases_cache[layer_idx] = all_centers
                    self.bases_valid_cache[layer_idx] = all_center_valid

                if layer_idx not in self.comp_kv_cache:
                    self.comp_kv_cache[layer_idx] = comp_kv
                    self.token_father_idx[layer_idx] = father_idx
                    self.comp_pos_cache[layer_idx] = _pos
                    if scale is not None:
                        self.comp_kv_scales[layer_idx] = scale
                        self.comp_kv_mins[layer_idx] = mn
                else:
                    self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], comp_kv], dim=1)
                    self.token_father_idx[layer_idx] = torch.cat([self.token_father_idx[layer_idx], father_idx], dim=1)
                    self.comp_pos_cache[layer_idx] = torch.cat([self.comp_pos_cache[layer_idx], _pos], dim=1)
                    if scale is not None:
                        self.comp_kv_scales[layer_idx] = torch.cat([self.comp_kv_scales[layer_idx], scale], dim=1)
                        self.comp_kv_mins[layer_idx] = torch.cat([self.comp_kv_mins[layer_idx], mn], dim=1)

        return this_response
