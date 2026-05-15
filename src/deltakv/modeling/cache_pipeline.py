from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache

from deltakv.configs.model_config_cls import parse_full_attn_layers
from sparsevllm.triton_kernel.quant import triton_quantize_and_pack_along_last_dim, unpack_4bit_to_16bit


DELTA_COMPRESSED_LATENT_WO_FULL = "delta_compressed_latent_wo_full"
DELTA_COMPRESSED_LATENT_W_FULL = "delta_compressed_latent_w_full"
DELTA_ORIGIN_WO_FULL = "delta_origin_wo_full"
DELTA_ORIGIN_W_FULL = "delta_origin_w_full"
HF_SPARSE_CACHE_OMNIKV = "omnikv"


def _bs1(key_states: torch.Tensor) -> None:
    if key_states.shape[0] != 1:
        raise NotImplementedError(
            "HF DeltaKV cache only supports batch_size=1 after the modeling refactor; "
            "use the Sparse-vLLM backend for batched inference."
        )


def _unsupported_non_cluster() -> None:
    raise NotImplementedError("HF DeltaKV modeling is cluster-only; non-cluster/chunk-ref paths were removed.")


class BaseCache(DynamicCache):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.tail_token_size = int(config.tail_token_size)
        self.sink_size = int(config.num_sink_tokens)
        self.top_token_idx: dict[int, torch.Tensor] = {}
        self.token_scores: dict[int, torch.Tensor] = {}
        self.num_prompt_tokens = None
        self._seen_tokens = 0

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return self._seen_tokens


class SnapKVCache(BaseCache):
    def __init__(self, config):
        super().__init__(config)
        self.is_pruned = False
        self.num_layers = config.num_hidden_layers

    @property
    def is_last_chunk(self):
        return self._seen_tokens == self.num_prompt_tokens and not self.is_pruned

    def delete_tokens(self, layer_idx, top_token_idx):
        if layer_idx == self.num_layers - 1:
            self.is_pruned = True
        if top_token_idx.dim() == 2:
            top_token_idx = top_token_idx.unsqueeze(1)
        bs, num_kv_heads_idx, _ = top_token_idx.shape
        kv_len = self.key_cache[layer_idx].shape[2]
        head_dim = self.key_cache[layer_idx].shape[3]
        num_kv_heads = self.key_cache[layer_idx].shape[1]
        if num_kv_heads_idx == 1 and num_kv_heads > 1:
            top_token_idx = top_token_idx.expand(-1, num_kv_heads, -1)
            num_kv_heads_idx = num_kv_heads
        token_idx = top_token_idx + self.sink_size
        device = token_idx.device
        sink_idx = torch.arange(self.sink_size, device=device)[None, None, :].expand(bs, num_kv_heads_idx, -1)
        recent_idx = torch.arange(kv_len - self.tail_token_size, kv_len, device=device)[None, None, :].expand(bs, num_kv_heads_idx, -1)
        gather_idx = torch.cat([sink_idx, token_idx, recent_idx], dim=2).unsqueeze(-1).expand(-1, -1, -1, head_dim)
        self.key_cache[layer_idx] = self.key_cache[layer_idx].gather(index=gather_idx, dim=2)
        self.value_cache[layer_idx] = self.value_cache[layer_idx].gather(index=gather_idx, dim=2)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None, compressor_down=None, compressor_up=None):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[2]
        if len(self.key_cache) <= layer_idx:
            for _ in range(len(self.key_cache), layer_idx):
                self.key_cache.append(torch.tensor([]))
                self.value_cache.append(torch.tensor([]))
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        elif not self.key_cache[layer_idx].numel():
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)
        return self.key_cache[layer_idx], self.value_cache[layer_idx]


class ClusterCachePipeline(BaseCache):
    def __init__(self, config, *, cache_impl: str = DELTA_COMPRESSED_LATENT_WO_FULL) -> None:
        if not getattr(config, "use_cluster", False):
            _unsupported_non_cluster()
        super().__init__(config)
        self.cache_impl = cache_impl
        self.full_attn_layers = parse_full_attn_layers(config.full_attn_layers)
        config.full_attn_layers = self.full_attn_layers
        self.layer_to_full_layer_idx = {}
        last_full = None
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.full_attn_layers:
                last_full = layer_idx
            self.layer_to_full_layer_idx[layer_idx] = last_full

        self.sink_key_cache = {}
        self.sink_value_cache = {}
        self.sink_pos_cache = {}
        self.sink_filled_count = {}
        self.buffer_key_cache = {}
        self.buffer_value_cache = {}
        self.buffer_pos_cache = {}
        self.buffer_visual_mask_cache = {}
        self.comp_kv_cache = {}
        self.comp_pos_cache = {}
        self.comp_kv_scales = {}
        self.comp_kv_mins = {}
        self.bases_cache = {}
        self.token_father_idx = {}
        self.cos = None
        self.sin = None
        self._cluster_next_center_abs_pos = None
        self._cluster_center_plan_cache_key = None
        self._cluster_center_plan_cache_val = None

    def _origin_codec(self) -> bool:
        return (
            self.cache_impl in {DELTA_ORIGIN_WO_FULL, DELTA_ORIGIN_W_FULL}
            or not getattr(self.config, "use_compression", False)
        )

    def _compress_full_layers(self) -> bool:
        return self.cache_impl in {
            DELTA_COMPRESSED_LATENT_W_FULL,
            DELTA_ORIGIN_W_FULL,
        }

    def _should_compress_layer(self, layer_idx: int) -> bool:
        if self.cache_impl == DELTA_ORIGIN_W_FULL:
            return True
        return layer_idx not in self.full_attn_layers or self._compress_full_layers()

    def _token_pos(self, key_states: torch.Tensor, cache_kwargs: Optional[dict]) -> torch.Tensor:
        q_len = key_states.shape[1]
        cache_position = (cache_kwargs or {}).get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(self._seen_tokens, self._seen_tokens + q_len, device=key_states.device)
        cache_position = cache_position.to(device=key_states.device, dtype=torch.long)
        if cache_position.dim() == 1:
            return cache_position.unsqueeze(0)
        return cache_position

    def _visual_mask(self, key_states: torch.Tensor, cache_kwargs: Optional[dict]) -> torch.Tensor:
        mask = (cache_kwargs or {}).get("deltakv_visual_token_mask")
        if mask is None:
            return torch.zeros(key_states.shape[:2], device=key_states.device, dtype=torch.bool)
        mask = mask.to(device=key_states.device, dtype=torch.bool)
        return mask.unsqueeze(0) if mask.dim() == 1 else mask

    def _ensure_layer(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, token_pos: torch.Tensor) -> None:
        if layer_idx in self.buffer_key_cache:
            return
        bs, _, k_dim = key_states.shape
        self.sink_key_cache[layer_idx] = key_states.new_zeros((bs, self.sink_size, k_dim))
        self.sink_value_cache[layer_idx] = value_states.new_zeros((bs, self.sink_size, k_dim))
        self.sink_pos_cache[layer_idx] = token_pos.new_empty((bs, 0))
        self.sink_filled_count[layer_idx] = 0
        self.buffer_key_cache[layer_idx] = key_states.new_empty((bs, 0, k_dim))
        self.buffer_value_cache[layer_idx] = value_states.new_empty((bs, 0, k_dim))
        self.buffer_pos_cache[layer_idx] = token_pos.new_empty((bs, 0))
        self.buffer_visual_mask_cache[layer_idx] = torch.empty((bs, 0), device=key_states.device, dtype=torch.bool)

    def _append(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, token_pos: torch.Tensor, visual_mask: torch.Tensor) -> None:
        filled = int(self.sink_filled_count[layer_idx])
        take = max(0, min(self.sink_size - filled, key_states.shape[1]))
        if take:
            end = filled + take
            self.sink_key_cache[layer_idx][:, filled:end] = key_states[:, :take]
            self.sink_value_cache[layer_idx][:, filled:end] = value_states[:, :take]
            self.sink_pos_cache[layer_idx] = torch.cat([self.sink_pos_cache[layer_idx], token_pos[:, :take]], dim=1)
            self.sink_filled_count[layer_idx] = end
        if take < key_states.shape[1]:
            self.buffer_key_cache[layer_idx] = torch.cat([self.buffer_key_cache[layer_idx], key_states[:, take:]], dim=1)
            self.buffer_value_cache[layer_idx] = torch.cat([self.buffer_value_cache[layer_idx], value_states[:, take:]], dim=1)
            self.buffer_pos_cache[layer_idx] = torch.cat([self.buffer_pos_cache[layer_idx], token_pos[:, take:]], dim=1)
            self.buffer_visual_mask_cache[layer_idx] = torch.cat([self.buffer_visual_mask_cache[layer_idx], visual_mask[:, take:]], dim=1)

    def _sink_kv(self, layer_idx: int) -> torch.Tensor:
        filled = int(self.sink_filled_count[layer_idx])
        return torch.cat(
            [
                self.sink_key_cache[layer_idx][:, :filled],
                self.sink_value_cache[layer_idx][:, :filled],
            ],
            dim=-1,
        )

    @staticmethod
    def _metric_l2(kv_states, all_centers, use_kv=False):
        if not use_kv:
            k_dim = kv_states.shape[-1] // 2
            kv_states = kv_states[..., :k_dim]
            all_centers = all_centers[..., :k_dim]
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
        return torch.matmul(F.normalize(kv_states, p=2, dim=-1), F.normalize(all_centers, p=2, dim=-1).transpose(-1, -2))

    def _center_rel(self, *, abs_start_pos: int, seq_len: int, base_step: int, device: torch.device) -> torch.Tensor:
        stride_alpha = float(getattr(self.config, "stride_alpha", 0.0) or 0.0)
        if stride_alpha <= 0.0:
            return torch.arange(0, seq_len, base_step, device=device, dtype=torch.long)
        if self._cluster_next_center_abs_pos is None:
            self._cluster_next_center_abs_pos = int(self.sink_size)
        key = (int(abs_start_pos), int(seq_len))
        if self._cluster_center_plan_cache_key == key and self._cluster_center_plan_cache_val is not None:
            return self._cluster_center_plan_cache_val
        end = int(abs_start_pos) + int(seq_len)
        pos = max(int(self._cluster_next_center_abs_pos), int(abs_start_pos))
        rel = []
        while pos < end:
            rel.append(pos - int(abs_start_pos))
            step = max(1, int(base_step) + int(stride_alpha * float(max(0, pos - int(self.sink_size)))))
            pos += step
        self._cluster_next_center_abs_pos = pos
        out = torch.tensor(rel, device=device, dtype=torch.long) if rel else torch.empty((0,), device=device, dtype=torch.long)
        self._cluster_center_plan_cache_key = key
        self._cluster_center_plan_cache_val = out
        return out

    def _cluster_refs(self, kv_states: torch.Tensor, existing_centers: torch.Tensor, *, abs_start_pos: int):
        bs, seq_len, kv_dim = kv_states.shape
        step = max(1, int(1 / self.config.cluster_ratio))
        center_rel = self._center_rel(abs_start_pos=abs_start_pos, seq_len=seq_len, base_step=step, device=kv_states.device)
        new_centers = kv_states.index_select(1, center_rel) if center_rel.numel() else kv_states[:, :0]
        all_centers = torch.cat([existing_centers, new_centers], dim=1) if existing_centers is not None else new_centers
        if all_centers.shape[1] == 0:
            raise RuntimeError("DeltaKV cluster compression has no reference centers.")
        metric = self.config.cluster_metric
        if metric == "l2":
            scores = self._metric_l2(kv_states, all_centers, use_kv=self.config.cluster_on_kv)
        elif metric == "dot":
            scores = self._metric_dot(kv_states, all_centers, use_kv=self.config.cluster_on_kv)
        elif metric == "cosine":
            scores = self._metric_cosine(kv_states, all_centers, use_kv=self.config.cluster_on_kv)
        else:
            raise ValueError(f"Unknown cluster_metric={metric!r}")
        rows = torch.arange(seq_len, device=kv_states.device).view(-1, 1)
        causal_new = center_rel.view(1, -1) <= rows
        causal = torch.cat([torch.ones((seq_len, existing_centers.shape[1]), device=kv_states.device, dtype=torch.bool), causal_new], dim=1)
        scores = scores.masked_fill(~causal.unsqueeze(0), float("-inf"))
        k = min(self.config.get_cluster_neighbor_count(), all_centers.shape[1])
        father_idx = torch.topk(scores, k=k, dim=-1).indices
        refs = all_centers.gather(1, father_idx.reshape(bs, -1)[:, :, None].expand(-1, -1, kv_dim)).view(bs, seq_len, k, kv_dim).mean(dim=2)
        return refs, all_centers, father_idx

    def _quantize(self, states: torch.Tensor):
        packed, scale, mn = triton_quantize_and_pack_along_last_dim(states.unsqueeze(1), states.shape[-1], 4)
        return packed.squeeze(1), scale.squeeze(1), mn.squeeze(1)

    def _dequantize(self, packed: torch.Tensor, scale: torch.Tensor, mn: torch.Tensor, dim: int):
        return unpack_4bit_to_16bit(packed.unsqueeze(1), scale.unsqueeze(1), mn.unsqueeze(1), dim).squeeze(1)

    def _store_history(self, layer_idx: int, key: torch.Tensor, value: torch.Tensor, pos: torch.Tensor, compressor_down: Optional[nn.Module]) -> None:
        if key.numel() == 0:
            return
        kv = torch.cat([key, value], dim=-1)
        existing = self.bases_cache.get(layer_idx)
        if existing is None:
            existing = self._sink_kv(layer_idx)
        refs, centers, father_idx = self._cluster_refs(kv, existing, abs_start_pos=int(pos[0, 0].item()))
        self.bases_cache[layer_idx] = centers
        if self._origin_codec():
            payload = kv - refs
            dequant_dim = kv.shape[-1]
        else:
            if compressor_down is None:
                raise ValueError("compressor_down is required for learned DeltaKV compression.")
            if os.getenv("REMOVE_COMP"):
                payload = torch.zeros(kv.shape[:-1] + (self.config.kv_compressed_size,), device=kv.device, dtype=kv.dtype)
            elif os.getenv("REMOVE_REF"):
                payload = compressor_down(kv)
            else:
                payload = compressor_down(kv) - compressor_down(refs)
            dequant_dim = self.config.kv_compressed_size

        scale = mn = None
        if self.config.kv_quant_bits == 4:
            payload, scale, mn = self._quantize(payload)
        if layer_idx in self.comp_kv_cache:
            self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], payload], dim=1)
            self.token_father_idx[layer_idx] = torch.cat([self.token_father_idx[layer_idx], father_idx], dim=1)
            self.comp_pos_cache[layer_idx] = torch.cat([self.comp_pos_cache[layer_idx], pos], dim=1)
            if scale is not None:
                self.comp_kv_scales[layer_idx] = torch.cat([self.comp_kv_scales[layer_idx], scale], dim=1)
                self.comp_kv_mins[layer_idx] = torch.cat([self.comp_kv_mins[layer_idx], mn], dim=1)
        else:
            self.comp_kv_cache[layer_idx] = payload
            self.token_father_idx[layer_idx] = father_idx
            self.comp_pos_cache[layer_idx] = pos
            if scale is not None:
                self.comp_kv_scales[layer_idx] = scale
                self.comp_kv_mins[layer_idx] = mn
        self._last_dequant_dim = dequant_dim

    def compress(self, kv_states: torch.Tensor, compressor_down: Optional[nn.Module], existing_centers: Optional[torch.Tensor]):
        refs, centers, father_idx = self._cluster_refs(kv_states, existing_centers, abs_start_pos=0)
        if self._origin_codec():
            payload = kv_states - refs
        else:
            if compressor_down is None:
                raise ValueError("compressor_down is required for learned DeltaKV compression.")
            payload = compressor_down(kv_states) - compressor_down(refs)
        scale = mn = None
        if self.config.kv_quant_bits == 4:
            payload, scale, mn = self._quantize(payload)
        return payload, centers, father_idx, scale, mn

    def _reconstruct(self, layer_idx: int, *, token_idx: Optional[torch.Tensor], compressor_up: Optional[nn.Module], k_dim: int):
        payload = self.comp_kv_cache[layer_idx]
        father_idx = self.token_father_idx[layer_idx]
        if token_idx is not None:
            payload = payload.gather(1, token_idx[:, :, None].expand(-1, -1, payload.shape[-1]))
            father_idx = father_idx.gather(1, token_idx[:, :, None].expand(-1, -1, father_idx.shape[-1]))
        dim = 2 * k_dim if self._origin_codec() else self.config.kv_compressed_size
        if self.config.kv_quant_bits == 4:
            scale = self.comp_kv_scales[layer_idx]
            mn = self.comp_kv_mins[layer_idx]
            if token_idx is not None:
                scale = scale.gather(1, token_idx[:, :, None].expand(-1, -1, scale.shape[-1]))
                mn = mn.gather(1, token_idx[:, :, None].expand(-1, -1, mn.shape[-1]))
            payload = self._dequantize(payload, scale, mn, dim)
        bases = self.bases_cache[layer_idx]
        k = father_idx.shape[-1]
        refs = bases.gather(1, father_idx.reshape(1, -1)[:, :, None].expand(-1, -1, bases.shape[-1])).view(1, father_idx.shape[1], k, -1).mean(dim=2)
        if self._origin_codec():
            recon = payload + refs
        else:
            if compressor_up is None:
                raise ValueError("compressor_up is required for learned DeltaKV reconstruction.")
            if os.getenv("REMOVE_COMP"):
                recon = refs
            elif os.getenv("REMOVE_REF"):
                recon = compressor_up(payload)
            else:
                recon = compressor_up(payload) + refs
        recon = recon.view(1, -1, 2, k_dim)
        return recon[:, :, 0], recon[:, :, 1]

    def _reconstruct_all_cluster_tokens(self, layer_idx: int, compressor_up: Optional[nn.Module], bs: int, k_dim: int):
        if bs != 1:
            raise NotImplementedError("HF DeltaKV baseline cache only supports batch_size=1.")
        return self._reconstruct(layer_idx, token_idx=None, compressor_up=compressor_up, k_dim=k_dim)

    def _view(self, layer_idx: int, compressor_up: Optional[nn.Module], k_dim: int):
        filled = int(self.sink_filled_count[layer_idx])
        keys = [self.sink_key_cache[layer_idx][:, :filled]]
        values = [self.sink_value_cache[layer_idx][:, :filled]]
        pos = [self.sink_pos_cache[layer_idx]]
        if layer_idx in self.comp_kv_cache:
            selector = self.layer_to_full_layer_idx.get(layer_idx)
            token_idx = self.top_token_idx.get(selector)
            recon_k, recon_v = self._reconstruct(layer_idx, token_idx=token_idx, compressor_up=compressor_up, k_dim=k_dim)
            keys.append(recon_k)
            values.append(recon_v)
            comp_pos = self.comp_pos_cache[layer_idx] if token_idx is None else self.comp_pos_cache[layer_idx].gather(1, token_idx)
            pos.append(comp_pos)
        keys.append(self.buffer_key_cache[layer_idx])
        values.append(self.buffer_value_cache[layer_idx])
        pos.append(self.buffer_pos_cache[layer_idx])
        return torch.cat(keys, dim=1), torch.cat(values, dim=1), torch.cat(pos, dim=1)

    def _flush(self, layer_idx: int, compressor_down: Optional[nn.Module]) -> None:
        if not self._should_compress_layer(layer_idx):
            return
        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        if buffer_len < self.tail_token_size * 2:
            return
        candidate_len = buffer_len - self.tail_token_size
        if getattr(self.config, "visual_token_prune_only", False):
            visual = self.buffer_visual_mask_cache[layer_idx][0, :candidate_len]
            idx = torch.nonzero(visual, as_tuple=False).flatten()
            if idx.numel() == 0:
                return
            keep_ratio = max(0.0, min(1.0, float(getattr(self.config, "visual_token_keep_ratio", 1.0) or 0.0)))
            keep = idx if keep_ratio >= 1.0 else idx[: max(1, int(round(idx.numel() * keep_ratio)))]
            drop = idx
            hist_key = self.buffer_key_cache[layer_idx].index_select(1, keep)
            hist_value = self.buffer_value_cache[layer_idx].index_select(1, keep)
            hist_pos = self.buffer_pos_cache[layer_idx].index_select(1, keep)
            keep_mask = torch.ones((buffer_len,), device=idx.device, dtype=torch.bool)
            keep_mask[drop] = False
        else:
            compress_len = (candidate_len // self.tail_token_size) * self.tail_token_size
            if compress_len <= 0:
                return
            hist_key = self.buffer_key_cache[layer_idx][:, :compress_len]
            hist_value = self.buffer_value_cache[layer_idx][:, :compress_len]
            hist_pos = self.buffer_pos_cache[layer_idx][:, :compress_len]
            keep_mask = torch.arange(buffer_len, device=hist_key.device) >= compress_len
        self.buffer_key_cache[layer_idx] = self.buffer_key_cache[layer_idx][:, keep_mask]
        self.buffer_value_cache[layer_idx] = self.buffer_value_cache[layer_idx][:, keep_mask]
        self.buffer_pos_cache[layer_idx] = self.buffer_pos_cache[layer_idx][:, keep_mask]
        self.buffer_visual_mask_cache[layer_idx] = self.buffer_visual_mask_cache[layer_idx][:, keep_mask]
        self._store_history(layer_idx, hist_key, hist_value, hist_pos, compressor_down)

    def get_compressed_positions(self, layer_idx: int, device: Optional[torch.device] = None):
        pos = self.comp_pos_cache.get(layer_idx)
        return None if pos is None else (pos.to(device=device) if device is not None else pos)

    def get_compressed_length(self, layer_idx: int) -> int:
        pos = self.get_compressed_positions(layer_idx)
        return 0 if pos is None else int(pos.shape[1])

    def get_observable_compressed_length(self, current_q_len: int) -> int:
        compressed_len = self._seen_tokens - self.tail_token_size - int(current_q_len) - self.sink_size
        if compressed_len <= 0:
            return 0
        return (compressed_len // self.tail_token_size) * self.tail_token_size

    def get_buffer_valid_lengths(self, layer_idx: int, device: Optional[torch.device] = None):
        if layer_idx not in self.buffer_pos_cache:
            return None
        out = torch.tensor([self.buffer_pos_cache[layer_idx].shape[1]], device=self.buffer_pos_cache[layer_idx].device)
        return out.to(device=device) if device is not None else out

    def get_buffer_candidate_positions(self, layer_idx: int, lengths: torch.Tensor, device: Optional[torch.device] = None):
        if layer_idx not in self.buffer_pos_cache:
            return None
        n = int(lengths.flatten()[0].item())
        pos = self.buffer_pos_cache[layer_idx][:, :n]
        valid = torch.ones_like(pos, dtype=torch.bool)
        if device is not None:
            pos = pos.to(device=device)
            valid = valid.to(device=device)
        return pos, valid

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None, compressor_down: Optional[nn.Module] = None, compressor_up: Optional[nn.Module] = None):
        _bs1(key_states)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[1]
        token_pos = self._token_pos(key_states, cache_kwargs)
        visual = self._visual_mask(key_states, cache_kwargs)
        self._ensure_layer(layer_idx, key_states, value_states, token_pos)
        self._append(layer_idx, key_states, value_states, token_pos, visual)
        if layer_idx not in self.bases_cache:
            self.bases_cache[layer_idx] = self._sink_kv(layer_idx)
        response = self._view(layer_idx, compressor_up, key_states.shape[-1])
        self._flush(layer_idx, compressor_down)
        return response


class OmniKVRawCache(BaseCache):
    """HF OmniKV cache: persistent raw sink/history/recent KV plus dynamic top-k views."""

    def __init__(self, config) -> None:
        super().__init__(config)
        self.full_attn_layers = parse_full_attn_layers(config.full_attn_layers)
        config.full_attn_layers = self.full_attn_layers
        self.layer_to_full_layer_idx = {}
        last_full = None
        for layer_idx in range(config.num_hidden_layers):
            if layer_idx in self.full_attn_layers:
                last_full = layer_idx
            self.layer_to_full_layer_idx[layer_idx] = last_full

        self.sink_key_cache = {}
        self.sink_value_cache = {}
        self.sink_pos_cache = {}
        self.sink_filled_count = {}
        self.history_key_cache = {}
        self.history_value_cache = {}
        self.history_pos_cache = {}
        self.buffer_key_cache = {}
        self.buffer_value_cache = {}
        self.buffer_pos_cache = {}
        self.cos = None
        self.sin = None

    def _token_pos(self, key_states: torch.Tensor, cache_kwargs: Optional[dict]) -> torch.Tensor:
        q_len = key_states.shape[1]
        cache_position = (cache_kwargs or {}).get("cache_position")
        if cache_position is None:
            cache_position = torch.arange(self._seen_tokens, self._seen_tokens + q_len, device=key_states.device)
        cache_position = cache_position.to(device=key_states.device, dtype=torch.long)
        return cache_position.unsqueeze(0) if cache_position.dim() == 1 else cache_position

    def _ensure_layer(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, token_pos: torch.Tensor) -> None:
        if layer_idx in self.buffer_key_cache:
            return
        bs, _, k_dim = key_states.shape
        self.sink_key_cache[layer_idx] = key_states.new_zeros((bs, self.sink_size, k_dim))
        self.sink_value_cache[layer_idx] = value_states.new_zeros((bs, self.sink_size, k_dim))
        self.sink_pos_cache[layer_idx] = token_pos.new_empty((bs, 0))
        self.sink_filled_count[layer_idx] = 0
        self.history_key_cache[layer_idx] = key_states.new_empty((bs, 0, k_dim))
        self.history_value_cache[layer_idx] = value_states.new_empty((bs, 0, k_dim))
        self.history_pos_cache[layer_idx] = token_pos.new_empty((bs, 0))
        self.buffer_key_cache[layer_idx] = key_states.new_empty((bs, 0, k_dim))
        self.buffer_value_cache[layer_idx] = value_states.new_empty((bs, 0, k_dim))
        self.buffer_pos_cache[layer_idx] = token_pos.new_empty((bs, 0))

    def _append(self, layer_idx: int, key_states: torch.Tensor, value_states: torch.Tensor, token_pos: torch.Tensor) -> None:
        filled = int(self.sink_filled_count[layer_idx])
        take = max(0, min(self.sink_size - filled, key_states.shape[1]))
        if take:
            end = filled + take
            self.sink_key_cache[layer_idx][:, filled:end] = key_states[:, :take]
            self.sink_value_cache[layer_idx][:, filled:end] = value_states[:, :take]
            self.sink_pos_cache[layer_idx] = torch.cat([self.sink_pos_cache[layer_idx], token_pos[:, :take]], dim=1)
            self.sink_filled_count[layer_idx] = end
        if take < key_states.shape[1]:
            self.buffer_key_cache[layer_idx] = torch.cat([self.buffer_key_cache[layer_idx], key_states[:, take:]], dim=1)
            self.buffer_value_cache[layer_idx] = torch.cat([self.buffer_value_cache[layer_idx], value_states[:, take:]], dim=1)
            self.buffer_pos_cache[layer_idx] = torch.cat([self.buffer_pos_cache[layer_idx], token_pos[:, take:]], dim=1)

    def _view(self, layer_idx: int):
        filled = int(self.sink_filled_count[layer_idx])
        keys = [self.sink_key_cache[layer_idx][:, :filled]]
        values = [self.sink_value_cache[layer_idx][:, :filled]]
        pos = [self.sink_pos_cache[layer_idx]]

        history_k = self.history_key_cache[layer_idx]
        if history_k.shape[1]:
            selector = self.layer_to_full_layer_idx.get(layer_idx)
            token_idx = None if layer_idx in self.full_attn_layers else self.top_token_idx.get(selector)
            if token_idx is None:
                keys.append(history_k)
                values.append(self.history_value_cache[layer_idx])
                pos.append(self.history_pos_cache[layer_idx])
            elif token_idx.numel():
                token_idx = token_idx.to(device=history_k.device, dtype=torch.long)
                gather = token_idx[:, :, None]
                keys.append(history_k.gather(1, gather.expand(-1, -1, history_k.shape[-1])))
                values.append(
                    self.history_value_cache[layer_idx].gather(
                        1, gather.expand(-1, -1, self.history_value_cache[layer_idx].shape[-1])
                    )
                )
                pos.append(self.history_pos_cache[layer_idx].gather(1, token_idx))

        keys.append(self.buffer_key_cache[layer_idx])
        values.append(self.buffer_value_cache[layer_idx])
        pos.append(self.buffer_pos_cache[layer_idx])
        return torch.cat(keys, dim=1), torch.cat(values, dim=1), torch.cat(pos, dim=1)

    def _flush(self, layer_idx: int) -> None:
        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        if buffer_len <= self.tail_token_size:
            return
        compress_len = buffer_len - self.tail_token_size
        if compress_len <= 0:
            return

        hist_key = self.buffer_key_cache[layer_idx][:, :compress_len]
        hist_value = self.buffer_value_cache[layer_idx][:, :compress_len]
        hist_pos = self.buffer_pos_cache[layer_idx][:, :compress_len]
        self.history_key_cache[layer_idx] = torch.cat([self.history_key_cache[layer_idx], hist_key], dim=1)
        self.history_value_cache[layer_idx] = torch.cat([self.history_value_cache[layer_idx], hist_value], dim=1)
        self.history_pos_cache[layer_idx] = torch.cat([self.history_pos_cache[layer_idx], hist_pos], dim=1)
        self.buffer_key_cache[layer_idx] = self.buffer_key_cache[layer_idx][:, compress_len:]
        self.buffer_value_cache[layer_idx] = self.buffer_value_cache[layer_idx][:, compress_len:]
        self.buffer_pos_cache[layer_idx] = self.buffer_pos_cache[layer_idx][:, compress_len:]

    def get_compressed_length(self, layer_idx: int) -> int:
        return 0 if layer_idx not in self.history_key_cache else int(self.history_key_cache[layer_idx].shape[1])

    def get_observable_compressed_length(self, current_q_len: int) -> int:
        compressed_len = self._seen_tokens - self.tail_token_size - int(current_q_len) - self.sink_size
        if compressed_len <= 0:
            return 0
        return int(compressed_len)

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None, compressor_down: Optional[nn.Module] = None, compressor_up: Optional[nn.Module] = None):
        del compressor_down, compressor_up
        _bs1(key_states)
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[1]
        token_pos = self._token_pos(key_states, cache_kwargs)
        self._ensure_layer(layer_idx, key_states, value_states, token_pos)
        self._append(layer_idx, key_states, value_states, token_pos)
        response = self._view(layer_idx)
        self._flush(layer_idx)
        return response


class DeltaCompressedLatentWoFullCache(ClusterCachePipeline):
    def __init__(self, config):
        super().__init__(config, cache_impl=DELTA_COMPRESSED_LATENT_WO_FULL)


class DeltaCompressedLatentWFullCache(ClusterCachePipeline):
    def __init__(self, config):
        super().__init__(config, cache_impl=DELTA_COMPRESSED_LATENT_W_FULL)


class DeltaOriginWoFullCache(ClusterCachePipeline):
    def __init__(self, config):
        super().__init__(config, cache_impl=DELTA_ORIGIN_WO_FULL)


class DeltaOriginWFullCache(ClusterCachePipeline):
    def __init__(self, config):
        super().__init__(config, cache_impl=DELTA_ORIGIN_W_FULL)
