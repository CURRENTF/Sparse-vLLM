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

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        return self._seen_tokens


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
        self.sink_key_cache = {}
        self.sink_value_cache = {}

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
        kv_chunks = kv_states.view(bs, -1, self.config.seq_chunk_size, kv_dim)

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
            bases = bases.repeat_interleave(self.config.seq_chunk_size, dim=1)[:, : comp_kv.shape[1]]
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

        if layer_idx not in self.buffer_key_cache:
            # init
            self.sink_size = min(self.sink_size, key_states.shape[1])
            self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx] = key_states[:, :self.sink_size], key_states[:, self.sink_size:]
            self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx] = value_states[:, :self.sink_size], value_states[:, self.sink_size:]
        else:
            self.buffer_key_cache[layer_idx] = torch.cat([self.buffer_key_cache[layer_idx], key_states], dim=1)
            self.buffer_value_cache[layer_idx] = torch.cat([self.buffer_value_cache[layer_idx], value_states], dim=1)

        # 初始化一些index
        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        seen_tokens = self._seen_tokens
        
        sink_idx = torch.arange(0, self.sink_size, device=key_states.device).unsqueeze(0).expand(bs, -1)
        buffer_idx = torch.arange(seen_tokens - buffer_len, seen_tokens, device=key_states.device).unsqueeze(0).expand(bs, -1)
        
        if layer_idx in self.full_attn_layers:
            # 做full attn的layer不需要压缩kv cache
            full_idx = torch.cat([sink_idx, buffer_idx], dim=1)
            return torch.cat([self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]], dim=1), \
                   torch.cat([self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]], dim=1), \
                   full_idx

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
                    bases_idx = rel_token_idx // self.config.seq_chunk_size
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

                # rel_token_idx + sink_size = absolute position
                full_idx = torch.cat([sink_idx, rel_token_idx + self.sink_size, buffer_idx], dim=1)

                this_response = (torch.cat([self.sink_key_cache[layer_idx], recon_k, self.buffer_key_cache[layer_idx]], dim=1),
                        torch.cat([self.sink_value_cache[layer_idx], recon_v, self.buffer_value_cache[layer_idx]], dim=1),
                        full_idx)
            else:
                # Standalone DeltaKV: if no external selector provided token ids, reconstruct all compressed history.
                history_k, history_v = self._decompress_all_history_kv(
                    layer_idx=layer_idx,
                    compressor_up=compressor_up,
                    bs=bs,
                    k_dim=k_dim,
                )
                k = torch.cat([self.sink_key_cache[layer_idx], history_k, self.buffer_key_cache[layer_idx]], dim=1)
                v = torch.cat([self.sink_value_cache[layer_idx], history_v, self.buffer_value_cache[layer_idx]], dim=1)
                full_idx = torch.arange(k.shape[1], device=k.device)[None].expand(bs, -1)
                this_response = (k, v, full_idx)
        else:
            full_idx = torch.cat([sink_idx, buffer_idx], dim=1)
            this_response = (torch.cat([self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]], dim=1),
                    torch.cat([self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]], dim=1),
                    full_idx)

        # 如果buffer内长度超过2倍size，做压缩，只剩下 tail_token_size个token
        if buffer_len >= self.tail_token_size * 2:
            compress_len = (buffer_len - self.tail_token_size)//self.tail_token_size * self.tail_token_size
            if compress_len > 0:
                _key, self.buffer_key_cache[layer_idx] = self.buffer_key_cache[layer_idx][:, :compress_len], self.buffer_key_cache[layer_idx][:, compress_len:]
                _val, self.buffer_value_cache[layer_idx] = self.buffer_value_cache[layer_idx][:, :compress_len], self.buffer_value_cache[layer_idx][:, compress_len:]

                bs, _len, k_dim = _key.shape
                assert _len == compress_len, f'{_len}, {compress_len}'

                if self.config.use_compression:
                    to_be_compress = torch.cat([_key, _val], dim=-1)
                    comp_kv, bases = self.compress(to_be_compress, compressor_down)

                    if layer_idx not in self.comp_kv_cache:
                        self.comp_kv_cache[layer_idx], self.bases_cache[layer_idx] = comp_kv, bases
                    else:
                        self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], comp_kv], dim=1)
                        self.bases_cache[layer_idx] = torch.cat([self.bases_cache[layer_idx], bases], dim=1)
                else:
                    if self.config.kv_quant_bits == 4:
                        packed_kv, scale, mn = self._quantize_kv_tokens(torch.cat([_key, _val], dim=-1))
                        if layer_idx not in self.comp_kv_cache:
                            self.comp_kv_cache[layer_idx] = packed_kv
                            self.comp_kv_scales[layer_idx] = scale
                            self.comp_kv_mins[layer_idx] = mn
                        else:
                            self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], packed_kv], dim=1)
                            self.comp_kv_scales[layer_idx] = torch.cat([self.comp_kv_scales[layer_idx], scale], dim=1)
                            self.comp_kv_mins[layer_idx] = torch.cat([self.comp_kv_mins[layer_idx], mn], dim=1)
                    else:
                        if layer_idx not in self.comp_kv_cache:
                            self.comp_kv_cache[layer_idx], self.bases_cache[layer_idx] = _key, _val
                        else:
                            self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], _key], dim=1)
                            self.bases_cache[layer_idx] = torch.cat([self.bases_cache[layer_idx], _val], dim=1)

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
            return torch.arange(0, seq_len, base_step, device=device, dtype=torch.long)

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
        self._cluster_center_plan_cache_key = cache_key
        self._cluster_center_plan_cache_val = center_rel
        return center_rel

    @staticmethod
    def _metric_l2(kv_states, all_centers, use_kv=False):
        if not use_kv:
            k_dim = kv_states.shape[-1] // 2
            kv_states = kv_states[..., :k_dim]
            all_centers = all_centers[..., :k_dim]
        return torch.cdist(kv_states, all_centers)

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
            # 使用负距离以便 topk 选取最近的
            scores = -self._metric_l2(kv_states, all_centers, use_kv=use_kv)
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
        k = max(1, self.config.seq_chunk_size)
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

        if layer_idx not in self.buffer_key_cache:
            # init
            self.sink_size = min(self.sink_size, key_states.shape[1])
            self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx] = key_states[:, :self.sink_size], key_states[:, self.sink_size:]
            self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx] = value_states[:, :self.sink_size], value_states[:, self.sink_size:]
            # 把 sink tokens 也作为初始聚类中心
            self.bases_cache[layer_idx] = torch.cat([self.sink_key_cache[layer_idx], self.sink_value_cache[layer_idx]], dim=-1)
            # Initialize dynamic stride cursor after sink size is finalized.
            if self._cluster_next_center_abs_pos is None:
                self._cluster_next_center_abs_pos = int(self.sink_size)
        else:
            self.buffer_key_cache[layer_idx] = torch.cat([self.buffer_key_cache[layer_idx], key_states], dim=1)
            self.buffer_value_cache[layer_idx] = torch.cat([self.buffer_value_cache[layer_idx], value_states], dim=1)

        # 初始化一些index
        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        seen_tokens = self._seen_tokens
        
        sink_idx = torch.arange(0, self.sink_size, device=key_states.device).unsqueeze(0).expand(bs, -1)
        buffer_idx = torch.arange(seen_tokens - buffer_len, seen_tokens, device=key_states.device).unsqueeze(0).expand(bs, -1)
        
        if layer_idx in self.full_attn_layers:
            # 做full attn的layer不需要压缩kv cache
            full_idx = torch.cat([sink_idx, buffer_idx], dim=1)
            return torch.cat([self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]], dim=1), \
                   torch.cat([self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]], dim=1), \
                   full_idx

        # 下面是 sparse attn layers :
        # 根据token idx重建所需kv cache
        if layer_idx in self.comp_kv_cache:
            if self.layer_to_full_layer_idx[layer_idx] in self.top_token_idx:
                token_idx = self.top_token_idx[self.layer_to_full_layer_idx[layer_idx]]  # bs, num_imp_tokens

                if not self.config.use_compression:
                    raise NotImplementedError("Cluster without compression is not implemented")

                recon_k, recon_v = self._reconstruct_selected_cluster_tokens(
                    layer_idx=layer_idx,
                    token_idx=token_idx,
                    compressor_up=compressor_up,
                    bs=bs,
                    k_dim=k_dim,
                )

                full_idx = torch.cat([sink_idx, token_idx + self.sink_size, buffer_idx], dim=1)

                this_response = (torch.cat([self.sink_key_cache[layer_idx], recon_k, self.buffer_key_cache[layer_idx]], dim=1),
                                torch.cat([self.sink_value_cache[layer_idx], recon_v, self.buffer_value_cache[layer_idx]], dim=1),
                                full_idx)
            else:
                if not self.config.use_compression:
                    raise NotImplementedError("Cluster without compression is not implemented")

                recon_k, recon_v = self._reconstruct_all_cluster_tokens(
                    layer_idx=layer_idx,
                    compressor_up=compressor_up,
                    bs=bs,
                    k_dim=k_dim,
                )
                num_hist = recon_k.shape[1]
                hist_idx = torch.arange(num_hist, device=key_states.device).unsqueeze(0).expand(bs, -1) + self.sink_size
                full_idx = torch.cat([sink_idx, hist_idx, buffer_idx], dim=1)
                this_response = (
                    torch.cat([self.sink_key_cache[layer_idx], recon_k, self.buffer_key_cache[layer_idx]], dim=1),
                    torch.cat([self.sink_value_cache[layer_idx], recon_v, self.buffer_value_cache[layer_idx]], dim=1),
                    full_idx,
                )
        else:
            full_idx = torch.cat([sink_idx, buffer_idx], dim=1)
            this_response = (torch.cat([self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]], dim=1),
                            torch.cat([self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]], dim=1),
                            full_idx)

        # 如果buffer内长度超过2倍size，做压缩
        if buffer_len >= self.tail_token_size * 2:
            compress_len = (buffer_len - self.tail_token_size) // self.tail_token_size * self.tail_token_size
            if compress_len > 0:
                _key, self.buffer_key_cache[layer_idx] = self.buffer_key_cache[layer_idx][:, :compress_len], self.buffer_key_cache[layer_idx][:, compress_len:]
                _val, self.buffer_value_cache[layer_idx] = self.buffer_value_cache[layer_idx][:, :compress_len], self.buffer_value_cache[layer_idx][:, compress_len:]

                bs, _len, k_dim = _key.shape
                to_be_compress = torch.cat([_key, _val], dim=-1)
                
                existing_centers = self.bases_cache.get(layer_idx, None)
                # compress 内部会处理量化逻辑
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
                else:
                    self.comp_kv_cache[layer_idx] = torch.cat([self.comp_kv_cache[layer_idx], comp_kv], dim=1)
                    self.token_father_idx[layer_idx] = torch.cat([self.token_father_idx[layer_idx], father_idx], dim=1)
                    if scale is not None:
                        self.comp_kv_scales[layer_idx] = torch.cat([self.comp_kv_scales[layer_idx], scale], dim=1)
                        self.comp_kv_mins[layer_idx] = torch.cat([self.comp_kv_mins[layer_idx], mn], dim=1)

        return this_response
