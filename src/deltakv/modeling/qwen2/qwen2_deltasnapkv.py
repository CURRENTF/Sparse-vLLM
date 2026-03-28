import torch

from dataclasses import dataclass
from typing import Optional, Union, Any
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, Cache, Unpack, FlashAttentionKwargs, rotate_half,   # noqa
    Callable, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM,   # noqa
    logger, BaseModelOutputWithPast, create_sliding_window_causal_mask, create_causal_mask,   # noqa
    Union, KwargsForCausalLM,  # noqa
)

from deltakv.modeling.kv_cache import ClusterCompressedKVCache
from deltakv.configs.model_config_cls import KVQwen2Config, parse_full_attn_layers
from deltakv.modeling.qwen2.qwen2_e2e import create_compressor
from deltakv.modeling.token_select import omnikv_token_selection
from pprint import pprint


@dataclass
class Output(BaseModelOutputWithPast):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    compress_loss: Optional[torch.FloatTensor] = None


def single_apply_rotary_pos_emb(k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return k_embed


def _get_ref_ratio(config: KVQwen2Config) -> float:
    ref_budget = config.deltasnapkv_ref_budget
    if ref_budget is None or float(ref_budget) < 0:
        ref_budget = config.cluster_ratio
    return max(float(ref_budget), 0.0)


def _get_compressed_keep_ratio(config: KVQwen2Config, head_dim: int) -> Union[int, float]:
    total_budget = config.deltasnapkv_total_budget
    if total_budget is None or float(total_budget) <= 0:
        return config.num_top_tokens_in_prefill

    ref_ratio = _get_ref_ratio(config)
    extra_budget = max(float(total_budget) - ref_ratio, 0.0)
    raw_kv_dim = 2 * config.num_key_value_heads * head_dim
    comp_token_cost = float(config.kv_compressed_size) / float(raw_kv_dim)
    if comp_token_cost <= 0:
        return config.num_top_tokens_in_prefill
    return min(extra_budget / comp_token_cost, 1.0)


class DeltaSnapKVCache(ClusterCompressedKVCache):
    def __init__(self, config: KVQwen2Config) -> None:
        super().__init__(config)
        self.ref_key_cache = {}
        self.ref_value_cache = {}
        self.ref_abs_idx = {}
        self.comp_abs_idx = {}
        self.ref_ratio = _get_ref_ratio(config)
        self.obs_window_size = 0
        self.is_finalized = {}

    @property
    def is_last_chunk(self):
        return self.num_prompt_tokens is not None and self._seen_tokens == self.num_prompt_tokens

    def _build_ref_mask(self, start_pos: int, length: int, device: torch.device) -> torch.Tensor:
        if self.ref_ratio <= 0 or length <= 0:
            return torch.zeros(length, dtype=torch.bool, device=device)
        step = max(1, int(round(1.0 / self.ref_ratio)))
        abs_pos = torch.arange(start_pos, start_pos + length, device=device)
        return ((abs_pos - self.sink_size) % step) == 0

    @staticmethod
    def _append_tensor(cache_dict, layer_idx: int, value: torch.Tensor):
        if value is None or value.shape[1] == 0:
            return
        if layer_idx not in cache_dict:
            cache_dict[layer_idx] = value
        else:
            cache_dict[layer_idx] = torch.cat([cache_dict[layer_idx], value], dim=1)

    def _protected_suffix_len(self, buffer_len: int) -> int:
        return min(buffer_len, self.tail_token_size + self.obs_window_size)

    def _get_history_partition(self, layer_idx: int):
        if layer_idx not in self.buffer_key_cache:
            return None

        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        protected_suffix = self._protected_suffix_len(buffer_len)
        history_len = max(buffer_len - protected_suffix, 0)
        if history_len <= 0:
            return None

        device = self.buffer_key_cache[layer_idx].device
        buffer_start = self._seen_tokens - buffer_len
        abs_idx = torch.arange(buffer_start, buffer_start + history_len, device=device)
        ref_mask = self._build_ref_mask(buffer_start, history_len, device)

        return {
            "history_len": history_len,
            "abs_idx": abs_idx,
            "ref_mask": ref_mask,
            "candidate_mask": ~ref_mask,
        }

    def get_candidate_abs_idx(self, layer_idx: int):
        part = self._get_history_partition(layer_idx)
        if part is None or not part["candidate_mask"].any():
            return None
        bs = self.buffer_key_cache[layer_idx].shape[0]
        return part["abs_idx"][part["candidate_mask"]][None, :].expand(bs, -1)

    def keep_compressed_tokens(self, layer_idx: int, keep_idx: torch.Tensor):
        if layer_idx not in self.comp_kv_cache:
            return
        keep_idx = torch.sort(keep_idx, dim=1).values
        self.comp_kv_cache[layer_idx] = self.comp_kv_cache[layer_idx].gather(
            1, keep_idx[:, :, None].expand(-1, -1, self.comp_kv_cache[layer_idx].shape[-1])
        )
        self.token_father_idx[layer_idx] = self.token_father_idx[layer_idx].gather(
            1, keep_idx[:, :, None].expand(-1, -1, self.token_father_idx[layer_idx].shape[-1])
        )
        self.comp_abs_idx[layer_idx] = self.comp_abs_idx[layer_idx].gather(1, keep_idx)
        if layer_idx in self.comp_kv_scales:
            self.comp_kv_scales[layer_idx] = self.comp_kv_scales[layer_idx].gather(
                1, keep_idx[:, :, None].expand(-1, -1, self.comp_kv_scales[layer_idx].shape[-1])
            )
            self.comp_kv_mins[layer_idx] = self.comp_kv_mins[layer_idx].gather(
                1, keep_idx[:, :, None].expand(-1, -1, self.comp_kv_mins[layer_idx].shape[-1])
            )

    def finalize_static_prune(
        self,
        layer_idx: int,
        keep_idx: Optional[torch.Tensor],
        compressor_down: Optional[nn.Module],
    ):
        if self.is_finalized.get(layer_idx, False) or layer_idx in self.full_attn_layers:
            return

        part = self._get_history_partition(layer_idx)
        self.is_finalized[layer_idx] = True
        if part is None:
            return

        history_len = part["history_len"]
        abs_idx = part["abs_idx"]
        ref_mask = part["ref_mask"]
        candidate_mask = part["candidate_mask"]

        history_k = self.buffer_key_cache[layer_idx][:, :history_len]
        history_v = self.buffer_value_cache[layer_idx][:, :history_len]
        self.buffer_key_cache[layer_idx] = self.buffer_key_cache[layer_idx][:, history_len:]
        self.buffer_value_cache[layer_idx] = self.buffer_value_cache[layer_idx][:, history_len:]

        bs = history_k.shape[0]
        existing_centers = self.bases_cache.get(layer_idx, None)

        if ref_mask.any():
            ref_k = history_k[:, ref_mask, :]
            ref_v = history_v[:, ref_mask, :]
            ref_abs_idx = abs_idx[ref_mask][None, :].expand(bs, -1)
            self.ref_key_cache[layer_idx] = ref_k
            self.ref_value_cache[layer_idx] = ref_v
            self.ref_abs_idx[layer_idx] = ref_abs_idx
            ref_kv = torch.cat([ref_k, ref_v], dim=-1)
            existing_centers = torch.cat([existing_centers, ref_kv], dim=1) if existing_centers is not None else ref_kv
            self.bases_cache[layer_idx] = existing_centers

        if not candidate_mask.any():
            return

        cand_k = history_k[:, candidate_mask, :]
        cand_v = history_v[:, candidate_mask, :]
        cand_abs_idx = abs_idx[candidate_mask][None, :].expand(bs, -1)

        if keep_idx is None:
            keep_idx = torch.arange(cand_k.shape[1], device=cand_k.device)[None, :].expand(bs, -1)
        elif keep_idx.numel() == 0:
            keep_idx = keep_idx.to(device=cand_k.device, dtype=torch.long)
        else:
            keep_idx = torch.sort(keep_idx, dim=1).values

        if keep_idx.shape[1] == 0:
            return

        cand_k = cand_k.gather(1, keep_idx[:, :, None].expand(-1, -1, cand_k.shape[-1]))
        cand_v = cand_v.gather(1, keep_idx[:, :, None].expand(-1, -1, cand_v.shape[-1]))
        cand_abs_idx = cand_abs_idx.gather(1, keep_idx)

        to_be_compress = torch.cat([cand_k, cand_v], dim=-1)
        comp_kv, all_centers, father_idx, scale, mn = self.compress(
            to_be_compress,
            compressor_down,
            existing_centers,
        )
        self.bases_cache[layer_idx] = all_centers
        self.comp_kv_cache[layer_idx] = comp_kv
        self.token_father_idx[layer_idx] = father_idx
        self.comp_abs_idx[layer_idx] = cand_abs_idx
        if scale is not None:
            self.comp_kv_scales[layer_idx] = scale
            self.comp_kv_mins[layer_idx] = mn

    def _compose_sorted_response(
        self,
        *,
        layer_idx: int,
        sink_idx: torch.Tensor,
        buffer_idx: torch.Tensor,
        recon_k: Optional[torch.Tensor],
        recon_v: Optional[torch.Tensor],
        recon_abs_idx: Optional[torch.Tensor],
    ):
        parts_k = [self.sink_key_cache[layer_idx]]
        parts_v = [self.sink_value_cache[layer_idx]]
        parts_idx = [sink_idx]

        ref_idx = self.ref_abs_idx.get(layer_idx, None)
        if ref_idx is not None and ref_idx.shape[1] > 0:
            parts_k.append(self.ref_key_cache[layer_idx])
            parts_v.append(self.ref_value_cache[layer_idx])
            parts_idx.append(ref_idx)

        if recon_k is not None and recon_k.shape[1] > 0:
            parts_k.append(recon_k)
            parts_v.append(recon_v)
            parts_idx.append(recon_abs_idx)

        parts_k.append(self.buffer_key_cache[layer_idx])
        parts_v.append(self.buffer_value_cache[layer_idx])
        parts_idx.append(buffer_idx)

        full_k = torch.cat(parts_k, dim=1)
        full_v = torch.cat(parts_v, dim=1)
        full_idx = torch.cat(parts_idx, dim=1)
        order = torch.argsort(full_idx, dim=1)
        gather_idx = order[:, :, None].expand(-1, -1, full_k.shape[-1])
        full_k = full_k.gather(1, gather_idx)
        full_v = full_v.gather(1, gather_idx)
        full_idx = full_idx.gather(1, order)
        return full_k, full_v, full_idx

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[dict[str, Any]] = None,
            compressor_down: Optional[nn.Module] = None,
            compressor_up: Optional[nn.Module] = None,
    ):
        if layer_idx == 0:
            self._seen_tokens += key_states.shape[-2]

        if key_states is None:
            raise NotImplementedError

        bs, _, k_dim = key_states.shape

        if layer_idx not in self.buffer_key_cache:
            self.sink_size = min(self.sink_size, key_states.shape[1])
            self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx] = key_states[:, :self.sink_size], key_states[:, self.sink_size:]
            self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx] = value_states[:, :self.sink_size], value_states[:, self.sink_size:]
            self.bases_cache[layer_idx] = torch.cat([self.sink_key_cache[layer_idx], self.sink_value_cache[layer_idx]], dim=-1)
        else:
            self.buffer_key_cache[layer_idx] = torch.cat([self.buffer_key_cache[layer_idx], key_states], dim=1)
            self.buffer_value_cache[layer_idx] = torch.cat([self.buffer_value_cache[layer_idx], value_states], dim=1)

        buffer_len = self.buffer_key_cache[layer_idx].shape[1]
        seen_tokens = self._seen_tokens

        sink_idx = torch.arange(0, self.sink_size, device=key_states.device)[None, :].expand(bs, -1)
        buffer_idx = torch.arange(seen_tokens - buffer_len, seen_tokens, device=key_states.device)[None, :].expand(bs, -1)

        if layer_idx in self.full_attn_layers:
            full_idx = torch.cat([sink_idx, buffer_idx], dim=1)
            return (
                torch.cat([self.sink_key_cache[layer_idx], self.buffer_key_cache[layer_idx]], dim=1),
                torch.cat([self.sink_value_cache[layer_idx], self.buffer_value_cache[layer_idx]], dim=1),
                full_idx,
            )

        if layer_idx in self.comp_kv_cache:
            recon_k, recon_v = self._reconstruct_all_cluster_tokens(
                layer_idx=layer_idx,
                compressor_up=compressor_up,
                bs=bs,
                k_dim=k_dim,
            )
            recon_abs_idx = self.comp_abs_idx[layer_idx]
            this_response = self._compose_sorted_response(
                layer_idx=layer_idx,
                sink_idx=sink_idx,
                buffer_idx=buffer_idx,
                recon_k=recon_k,
                recon_v=recon_v,
                recon_abs_idx=recon_abs_idx,
            )
        else:
            this_response = self._compose_sorted_response(
                layer_idx=layer_idx,
                sink_idx=sink_idx,
                buffer_idx=buffer_idx,
                recon_k=None,
                recon_v=None,
                recon_abs_idx=None,
            )

        return this_response


class Qwen2DeltaSnapKVAttention(Qwen2Attention):
    def __init__(self, config: KVQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)

        full_layers = parse_full_attn_layers(config.full_attn_layers)
        config.full_attn_layers = full_layers

        self.is_full_layer = layer_idx in full_layers

        self.compress_down = create_compressor(is_down=True, config=config)
        self.compress_up = create_compressor(is_down=False, config=config)
        self.config = config
        self.layer_idx = layer_idx

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[DeltaSnapKVCache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        bs, q_len, ___ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.config.collect_kv_before_rope:
            cache_kwargs = {"cache_position": cache_position}
            key_states, value_states, full_idx = past_key_value.update(
                key_states,
                value_states,
                self.layer_idx,
                cache_kwargs,
                compressor_down=self.compress_down,
                compressor_up=self.compress_up,
            )
            # `device_map="auto"` may shard layers across GPUs. Make sure indices used
            # for gather/searchsorted live on the local layer device.
            if full_idx is not None and full_idx.device != key_states.device:
                full_idx = full_idx.to(key_states.device)
        else:
            raise NotImplementedError

        query_states = query_states.view(bs, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bs, -1, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bs, -1, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        cur_cos, cur_sin = position_embeddings
        query_states = single_apply_rotary_pos_emb(query_states, cur_cos, cur_sin)

        k_cos = past_key_value.cos.gather(
            1, full_idx.unsqueeze(-1).expand(-1, -1, self.head_dim)
        )
        k_sin = past_key_value.sin.gather(
            1, full_idx.unsqueeze(-1).expand(-1, -1, self.head_dim)
        )
        key_states = single_apply_rotary_pos_emb(key_states, k_cos, k_sin)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        is_prefill = q_len > 1
        do_prune = (
            (not self.is_full_layer)
            and is_prefill
            and past_key_value.is_last_chunk
            and (not past_key_value.is_finalized.get(self.layer_idx, False))
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        if do_prune:
            candidate_abs_idx = past_key_value.get_candidate_abs_idx(self.layer_idx)
            keep_idx = None
            if candidate_abs_idx is not None and candidate_abs_idx.shape[1] > 0:
                if candidate_abs_idx.device != full_idx.device:
                    candidate_abs_idx = candidate_abs_idx.to(full_idx.device)
                keep_ratio = _get_compressed_keep_ratio(self.config, self.head_dim)
                if isinstance(keep_ratio, float) and keep_ratio <= 0:
                    keep_idx = torch.empty((bs, 0), dtype=torch.long, device=key_states.device)
                elif isinstance(keep_ratio, float) and keep_ratio >= 1:
                    keep_idx = torch.arange(candidate_abs_idx.shape[1], device=key_states.device)[None, :].expand(bs, -1)
                else:
                    candidate_pos = torch.searchsorted(full_idx, candidate_abs_idx)
                    gather_idx = candidate_pos[:, None, :, None].expand(-1, self.config.num_key_value_heads, -1, self.head_dim)
                    candidate_key = key_states.gather(2, gather_idx)
                    keep_idx, _ = omnikv_token_selection(
                        self,
                        query_states,
                        candidate_key,
                        self.scaling,
                        keep_ratio,
                        pool_kernel_size=1,
                        last_token_scores=None,
                        score_method=self.config.omnikv_score_method,
                    )
            past_key_value.finalize_static_prune(self.layer_idx, keep_idx, self.compress_down)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2DeltaSnapKVLayer(Qwen2DecoderLayer):
    def __init__(self, config: KVQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2DeltaSnapKVAttention(config=config, layer_idx=layer_idx)


class Qwen2DeltaSnapKVModel(Qwen2Model):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        full_layers = parse_full_attn_layers(config.full_attn_layers)
        assert len(full_layers) == 0, (
            "deltasnapkv requires full_attn_layers to be empty; "
            "this implementation assumes all decoder layers use the same static-prune path."
        )
        config.full_attn_layers = []
        self.layers = nn.ModuleList(
            [Qwen2DeltaSnapKVLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.config = config
        pprint(config)
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[DeltaSnapKVCache] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        assert isinstance(past_key_values, DeltaSnapKVCache)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        cos, sin = position_embeddings
        if past_key_values.cos is None:
            past_key_values.cos = cos
            past_key_values.sin = sin
        else:
            past_key_values.cos = torch.cat([past_key_values.cos, cos], dim=1)
            past_key_values.sin = torch.cat([past_key_values.sin, sin], dim=1)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states[:, -1:],
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2DeltaSnapKVForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        self.model = Qwen2DeltaSnapKVModel(config)
        self.config = config
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[DeltaSnapKVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ):
        assert input_ids is not None and attention_mask is None
        assert input_ids.shape[0] == 1
        assert position_ids is None and use_cache
        assert use_cache, "Inference model must use cache"

        if not isinstance(past_key_values, DeltaSnapKVCache):
            past_key_values = DeltaSnapKVCache(config=self.config)

        chunk_size = self.config.chunk_prefill_size
        obs_window_size = self.config.snapkv_window_size
        outputs = None
        if input_ids.shape[1] > 1:
            past_key_values.num_prompt_tokens = input_ids.shape[1]
            if input_ids.shape[1] <= obs_window_size or obs_window_size <= 0:
                chunk_input_ids = list(input_ids.split(chunk_size, dim=-1))
                past_key_values.obs_window_size = input_ids.shape[1]
            else:
                context_ids = input_ids[:, :-obs_window_size]
                obs_ids = input_ids[:, -obs_window_size:]
                chunk_input_ids = list(context_ids.split(chunk_size, dim=-1)) + [obs_ids]
                past_key_values.obs_window_size = obs_ids.shape[1]
        else:
            chunk_input_ids = [input_ids]

        for _ipt_ids in chunk_input_ids:
            outputs = super().forward(_ipt_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values

        return outputs
