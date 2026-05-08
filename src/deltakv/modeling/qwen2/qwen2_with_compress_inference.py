import torch
import os

from typing import Optional
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, Cache, Unpack, FlashAttentionKwargs, rotate_half,   # noqa
    Callable, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM,   # noqa
    logger, BaseModelOutputWithPast, create_sliding_window_causal_mask, create_causal_mask,   # noqa
    Union, KwargsForCausalLM,  # noqa
)   # noqa
from transformers.modeling_outputs import CausalLMOutputWithPast

from deltakv.modeling.kv_cache import CompressedKVCache, ClusterCompressedKVCache
from deltakv.configs.model_config_cls import KVQwen2Config, parse_full_attn_layers
from deltakv.modeling.cache_factory import create_deltakv_cache, is_deltakv_cache_instance
from deltakv.modeling.qwen2.qwen2_e2e import create_compressor
from deltakv.modeling.token_select import omnikv_token_selection
from dataclasses import dataclass
from accelerate import Accelerator
from pprint import pprint

accelerator = Accelerator()


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


def _position_ids_from_attention_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    return position_ids


def _cache_aligned_padding_mask(
    attention_mask_2d: Optional[torch.Tensor],
    full_idx: torch.Tensor,
) -> Optional[torch.Tensor]:
    if attention_mask_2d is None:
        return None

    device = full_idx.device
    attention_mask_2d = attention_mask_2d.to(device=device, dtype=torch.bool)
    mask_len = attention_mask_2d.shape[1]
    safe_full_idx = full_idx.clamp(min=0, max=max(mask_len - 1, 0))
    key_valid = attention_mask_2d.gather(1, safe_full_idx) & (full_idx < mask_len)
    return key_valid


def _cache_aligned_additive_mask(
    attention_mask_2d: Optional[torch.Tensor],
    full_idx: torch.Tensor,
    cache_position: Optional[torch.LongTensor],
    q_len: int,
    dtype: torch.dtype,
) -> Optional[torch.Tensor]:
    if attention_mask_2d is None:
        return None

    key_valid = _cache_aligned_padding_mask(attention_mask_2d, full_idx)

    bs, k_len = full_idx.shape
    device = full_idx.device

    if cache_position is None:
        mask_len = attention_mask_2d.shape[1]
        query_pos = torch.arange(mask_len - q_len, mask_len, device=device, dtype=torch.long)
    else:
        query_pos = cache_position.to(device=device, dtype=torch.long)
    if query_pos.dim() == 1:
        query_pos = query_pos.unsqueeze(0).expand(bs, -1)
    else:
        query_pos = query_pos[:, -q_len:]

    causal = full_idx[:, None, :] <= query_pos[:, :, None]
    allowed = key_valid[:, None, :] & causal
    attn_mask = torch.zeros((bs, 1, q_len, k_len), device=device, dtype=dtype)
    return attn_mask.masked_fill(~allowed[:, None, :, :], torch.finfo(dtype).min)


def _gather_valid_mask(attention_mask_2d: Optional[torch.Tensor], token_pos: torch.Tensor) -> Optional[torch.Tensor]:
    if attention_mask_2d is None:
        return None
    attention_mask_2d = attention_mask_2d.to(device=token_pos.device, dtype=torch.bool)
    mask_len = attention_mask_2d.shape[1]
    safe_pos = token_pos.clamp(min=0, max=max(mask_len - 1, 0))
    return attention_mask_2d.gather(1, safe_pos) & (token_pos < mask_len)


class Qwen2AttnKVCompress(Qwen2Attention):
    def __init__(self, config: KVQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)

        full_layers = parse_full_attn_layers(config.full_attn_layers)
        config.full_attn_layers = full_layers

        self.is_full_layer = (layer_idx in full_layers)
        self.is_obs_layer = bool(full_layers) and self.is_full_layer and (layer_idx + 1) not in full_layers

        if self.is_obs_layer:
            all_obs_layers = sorted([idx for idx in full_layers if (idx + 1) not in full_layers])
            self.obs_index = all_obs_layers.index(layer_idx)
        else:
            self.obs_index = None

        # Initialize compressors here to match training code structure
        self.compress_down = create_compressor(is_down=True, config=config)
        self.compress_up = create_compressor(is_down=False, config=config)
        self.config = config
        self.layer_idx = layer_idx

    # 主要修改这个函数
    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Union[CompressedKVCache, ClusterCompressedKVCache]] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ):
        deltakv_visual_token_mask = kwargs.pop("deltakv_visual_token_mask", None)
        attention_mask_2d = kwargs.pop("deltakv_attention_mask_2d", None)
        input_shape = hidden_states.shape[:-1]
        bs, q_len, ___ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        # now shape --> bs, seq_len, dim

        if self.config.collect_kv_before_rope:
            cache_kwargs = {
                "cache_position": cache_position,
                "deltakv_visual_token_mask": deltakv_visual_token_mask,
                "deltakv_attention_mask": attention_mask_2d,
            }
            res = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
                compressor_down=self.compress_down,
                compressor_up=self.compress_up
            )
            key_states, value_states, full_idx = res
        else:
            raise NotImplementedError

        aligned_padding_mask = _cache_aligned_padding_mask(attention_mask_2d, full_idx)
        aligned_additive_mask = None
        if attention_mask_2d is not None and self.config._attn_implementation != "flash_attention_2":
            aligned_additive_mask = _cache_aligned_additive_mask(
                attention_mask_2d,
                full_idx,
                cache_position,
                q_len,
                hidden_states.dtype,
            )
        if self.config._attn_implementation == "flash_attention_2":
            attention_mask = aligned_padding_mask
        elif aligned_additive_mask is not None:
            attention_mask = aligned_additive_mask

        query_states = query_states.view(bs, q_len, self.config.num_attention_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bs, -1, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bs, -1, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)
        # now shape --> bs, heads, seq_len, head_dim

        # 应用位置编码
        # query_states 使用当前位置的 cos, sin
        cur_cos, cur_sin = position_embeddings
        query_states = single_apply_rotary_pos_emb(query_states, cur_cos, cur_sin)

        # key_states 使用 gathered cos, sin
        # past_key_value.cos shape: (bs, seq_len, head_dim)
        # full_idx shape: (bs, k_len)
        safe_full_idx = full_idx.clamp(min=0, max=past_key_value.cos.shape[1] - 1)
        k_cos = past_key_value.cos.gather(1, safe_full_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
        k_sin = past_key_value.sin.gather(1, safe_full_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
        key_states = single_apply_rotary_pos_emb(key_states, k_cos, k_sin)

        attention_interface: Callable = eager_attention_forward

        sink_size = self.config.num_sink_tokens
        is_prefill, is_decode = (q_len > 1), (q_len == 1)

        visual_token_prune_only = bool(getattr(self.config, "visual_token_prune_only", False))
        candidate_abs_idx = None
        candidate_buffer_pos = None
        candidate_buffer_valid = None
        if visual_token_prune_only:
            target_layer_idx = None
            comp_pos_cache = getattr(past_key_value, "comp_pos_cache", {})
            for candidate_layer_idx in sorted(comp_pos_cache):
                if past_key_value.layer_to_full_layer_idx.get(candidate_layer_idx) == self.layer_idx:
                    target_layer_idx = candidate_layer_idx
                    break
            if target_layer_idx is not None:
                candidate_abs_idx = past_key_value.get_compressed_positions(target_layer_idx, device=key_states.device)
            compressed_len = 0 if candidate_abs_idx is None else int(candidate_abs_idx.shape[1])
        else:
            compressible_lengths = None
            if hasattr(past_key_value, "get_buffer_valid_lengths") and self.is_obs_layer:
                buffer_valid_lengths = past_key_value.get_buffer_valid_lengths(self.layer_idx, device=key_states.device)
                if buffer_valid_lengths is not None:
                    if attention_mask_2d is None:
                        current_valid_lengths = torch.full_like(buffer_valid_lengths, q_len)
                    else:
                        current_valid_lengths = attention_mask_2d.to(device=key_states.device, dtype=torch.bool)[:, -q_len:].sum(dim=1)
                    history_valid_lengths = (buffer_valid_lengths - current_valid_lengths).clamp_min(0)
                    compressible_lengths = torch.div(
                        (history_valid_lengths - self.config.tail_token_size).clamp_min(0),
                        self.config.tail_token_size,
                        rounding_mode="floor",
                    ) * self.config.tail_token_size
            if compressible_lengths is not None:
                compressed_len = int(compressible_lengths.max().item())
                if compressed_len > 0 and hasattr(past_key_value, "get_buffer_candidate_positions"):
                    candidate = past_key_value.get_buffer_candidate_positions(
                        self.layer_idx,
                        compressible_lengths,
                        device=key_states.device,
                    )
                    if candidate is not None:
                        candidate_buffer_pos, candidate_buffer_valid = candidate
            else:
                compressed_len = (past_key_value.get_seq_length() - self.config.tail_token_size - q_len - sink_size) // self.config.tail_token_size * self.config.tail_token_size
        use_omnikv_selection = bool(self.config.deltakv_use_omnikv_selection)
        do_obs = (
            use_omnikv_selection
            and self.is_obs_layer
            and compressed_len > 0
            and (self.config.chunk_prefill_accel_omnikv or is_decode)
        )

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # print(f'at layer {self.layer_idx}, num of tokens is {key_states.shape[2]}, {q_len=}, {is_prefill=}, {do_obs=}, {len(past_key_value.top_token_idx)=}')
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  # main diff with Llama
            **kwargs,
        )

        if os.getenv('DEBUG'):
            if self.layer_idx == 0:
                print(f'L{self.layer_idx}  {key_states.shape=}  {do_obs=}  {q_len=}')
            if self.layer_idx == 14:
                print(f'L{self.layer_idx}  {key_states.shape=}  {do_obs=}  {q_len=}')
            if self.layer_idx == 15:
                print(f'L{self.layer_idx}  {key_states.shape=}  {do_obs=}  {q_len=}')

        # attn weights shape -> bs, heads, q_len, kv_len
        if do_obs:
            # 重新计算 attention score 以支持 Flash Attention 并处理因果掩码问题
            candidate_mask = None
            query_mask = None
            if attention_mask_2d is not None:
                query_mask = attention_mask_2d.to(device=query_states.device, dtype=torch.bool)[:, -q_len:]
            if visual_token_prune_only:
                if bs != 1:
                    raise NotImplementedError("visual_token_prune_only currently supports batch_size=1.")
                gather_pos = torch.searchsorted(full_idx, candidate_abs_idx)
                valid = (
                    (gather_pos < full_idx.shape[1])
                    & (full_idx.gather(1, gather_pos.clamp(max=full_idx.shape[1] - 1)) == candidate_abs_idx)
                )
                if not bool(valid.all()):
                    gather_pos = gather_pos[valid].unsqueeze(0)
                    compressed_len = int(gather_pos.shape[1])
                candidate_key = key_states.gather(
                    2,
                    gather_pos[:, None, :, None].expand(-1, key_states.shape[1], -1, key_states.shape[-1]),
                )
            else:
                if candidate_buffer_pos is not None and candidate_buffer_valid is not None:
                    gather_pos = torch.searchsorted(full_idx.contiguous(), candidate_buffer_pos.contiguous())
                    gather_pos = gather_pos.clamp(max=full_idx.shape[1] - 1)
                    matched = full_idx.gather(1, gather_pos) == candidate_buffer_pos
                    candidate_mask = candidate_buffer_valid & matched
                    candidate_key = key_states.gather(
                        2,
                        gather_pos[:, None, :, None].expand(-1, key_states.shape[1], -1, key_states.shape[-1]),
                    )
                else:
                    candidate_key = key_states[:, :, sink_size : sink_size + compressed_len, :]
                    candidate_pos = full_idx[:, sink_size : sink_size + compressed_len]
                    candidate_mask = _gather_valid_mask(attention_mask_2d, candidate_pos)
                    if compressible_lengths is not None:
                        candidate_rank = torch.arange(candidate_pos.shape[1], device=candidate_pos.device)
                        rank_mask = candidate_rank.unsqueeze(0) < compressible_lengths.to(candidate_pos.device).unsqueeze(1)
                        candidate_mask = rank_mask if candidate_mask is None else (candidate_mask & rank_mask)
            
            num_top_tokens = self.config.num_top_tokens_in_prefill if is_prefill else self.config.num_top_tokens
            if isinstance(num_top_tokens, (list, tuple)):
                num_top_tokens = num_top_tokens[self.obs_index]
            elif isinstance(num_top_tokens, str) and ',' in num_top_tokens:
                num_top_tokens = [float(x.strip()) for x in num_top_tokens.split(',')]
                num_top_tokens = num_top_tokens[self.obs_index]

            last_token_scores = past_key_value.token_scores.get(self.layer_idx, None)
            
            # Token Selection
            top_token_idx, token_scores = omnikv_token_selection(
                self,
                query_states,
                candidate_key,
                self.scaling,
                num_top_tokens,
                pool_kernel_size=self.config.pool_kernel_size,
                last_token_scores=last_token_scores,
                score_method=self.config.omnikv_score_method,
                candidate_mask=candidate_mask,
                query_mask=query_mask,
            )

            past_key_value.token_scores[self.layer_idx] = token_scores
            past_key_value.top_token_idx[self.layer_idx] = top_token_idx

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2LayerKVCompress(Qwen2DecoderLayer):
    def __init__(self, config: KVQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2AttnKVCompress(config=config, layer_idx=layer_idx)


class Qwen2ModelKVCompress(Qwen2Model):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2LayerKVCompress(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.config = config

        print('🚗检查🚗 config')
        pprint(config)

        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Union[CompressedKVCache, ClusterCompressedKVCache]] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            cache_position: Optional[torch.LongTensor] = None,
            deltakv_visual_token_mask: Optional[torch.Tensor] = None,
            **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ):
        return_full_hidden = bool(flash_attn_kwargs.pop("deltakv_return_full_hidden", False))
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

        assert isinstance(past_key_values, (CompressedKVCache, ClusterCompressedKVCache))

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        attention_mask_2d = attention_mask if isinstance(attention_mask, torch.Tensor) and attention_mask.dim() == 2 else None
        if position_ids is None:
            if attention_mask_2d is not None:
                position_ids = _position_ids_from_attention_mask(attention_mask_2d)[:, -inputs_embeds.shape[1]:]
            else:
                position_ids = cache_position.unsqueeze(0)

        # It may already have been prepared by e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # Prepare mask arguments
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # Create the masks
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # The sliding window alternating layers are not always activated depending on the config
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        # 修改点：把计算的位置编码直接存起来
        cos, sin = position_embeddings
        if past_key_values.cos is None:
            past_key_values.cos = cos
            past_key_values.sin = sin
        else:
            past_key_values.cos = torch.cat([past_key_values.cos, cos], dim=1)
            past_key_values.sin = torch.cat([past_key_values.sin, sin], dim=1)

        # decoder layers
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
                deltakv_visual_token_mask=deltakv_visual_token_mask,
                deltakv_attention_mask_2d=attention_mask_2d,
                **flash_attn_kwargs,
            )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        # 修改点: 优化显存
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states if return_full_hidden else hidden_states[:, -1:],  # 节省显存
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class Qwen2KVCompress(Qwen2ForCausalLM):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        self.model = Qwen2ModelKVCompress(config)
        self.config = config
        self.post_init()

    def _forward_padded_prefill_by_valid_chunks(
        self,
        *,
        input_ids: torch.LongTensor,
        attention_mask: torch.Tensor,
        past_key_values: Union[CompressedKVCache, ClusterCompressedKVCache],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        kwargs: dict,
    ) -> CausalLMOutputWithPast:
        bs = input_ids.shape[0]
        chunk_size = max(1, int(self.config.chunk_prefill_size))
        valid_mask = attention_mask.to(dtype=torch.bool)
        valid_token_ids = [input_ids[b, valid_mask[b]] for b in range(bs)]
        valid_lengths = [int(tokens.numel()) for tokens in valid_token_ids]
        max_valid_len = max(valid_lengths) if valid_lengths else 0
        if max_valid_len == 0:
            raise ValueError("Cannot prefill a batch with no valid tokens.")

        pad_token_id = self.config.pad_token_id
        if pad_token_id is None:
            pad_token_id = self.config.eos_token_id if self.config.eos_token_id is not None else 0

        cache_attention_mask = input_ids.new_zeros((bs, 0))
        final_logits = None
        last_model_outputs = None

        for start in range(0, max_valid_len, chunk_size):
            chunk_lens = [max(0, min(chunk_size, length - start)) for length in valid_lengths]
            wave_len = max(chunk_lens)
            if wave_len == 0:
                continue

            chunk_ids = input_ids.new_full((bs, wave_len), int(pad_token_id))
            chunk_mask = input_ids.new_zeros((bs, wave_len))
            active_rows = []
            last_token_offsets = []
            for b, chunk_len in enumerate(chunk_lens):
                if chunk_len <= 0:
                    continue
                offset = wave_len - chunk_len
                chunk_ids[b, offset:] = valid_token_ids[b][start : start + chunk_len]
                chunk_mask[b, offset:] = 1
                active_rows.append(b)
                last_token_offsets.append(wave_len - 1)

            cache_attention_mask = torch.cat([cache_attention_mask, chunk_mask], dim=1)
            position_ids = _position_ids_from_attention_mask(cache_attention_mask)[:, -wave_len:]
            cache_start = cache_attention_mask.shape[1] - wave_len
            cache_position = torch.arange(
                cache_start,
                cache_start + wave_len,
                device=input_ids.device,
                dtype=torch.long,
            )

            model_outputs = self.model(
                input_ids=chunk_ids,
                attention_mask=cache_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                deltakv_return_full_hidden=True,
                **kwargs,
            )
            past_key_values = model_outputs.past_key_values
            past_key_values.deltakv_attention_mask = cache_attention_mask
            last_model_outputs = model_outputs

            if active_rows:
                row_idx = torch.tensor(active_rows, device=input_ids.device, dtype=torch.long)
                token_idx = torch.tensor(last_token_offsets, device=input_ids.device, dtype=torch.long)
                selected_hidden = model_outputs.last_hidden_state[row_idx, token_idx, :]
                selected_logits = self.lm_head(selected_hidden)
                if final_logits is None:
                    final_logits = selected_logits.new_zeros((bs, selected_logits.shape[-1]))
                final_logits[row_idx] = selected_logits

        if final_logits is None or last_model_outputs is None:
            raise RuntimeError("Padded prefill produced no logits.")

        return CausalLMOutputWithPast(
            loss=None,
            logits=final_logits[:, None, :],
            past_key_values=past_key_values,
            hidden_states=last_model_outputs.hidden_states,
            attentions=last_model_outputs.attentions,
        )

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[CompressedKVCache, ClusterCompressedKVCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ):
        if input_ids is None:
            raise ValueError("Qwen2KVCompress currently expects input_ids for generation.")
        use_cache = True if use_cache is None else use_cache
        assert use_cache, "Inference model must use cache"

        # --- Chunk Prefill Logic ---
        # 初始化自定义kv cache
        if not is_deltakv_cache_instance(past_key_values, self.config):
            past_key_values = create_deltakv_cache(self.config)

        bs, seq_len = input_ids.shape
        base_past_len = int(past_key_values.get_seq_length())

        if attention_mask is not None:
            attention_mask = attention_mask.to(device=input_ids.device)

        cached_attention_mask = getattr(past_key_values, "deltakv_attention_mask", None)
        if cached_attention_mask is not None and base_past_len > 0:
            current_attention_mask = (
                attention_mask[:, -seq_len:]
                if attention_mask is not None
                else input_ids.new_ones((bs, seq_len))
            )
            attention_mask = torch.cat(
                [cached_attention_mask.to(device=input_ids.device), current_attention_mask],
                dim=1,
            )
            has_padding = bool((attention_mask == 0).any())
        elif attention_mask is None:
            attention_mask = input_ids.new_ones((bs, base_past_len + seq_len))
            has_padding = False
        else:
            has_padding = bool((attention_mask == 0).any())

        if has_padding and bs > 1 and seq_len > 1 and base_past_len == 0:
            return self._forward_padded_prefill_by_valid_chunks(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                kwargs=kwargs,
            )

        if cache_position is None:
            cache_position = torch.arange(
                base_past_len, base_past_len + seq_len, device=input_ids.device, dtype=torch.long
            )
        else:
            cache_position = cache_position.to(device=input_ids.device, dtype=torch.long)

        if position_ids is None:
            full_position_ids = _position_ids_from_attention_mask(attention_mask)
            position_ids = full_position_ids[:, -seq_len:]
        else:
            position_ids = position_ids.to(device=input_ids.device)
            if position_ids.shape[1] != seq_len:
                position_ids = position_ids[:, -seq_len:]

        chunk_size = max(1, int(self.config.chunk_prefill_size))
        outputs = None

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_attention_len = base_past_len + end
            chunk_attention_mask = attention_mask[:, :chunk_attention_len] if has_padding else None
            outputs = super().forward(
                input_ids=input_ids[:, start:end],
                attention_mask=chunk_attention_mask,
                position_ids=position_ids[:, start:end],
                past_key_values=past_key_values,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position[start:end],
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
            past_key_values = outputs.past_key_values
            past_key_values.deltakv_attention_mask = chunk_attention_mask

        return outputs
