import os
import torch

from dataclasses import dataclass
from pprint import pprint
from typing import Optional
from torch import nn
from accelerate import Accelerator
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, Cache, Unpack, FlashAttentionKwargs, apply_rotary_pos_emb,
    Callable, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, Qwen3DecoderLayer, Qwen3Model, Qwen3ForCausalLM,
    logger, BaseModelOutputWithPast, create_sliding_window_causal_mask, create_causal_mask,
    Union, KwargsForCausalLM,
)

from deltakv.configs.model_config_cls import KVQwen3Config
from deltakv.modeling.kv_cache import CompressedKVCache, ClusterCompressedKVCache
from deltakv.modeling.qwen3.qwen3_e2e import create_compressor
from deltakv.modeling.token_select import omnikv_token_selection

accelerator = Accelerator()


@dataclass
class Output(BaseModelOutputWithPast):
    last_hidden_state: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[tuple[torch.FloatTensor, ...]] = None
    compress_loss: Optional[torch.FloatTensor] = None
class Qwen3AttnKVCompress(Qwen3Attention):
    def __init__(self, config: KVQwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)

        if isinstance(config.full_attn_layers, str):
            config.full_attn_layers = config.full_attn_layers.split(',')
        full_layers = [int(_) for _ in config.full_attn_layers]
        assert 0 in full_layers

        self.is_full_layer = (layer_idx in full_layers)
        self.is_obs_layer = (self.is_full_layer and (layer_idx + 1) not in full_layers)

        if self.is_obs_layer:
            all_obs_layers = sorted([idx for idx in full_layers if (idx + 1) not in full_layers])
            self.obs_index = all_obs_layers.index(layer_idx)
        else:
            self.obs_index = None

        self.compress_down = create_compressor(is_down=True, config=config)
        self.compress_up = create_compressor(is_down=False, config=config)
        self.config = config
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Union[CompressedKVCache, ClusterCompressedKVCache]] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        bs, q_len, _ = hidden_states.shape

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        if self.config.collect_kv_before_rope:
            cache_kwargs = {"cache_position": cache_position}
            res = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs,
                compressor_down=self.compress_down,
                compressor_up=self.compress_up,
            )
            key_states, value_states, full_idx = res
        else:
            raise NotImplementedError

        query_states = query_states.view(bs, q_len, self.config.num_attention_heads, self.head_dim)
        if hasattr(self, "q_norm"):
            query_states = self.q_norm(query_states)
        query_states = query_states.transpose(1, 2)

        key_states = key_states.view(bs, -1, self.config.num_key_value_heads, self.head_dim)
        if hasattr(self, "k_norm"):
            key_states = self.k_norm(key_states)
        key_states = key_states.transpose(1, 2)

        value_states = value_states.view(bs, -1, self.config.num_key_value_heads, self.head_dim).transpose(1, 2)

        cur_cos, cur_sin = position_embeddings
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cur_cos, cur_sin)

        k_cos = past_key_value.cos.gather(1, full_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
        k_sin = past_key_value.sin.gather(1, full_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
        _, key_states = apply_rotary_pos_emb(key_states, key_states, k_cos, k_sin)

        attention_interface: Callable = eager_attention_forward

        sink_size = self.config.num_sink_tokens
        is_prefill, is_decode = (q_len > 1), (q_len == 1)
        compressed_len = (
            past_key_value.get_seq_length() - self.config.tail_token_size - q_len - sink_size
        ) // self.config.tail_token_size * self.config.tail_token_size
        use_omnikv_selection = bool(self.config.deltakv_use_omnikv_selection)
        do_obs = (
            use_omnikv_selection
            and self.is_obs_layer
            and compressed_len > 0
            and (self.config.chunk_prefill_accel_omnikv or is_decode)
        )

        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

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

        if os.getenv("DEBUG"):
            if self.layer_idx in {0, 14, 15}:
                print(f"L{self.layer_idx}  {key_states.shape=}  {do_obs=}  {q_len=}")

        if do_obs:
            candidate_key = key_states[:, :, sink_size:sink_size + compressed_len, :]

            num_top_tokens = self.config.num_top_tokens_in_prefill if is_prefill else self.config.num_top_tokens
            if isinstance(num_top_tokens, (list, tuple)):
                num_top_tokens = num_top_tokens[self.obs_index]
            elif isinstance(num_top_tokens, str) and ',' in num_top_tokens:
                num_top_tokens = [float(x.strip()) for x in num_top_tokens.split(',')]
                num_top_tokens = num_top_tokens[self.obs_index]

            last_token_scores = past_key_value.token_scores.get(self.layer_idx, None)
            top_token_idx, token_scores = omnikv_token_selection(
                self,
                query_states,
                candidate_key,
                self.scaling,
                num_top_tokens,
                pool_kernel_size=self.config.pool_kernel_size,
                last_token_scores=last_token_scores,
                score_method=self.config.omnikv_score_method,
            )

            past_key_value.token_scores[self.layer_idx] = token_scores
            past_key_value.top_token_idx[self.layer_idx] = top_token_idx

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3LayerKVCompress(Qwen3DecoderLayer):
    def __init__(self, config: KVQwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3AttnKVCompress(config=config, layer_idx=layer_idx)


class Qwen3ModelKVCompress(Qwen3Model):
    def __init__(self, config: KVQwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3LayerKVCompress(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.config = config

        print("🚗检查🚗 config")
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

        assert isinstance(past_key_values, (CompressedKVCache, ClusterCompressedKVCache))

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


class Qwen3KVCompress(Qwen3ForCausalLM):
    def __init__(self, config: KVQwen3Config):
        super().__init__(config)
        self.model = Qwen3ModelKVCompress(config)
        self.config = config
        self.post_init()

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
        del attention_mask, position_ids, labels, output_attentions, output_hidden_states, cache_position, logits_to_keep, kwargs
        assert input_ids is not None
        assert inputs_embeds is None
        assert input_ids.shape[0] == 1
        assert use_cache, "Inference model must use cache"

        if not isinstance(past_key_values, (CompressedKVCache, ClusterCompressedKVCache)):
            if self.config.use_cluster:
                past_key_values = ClusterCompressedKVCache(config=self.config)
            else:
                past_key_values = CompressedKVCache(config=self.config)

        chunk_size = self.config.chunk_prefill_size
        outputs = None

        for chunk_input_ids in input_ids.split(chunk_size, dim=-1):
            outputs = super().forward(chunk_input_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values

        return outputs
