from typing import Optional, Union

import torch
from torch import nn
from transformers.models.qwen3 import modeling_qwen3 as qwen3_modeling
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    Callable,
    FlashAttentionKwargs,
    Qwen3Attention,
    Qwen3DecoderLayer,
    Qwen3ForCausalLM,
    Qwen3Model,
    Unpack,
    apply_rotary_pos_emb,
    eager_attention_forward,
)

from deltakv.configs.model_config_cls import KVQwen3Config
from deltakv.modeling.cache_pipeline import SnapKVCache
from deltakv.modeling.token_select import snapkv_token_selection
from sparsevllm.utils.log import log_once

KwargsForCausalLM = getattr(
    qwen3_modeling,
    "TransformersKwargs",
    getattr(qwen3_modeling, "KwargsForCausalLM", FlashAttentionKwargs),
)


class Qwen3SnapKVAttention(Qwen3Attention):
    def __init__(self, config: KVQwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[SnapKVCache] = None,
        past_key_values: Optional[SnapKVCache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if past_key_value is None:
            past_key_value = past_key_values
        kwargs.pop("position_ids", None)
        kwargs.pop("use_cache", None)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bs, q_len, _ = hidden_states.shape

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        assert past_key_value is not None
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        do_obs = past_key_value.is_last_chunk and self.layer_idx >= self.config.snapkv_num_full_layers

        attention_interface: Callable = eager_attention_forward
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

        if do_obs:
            assert self.config.tail_token_size >= self.config.snapkv_window_size
            candidate_key = key_states[:, :, self.config.num_sink_tokens : -self.config.tail_token_size, :]
            top_token_idx = snapkv_token_selection(
                self,
                query_states,
                candidate_key,
                self.scaling,
                self.config.num_top_tokens,
                pool_kernel_size=self.config.pool_kernel_size,
                output_2d=True,
            )
            past_key_value.delete_tokens(self.layer_idx, top_token_idx)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3SnapKVLayer(Qwen3DecoderLayer):
    def __init__(self, config: KVQwen3Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3SnapKVAttention(config=config, layer_idx=layer_idx)


class Qwen3SnapKVModel(Qwen3Model):
    def __init__(self, config: KVQwen3Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen3SnapKVLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


class Qwen3SnapKVForCausalLM(Qwen3ForCausalLM):
    def __init__(self, config: KVQwen3Config):
        super().__init__(config)
        self.model = Qwen3SnapKVModel(config)
        self.config = config
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[SnapKVCache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ):
        assert input_ids is not None
        if attention_mask is not None:
            assert attention_mask.all(), "目前只支持 bs = 1"
        assert input_ids.shape[0] == 1
        assert position_ids is None and use_cache
        del inputs_embeds, labels, output_attentions, output_hidden_states, cache_position, kwargs

        if past_key_values is None or not isinstance(past_key_values, SnapKVCache):
            if past_key_values:
                assert past_key_values.get_seq_length() == 0
            past_key_values = SnapKVCache(self.config)

        snapkv_window_size = self.config.snapkv_window_size
        chunk_size = self.config.chunk_prefill_size
        outputs = None

        seq_len = input_ids.shape[1]
        if seq_len > 1:
            if seq_len <= snapkv_window_size:
                log_once("只应该在多轮对话中出现这种情况")
                chunk_input_ids = [input_ids]
            else:
                chunk_input_ids = list(input_ids[:, :-snapkv_window_size].split(chunk_size, dim=-1))
                chunk_input_ids.append(input_ids[:, -snapkv_window_size:])
                past_key_values.num_prompt_tokens = seq_len
        else:
            chunk_input_ids = [input_ids]

        for chunk_ids in chunk_input_ids:
            outputs = super().forward(
                chunk_ids,
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=logits_to_keep,
            )
            past_key_values = outputs.past_key_values

        return outputs
