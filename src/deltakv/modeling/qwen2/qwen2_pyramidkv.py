import torch
from typing import Optional, Union
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, Unpack, FlashAttentionKwargs, Callable, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM,
    KwargsForCausalLM, apply_rotary_pos_emb
)
from deltakv.modeling.kv_cache import SnapKVCache
from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.token_select import snapkv_token_selection
from sparsevllm.utils.log import log_once


class Qwen2PyramidKVAttention(Qwen2Attention):
    def __init__(self, config: KVQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.config = config
        self.layer_idx = layer_idx
        
        # 计算当前层的 budget ratio
        least_layer = config.pyramidkv_least_layer or (config.num_hidden_layers - 1)
        start_layer = config.pyramidkv_start_layer
        start_ratio = config.pyramidkv_start_ratio
        least_ratio = config.pyramidkv_least_ratio
        
        if self.layer_idx < start_layer:
            self.layer_ratio = 1.0
        elif self.layer_idx <= least_layer:
            # 线性插值
            assert least_layer > start_layer
            self.layer_ratio = start_ratio - (start_ratio - least_ratio) * ((self.layer_idx - start_layer) / (least_layer - start_layer))
        else:
            self.layer_ratio = least_ratio
            
        # 记录日志，仅记录一次所有层的 ratio (通过 log_once 和拼接字符串实现)
        all_ratios = []
        for i in range(config.num_hidden_layers):
            if i < start_layer:
                r = 1.0
            elif i <= least_layer:
                if least_layer <= start_layer:
                    r = start_ratio
                else:
                    r = start_ratio - (start_ratio - least_ratio) * ((i - start_layer) / (least_layer - start_layer))
            else:
                r = least_ratio
            all_ratios.append(f"{r:.3f}")
        
        log_once(f"PyramidKV Layer Ratios: {', '.join(all_ratios)}", "INFO")

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[SnapKVCache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)
        bs, q_len, ___ = hidden_states.shape

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        assert past_key_value is not None
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward

        # 只有在处理 Prompt 的最后一个 chunk 时，才触发压缩逻辑
        do_obs = past_key_value.is_last_chunk

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

            # 1. 截取“候选区”的 Key
            candidate_key = key_states[:, :, self.config.num_sink_tokens: -self.config.tail_token_size, :]
            
            # 2. Token Selection: 使用当前层的 layer_ratio
            top_token_idx = snapkv_token_selection(
                self, 
                query_states, 
                candidate_key, 
                self.scaling, 
                self.layer_ratio, 
                pool_kernel_size=getattr(self.config, 'pool_kernel_size', 5),
                output_2d=True
            )

            # 3. 执行压缩
            past_key_value.delete_tokens(self.layer_idx, top_token_idx)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2PyramidKVLayer(Qwen2DecoderLayer):
    def __init__(self, config: KVQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2PyramidKVAttention(config=config, layer_idx=layer_idx)


class Qwen2PyramidKVModel(Qwen2Model):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2PyramidKVLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


class Qwen2PyramidKVForCausalLM(Qwen2ForCausalLM):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        self.model = Qwen2PyramidKVModel(config)
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
        assert input_ids is not None and attention_mask is None
        assert input_ids.shape[0] == 1
        assert position_ids is None and use_cache

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
                log_once('只应该在多轮对话中出现这种情况')
                chunk_input_ids = [input_ids]
            else:
                chunk_input_ids = list(input_ids[:, :-snapkv_window_size].split(chunk_size, dim=-1)) + [input_ids[:, -snapkv_window_size:]]
                past_key_values.num_prompt_tokens = seq_len
        else:
            chunk_input_ids = [input_ids]

        for _ipt_ids in chunk_input_ids:
            outputs = super().forward(_ipt_ids, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values

        return outputs
