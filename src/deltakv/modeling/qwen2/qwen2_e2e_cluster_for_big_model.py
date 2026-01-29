import os

import torch
import wandb
import torch.nn.functional as F

from typing import Optional, Union
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, Cache, Unpack, FlashAttentionKwargs, apply_rotary_pos_emb,  # noqa
    Callable, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM,  # noqa
    BaseModelOutputWithPast, CausalLMOutputWithPast, KwargsForCausalLM,  # noqa
    DynamicCache, create_causal_mask, logger, create_sliding_window_causal_mask,  # noqa
)
from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.qwen2.qwen2_e2e import create_compressor
from accelerate import Accelerator

MODEL_LOG_STEPS = 10
CUR_STEP = 1
accelerator = Accelerator()
# ⚠️只支持多进程多卡训练；也不支持流水线（大概）
CURRENT_RUN_MODE = None


class Qwen2AttnKVClusterCompress(Qwen2Attention):
    def __init__(self, config: KVQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        # 统一的压缩投影器
        self.compress_down = create_compressor(is_down=True, config=config)
        self.compress_up = create_compressor(is_down=False, config=config)
        
        self.cluster_metric = config.cluster_metric
        self.cluster_on_kv = config.cluster_on_kv

        self.buffer_raw_kv = None

    def comp_then_reconstruct(self, key_states, value_states):
        bs, seq_len, k_dim = key_states.shape
        # kv_flat shape -> bs, seq_len, 2*k_dim
        kv_flat = torch.cat([key_states, value_states], dim=-1)

        sink_size = 16
        assert seq_len > sink_size, '训练 seq len 太短了'

        # 1. 直接 split，去掉前16个token，作为 sink 不压缩
        kv_sink = kv_flat[:, :sink_size, :]
        kv_rem = kv_flat[:, sink_size:, :]

        # 选择用于聚类分配的特征
        if not self.cluster_on_kv:
            feat = key_states
        else:
            feat = kv_flat

        feat_rem = feat[:, sink_size:, :]

        # 2. 选择聚类中心 (Prototypes) - 模拟推理时的采样过程
        cluster_step = max(1, int(1 / self.config.cluster_ratio))
        
        # Sink tokens (0-15) 始终作为聚类中心，因为它们在推理时始终存在
        sink_indices = torch.arange(0, sink_size, device=feat.device)
        # 剩余部分按 cluster_ratio 采样
        rem_prototype_indices = torch.arange(sink_size, seq_len, cluster_step, device=feat.device)
        
        all_prototype_indices = torch.cat([sink_indices, rem_prototype_indices])

        prototypes_feat = feat[:, all_prototype_indices, :]
        prototypes_kv = kv_flat[:, all_prototype_indices, :]
        # shape -> bs, cluster_len, 2*k_dim

        # 3. 计算剩余 token 与聚类中心的得分 (bs, seq_len_rem, num_prototypes)
        if self.cluster_metric == 'l2':
            # cdist 返回正值，取负值以便 topk 选取最近的
            scores = -torch.cdist(feat_rem, prototypes_feat)
        elif self.cluster_metric == 'dot':
            scores = torch.matmul(feat_rem, prototypes_feat.transpose(-1, -2))
        elif self.cluster_metric == 'cosine':
            feat_rem_norm = F.normalize(feat_rem, p=2, dim=-1)
            prototypes_feat_norm = F.normalize(prototypes_feat, p=2, dim=-1)
            scores = torch.matmul(feat_rem_norm, prototypes_feat_norm.transpose(-1, -2))
        else:
            raise ValueError(f"Unknown cluster_metric: {self.cluster_metric}")

        # 4. 应用因果掩码: 第 i 个剩余 token (全局索引 i+limit) 只能看到其之前的中心
        num_prototypes = prototypes_feat.shape[1]
        seq_len_rem = seq_len - sink_size
        rows = torch.arange(seq_len_rem, device=feat.device).view(-1, 1) + sink_size
        cols = all_prototype_indices.view(1, -1)
        mask = (cols <= rows).int()  # 允许看到当前位置及之前的中心
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. 选取聚类中心并生成参考基
        # 选取最相似的 k 个聚类中心并采用 加权/直接 平均作为参考基
        k = max(1, self.config.seq_chunk_size)
        topk_scores, topk_indices = torch.topk(scores, k=min(k, num_prototypes), dim=-1)
        # shape -> bs, seq_len_rem, k

        # 从 prototypes_kv 中 gather
        indices = topk_indices.view(bs, -1)[:, :, None].expand(-1, -1, 2 * k_dim)

        if self.config.cluster_soft_assignment:
            # 计算权重 (使用 config 中固定的温度)
            temp = self.config.cluster_temp
            weights = F.softmax(topk_scores / temp, dim=-1).to(prototypes_kv.dtype)
            # shape -> bs, seq_len_rem * k, 1 -> bs, seq_len_rem * k, 2 * k_dim
            gathered_fathers_raw = prototypes_kv.gather(dim=1, index=indices).view(bs, seq_len_rem, -1, 2 * k_dim)
            
            # 加权平均
            gathered_fathers = (gathered_fathers_raw * weights.unsqueeze(-1)).sum(dim=2)
        else:
            gathered_fathers = prototypes_kv.gather(dim=1, index=indices).view(bs, seq_len_rem, -1, 2 * k_dim).mean(dim=2)

        # 6. 计算残差并压缩重建 rem 部分
        # Formula: comp_kv = down(kv) - down(father)
        # Recon: recon_kv = up(comp_kv) + father
        comp_kv_rem = self.compress_down(kv_rem) - self.compress_down(gathered_fathers)
        kv_recon_rem = (self.compress_up(comp_kv_rem) + gathered_fathers).to(kv_sink.dtype)

        # 7. Cat back
        recon_kv = torch.cat([kv_sink, kv_recon_rem], dim=1)
        
        # 直接计算 MSE Loss 并返回
        mse_loss = F.mse_loss(recon_kv, self.buffer_raw_kv)
        k, v = torch.split(recon_kv, k_dim, dim=-1)

        return k, v, mse_loss

    def forward(
            self,
            hidden_states: torch.Tensor,
            position_embeddings: tuple[torch.Tensor, torch.Tensor],
            attention_mask: Optional[torch.Tensor],
            past_key_value: Optional[Cache] = None,
            cache_position: Optional[torch.LongTensor] = None,
            **kwargs: Unpack[FlashAttentionKwargs],
    ):
        if CURRENT_RUN_MODE == 'raw':
            return self.raw_forward(hidden_states, position_embeddings, attention_mask, past_key_value, cache_position, **kwargs)

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        bs, seq_len, k_dim = key_states.shape

        mse_loss = None
        if self.config.collect_kv_before_rope:
            key_states, value_states, mse_loss = self.comp_then_reconstruct(key_states, value_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        # now shape --> bs, heads, seq_len, head_dim

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not self.config.collect_kv_before_rope:
            raise NotImplementedError("Cluster compression currently only supports collect_kv_before_rope=True")

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, mse_loss

    def raw_forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ):
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        self.buffer_raw_kv = torch.cat([key_states, value_states], dim=-1)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

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

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, torch.tensor(0.0, device=hidden_states.device)


class Qwen2LayerKVClusterCompress(Qwen2DecoderLayer):
    def __init__(self, config: KVQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2AttnKVClusterCompress(config=config, layer_idx=layer_idx)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Cache] = None,
            output_attentions: Optional[bool] = False,
            use_cache: Optional[bool] = False,
            cache_position: Optional[torch.LongTensor] = None,
            position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
            **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights, mse_loss = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs + (mse_loss, )


class Qwen2ModelKVClusterCompress(Qwen2Model):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2LayerKVClusterCompress(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
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

        if not isinstance(past_key_values, (type(None), Cache)):
            raise ValueError("The `past_key_values` should be either a `Cache` object or `None`.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
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

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        mse_loss = 0
        for layer_idx, decoder_layer in enumerate(self.layers[: self.config.num_hidden_layers]):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            # print(f'at Layer{layer_idx}')
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
            mse_loss = mse_loss + layer_outputs[-1]

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), mse_loss


class Qwen2KVClusterCompress(Qwen2ForCausalLM):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        self.model = Qwen2ModelKVClusterCompress(config)
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
        global CUR_STEP, CURRENT_RUN_MODE
        
        # 1. 运行原始前向传播，收集目标 KV
        CURRENT_RUN_MODE = 'raw'
        with torch.no_grad():
            if labels is not None:
                self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    inputs_embeds=inputs_embeds,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    cache_position=cache_position,
                    **kwargs,
                )

        # 2. 运行带压缩的前向传播
        CURRENT_RUN_MODE = 'comp'
        model_outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            cache_position=cache_position,
            **kwargs,
        )
        
        outputs, mse_loss = model_outputs

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            CUR_STEP += 1
            if CUR_STEP % MODEL_LOG_STEPS == 0 and accelerator.is_main_process and wandb.run is not None:
                log_data = {
                    "train/mse_loss": mse_loss.item(), 
                    "train/ntp_loss": loss.item(),
                }
                wandb.log(log_data, step=CUR_STEP)

        return CausalLMOutputWithPast(
            loss=loss + mse_loss if loss is not None else mse_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
