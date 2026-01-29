import os
import torch
import wandb
import torch.nn.functional as F

from typing import Optional, Union
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention, Cache, Unpack, FlashAttentionKwargs, apply_rotary_pos_emb,
    Callable, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM,
    BaseModelOutputWithPast, CausalLMOutputWithPast, KwargsForCausalLM
)
from deltakv.configs.model_config_cls import KVLlamaConfig
from deltakv.modeling.qwen2.qwen2_e2e import create_compressor
from accelerate import Accelerator

MODEL_LOG_STEPS = 10
CUR_STEP = 1
accelerator = Accelerator()
# ⚠️只支持多进程多卡训练；也不支持流水线（大概）
CURRENT_RUN_MODE = None


class LlamaAttnKVClusterCompress(LlamaAttention):
    def __init__(self, config: KVLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        # 统一的压缩投影器
        self.compress_down = create_compressor(is_down=True, config=config)
        self.compress_up = create_compressor(is_down=False, config=config)
        
        self.cluster_metric = config.cluster_metric
        self.cluster_on_kv = config.cluster_on_kv

        self.buffer_recon_kv = None
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
        
        # Sink tokens (0-15) 始终作为聚类中心
        sink_indices = torch.arange(0, sink_size, device=feat.device)
        # 剩余部分按 cluster_ratio 采样
        rem_prototype_indices = torch.arange(sink_size, seq_len, cluster_step, device=feat.device)
        
        all_prototype_indices = torch.cat([sink_indices, rem_prototype_indices])

        prototypes_feat = feat[:, all_prototype_indices, :]
        prototypes_kv = kv_flat[:, all_prototype_indices, :]

        # 3. 计算剩余 token 与聚类中心的得分
        if self.cluster_metric == 'l2':
            scores = -torch.cdist(feat_rem, prototypes_feat)
        elif self.cluster_metric == 'dot':
            scores = torch.matmul(feat_rem, prototypes_feat.transpose(-1, -2))
        elif self.cluster_metric == 'cosine':
            feat_rem_norm = F.normalize(feat_rem, p=2, dim=-1)
            prototypes_feat_norm = F.normalize(prototypes_feat, p=2, dim=-1)
            scores = torch.matmul(feat_rem_norm, prototypes_feat_norm.transpose(-1, -2))
        else:
            raise ValueError(f"Unknown cluster_metric: {self.cluster_metric}")

        # 4. 应用因果掩码
        num_prototypes = prototypes_feat.shape[1]
        seq_len_rem = seq_len - sink_size
        rows = torch.arange(seq_len_rem, device=feat.device).view(-1, 1) + sink_size
        cols = all_prototype_indices.view(1, -1)
        mask = (cols <= rows).int()
        scores = scores.masked_fill(mask == 0, float('-inf'))

        # 5. 选取聚类中心并生成参考基
        k = max(1, self.config.seq_chunk_size)
        topk_scores, topk_indices = torch.topk(scores, k=min(k, num_prototypes), dim=-1)

        indices = topk_indices.view(bs, -1)[:, :, None].expand(-1, -1, 2 * k_dim)

        if self.config.cluster_soft_assignment:
            temp = self.config.cluster_temp
            weights = F.softmax(topk_scores / temp, dim=-1).to(prototypes_kv.dtype)
            gathered_fathers_raw = prototypes_kv.gather(dim=1, index=indices).view(bs, seq_len_rem, -1, 2 * k_dim)
            gathered_fathers = (gathered_fathers_raw * weights.unsqueeze(-1)).sum(dim=2)
        else:
            gathered_fathers = prototypes_kv.gather(dim=1, index=indices).view(bs, seq_len_rem, -1, 2 * k_dim).mean(dim=2)

        # 6. 计算残差并压缩重建
        comp_kv_rem = self.compress_down(kv_rem) - self.compress_down(gathered_fathers)
        kv_recon_rem = (self.compress_up(comp_kv_rem) + gathered_fathers).to(kv_sink.dtype)

        # 7. Cat back
        self.buffer_recon_kv = torch.cat([kv_sink, kv_recon_rem], dim=1)

        return torch.split(self.buffer_recon_kv, k_dim, dim=-1)

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

        if self.config.collect_kv_before_rope:
            key_states, value_states = self.comp_then_reconstruct(key_states, value_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

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
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class LlamaLayerKVClusterCompress(LlamaDecoderLayer):
    def __init__(self, config: KVLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaAttnKVClusterCompress(config=config, layer_idx=layer_idx)


class LlamaModelKVClusterCompress(LlamaModel):
    def __init__(self, config: KVLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaLayerKVClusterCompress(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


class KVModelCompress(LlamaForCausalLM):
    def __init__(self, config: KVLlamaConfig):
        super().__init__(config)
        self.model = LlamaModelKVClusterCompress(config)
        self.mse = nn.MSELoss()
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

        CURRENT_RUN_MODE = 'comp'
        outputs: BaseModelOutputWithPast = self.model(
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

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        mse_loss = 0
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

            for n, mod in self.named_modules():
                if isinstance(mod, LlamaAttnKVClusterCompress):
                    mse_loss = mse_loss + self.mse(mod.buffer_recon_kv, mod.buffer_raw_kv)

            CUR_STEP += 1
            if CUR_STEP % MODEL_LOG_STEPS == 0 and accelerator.is_main_process:
                log_data = {
                    "train/mse_loss": mse_loss.item(), 
                    "train/ntp_loss": loss.item(),
                }
                wandb.log(log_data, step=CUR_STEP)

        total_loss = loss + mse_loss if loss is not None else None
        if total_loss is not None:
            if os.getenv('MSE_DETACH'):
                total_loss = loss + mse_loss.detach()
            elif os.getenv('NTP_DETACH'):
                total_loss = loss.detach() + mse_loss

        return CausalLMOutputWithPast(
            loss=total_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
