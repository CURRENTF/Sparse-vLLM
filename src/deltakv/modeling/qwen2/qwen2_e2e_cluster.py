import os

import torch
import wandb
import torch.nn.functional as F

from typing import Optional, Union
from torch import nn
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, Cache, Unpack, FlashAttentionKwargs, apply_rotary_pos_emb,  # noqa
    Callable, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, Qwen2DecoderLayer, Qwen2Model, Qwen2ForCausalLM,  # noqa
    BaseModelOutputWithPast, CausalLMOutputWithPast, KwargsForCausalLM  # noqa
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
        if config.split_kv:
            self.k_compress_down = create_compressor(is_down=True, config=config)
            self.k_compress_up = create_compressor(is_down=False, config=config)
            self.v_compress_down = create_compressor(is_down=True, config=config)
            self.v_compress_up = create_compressor(is_down=False, config=config)
        else:
            self.compress_down = create_compressor(is_down=True, config=config)
            self.compress_up = create_compressor(is_down=False, config=config)
        
        self.cluster_metric = config.cluster_metric
        self.cluster_on_kv = config.cluster_on_kv

        self.buffer_recon_kv = None
        self.buffer_raw_kv = None
        self.buffer_comp_kv = None
        self.buffer_ideal_res = None

    def _compute_scores(self, feat_rem, prototypes_feat):
        if self.cluster_metric == 'l2':
            return -torch.cdist(feat_rem, prototypes_feat)
        if self.cluster_metric == 'dot':
            return torch.matmul(feat_rem, prototypes_feat.transpose(-1, -2))
        if self.cluster_metric == 'cosine':
            feat_rem_norm = F.normalize(feat_rem, p=2, dim=-1)
            prototypes_feat_norm = F.normalize(prototypes_feat, p=2, dim=-1)
            return torch.matmul(feat_rem_norm, prototypes_feat_norm.transpose(-1, -2))
        raise ValueError(f"Unknown cluster_metric: {self.cluster_metric}")

    def _gather_references(self, scores, prototypes):
        bs, seq_len_rem, _ = scores.shape
        num_prototypes = prototypes.shape[1]
        token_dim = prototypes.shape[-1]
        k = self.config.get_cluster_k_neighbors()
        topk_scores, topk_indices = torch.topk(scores, k=min(k, num_prototypes), dim=-1)
        indices = topk_indices.view(bs, -1)[:, :, None].expand(-1, -1, token_dim)
        gathered = prototypes.gather(dim=1, index=indices).view(bs, seq_len_rem, -1, token_dim)

        if self.config.cluster_soft_assignment:
            temp = self.config.cluster_temp
            weights = F.softmax(topk_scores / temp, dim=-1).to(prototypes.dtype)
            return (gathered * weights.unsqueeze(-1)).sum(dim=2)
        return gathered.mean(dim=2)

    def comp_then_reconstruct(self, key_states, value_states):
        bs, seq_len, k_dim = key_states.shape
        kv_flat = torch.cat([key_states, value_states], dim=-1)

        sink_size = 16
        assert seq_len > sink_size, '训练 seq len 太短了'

        # 2. 选择聚类中心 (Prototypes) - 模拟推理时的采样过程
        # TODO 目前选择聚类中心的方法还是比较粗糙
        cluster_step = max(1, int(1 / self.config.cluster_ratio))
        
        # Sink tokens (0-15) 始终作为聚类中心，因为它们在推理时始终存在
        sink_indices = torch.arange(0, sink_size, device=key_states.device)
        # 剩余部分按 cluster_ratio 采样
        rem_prototype_indices = torch.arange(sink_size, seq_len, cluster_step, device=key_states.device)
        
        all_prototype_indices = torch.cat([sink_indices, rem_prototype_indices])

        # 4. 应用因果掩码: 第 i 个剩余 token (全局索引 i+limit) 只能看到其之前的中心
        seq_len_rem = seq_len - sink_size
        rows = torch.arange(seq_len_rem, device=key_states.device).view(-1, 1) + sink_size
        cols = all_prototype_indices.view(1, -1)
        mask = (cols <= rows).int()  # 允许看到当前位置及之前的中心

        if not self.config.split_kv:
            kv_sink = kv_flat[:, :sink_size, :]
            kv_rem = kv_flat[:, sink_size:, :]
            feat = key_states if not self.cluster_on_kv else kv_flat
            feat_rem = feat[:, sink_size:, :]
            prototypes_feat = feat[:, all_prototype_indices, :]
            prototypes_kv = kv_flat[:, all_prototype_indices, :]

            scores = self._compute_scores(feat_rem, prototypes_feat).masked_fill(mask == 0, float('-inf'))
            gathered_fathers = self._gather_references(scores, prototypes_kv)
            comp_kv_rem = self.compress_down(kv_rem) - self.compress_down(gathered_fathers)

            if os.getenv('ANALYSIS'):
                self.buffer_comp_kv = comp_kv_rem
                self.buffer_ideal_res = kv_rem - gathered_fathers

            kv_recon_rem = (self.compress_up(comp_kv_rem) + gathered_fathers).to(kv_sink.dtype)
            self.buffer_recon_kv = torch.cat([kv_sink, kv_recon_rem], dim=1)
            return torch.split(self.buffer_recon_kv, k_dim, dim=-1)

        k_sink = key_states[:, :sink_size, :]
        v_sink = value_states[:, :sink_size, :]
        k_rem = key_states[:, sink_size:, :]
        v_rem = value_states[:, sink_size:, :]

        k_prototypes_feat = key_states[:, all_prototype_indices, :]
        v_feat = value_states if self.cluster_on_kv else key_states
        v_prototypes_feat = value_states[:, all_prototype_indices, :] if self.cluster_on_kv else k_prototypes_feat

        k_scores = self._compute_scores(key_states[:, sink_size:, :], k_prototypes_feat).masked_fill(mask == 0, float('-inf'))
        v_scores = self._compute_scores(v_feat[:, sink_size:, :], v_prototypes_feat).masked_fill(mask == 0, float('-inf'))

        gathered_k = self._gather_references(k_scores, key_states[:, all_prototype_indices, :])
        gathered_v = self._gather_references(v_scores, value_states[:, all_prototype_indices, :])

        comp_k_rem = self.k_compress_down(k_rem) - self.k_compress_down(gathered_k)
        comp_v_rem = self.v_compress_down(v_rem) - self.v_compress_down(gathered_v)

        if os.getenv('ANALYSIS'):
            self.buffer_comp_kv = torch.cat([comp_k_rem, comp_v_rem], dim=-1)
            self.buffer_ideal_res = torch.cat([k_rem - gathered_k, v_rem - gathered_v], dim=-1)

        recon_k = torch.cat([k_sink, (self.k_compress_up(comp_k_rem) + gathered_k).to(k_sink.dtype)], dim=1)
        recon_v = torch.cat([v_sink, (self.v_compress_up(comp_v_rem) + gathered_v).to(v_sink.dtype)], dim=1)
        self.buffer_recon_kv = torch.cat([recon_k, recon_v], dim=-1)
        return recon_k, recon_v

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
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen2LayerKVClusterCompress(Qwen2DecoderLayer):
    def __init__(self, config: KVQwen2Config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen2AttnKVClusterCompress(config=config, layer_idx=layer_idx)


class Qwen2ModelKVClusterCompress(Qwen2Model):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [Qwen2LayerKVClusterCompress(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


class Qwen2KVClusterCompress(Qwen2ForCausalLM):
    def __init__(self, config: KVQwen2Config):
        super().__init__(config)
        self.model = Qwen2ModelKVClusterCompress(config)
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

            # MSE Loss
            for n, mod in self.named_modules():
                if isinstance(mod, Qwen2AttnKVClusterCompress):
                    mse_loss = mse_loss + self.mse(mod.buffer_recon_kv, mod.buffer_raw_kv)

            CUR_STEP += 1
            if CUR_STEP % MODEL_LOG_STEPS == 0 and accelerator.is_main_process and wandb.run is not None:
                
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
