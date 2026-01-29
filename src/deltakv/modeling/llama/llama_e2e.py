import torch
import wandb
import torch.nn.functional as F

from typing import Optional, Union
from torch import nn
from transformers.models.llama.modeling_llama import (
    LlamaAttention, Cache, Unpack, FlashAttentionKwargs, apply_rotary_pos_emb,
    Callable, eager_attention_forward, ALL_ATTENTION_FUNCTIONS, LlamaDecoderLayer, LlamaModel, LlamaForCausalLM,
    BaseModelOutputWithPast, CausalLMOutputWithPast, KwargsForCausalLM)
from deltakv.configs.model_config_cls import KVLlamaConfig
from deltakv.modeling.qwen2.qwen2_e2e import create_compressor
from accelerate import Accelerator
from sparsevllm.utils.log import log_once

MODEL_LOG_STEPS = 10
CUR_STEP = 1
accelerator = Accelerator()
# ⚠️只支持多进程多卡训练；也不支持流水线（大概）
CURRENT_RUN_MODE = None


class LlamaAttnKVCompress(LlamaAttention):
    def __init__(self, config: KVLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.seq_chunk_size = config.seq_chunk_size
        self.ref_mode = config.ref_mode
        self.recon_mode = config.recon_mode

        if config.split_kv:
            # 基础压缩器
            self.k_compress_down = create_compressor(is_down=True, config=config)
            self.k_compress_up = create_compressor(is_down=False, config=config)
            self.v_compress_down = create_compressor(is_down=True, config=config)
            self.v_compress_up = create_compressor(is_down=False, config=config)
        else:
            # 统一压缩器
            self.compress_down = create_compressor(is_down=True, config=config)
            self.compress_up = create_compressor(is_down=False, config=config)

        self.buffer_recon_kv = None
        self.buffer_raw_kv = None

    def _single_tensor_comp_then_reconstruct(self, kv, compress_down, compress_up):
        bs, seq_len, dim = kv.shape
        
        # 支持 Sink Token 机制，前若干个 token 不压缩
        sink_size = getattr(self.config, 'num_sink_tokens', 0)
        if seq_len <= sink_size:
            return kv

        kv_sink = kv[:, :sink_size, :]
        kv_rem = kv[:, sink_size:, :]
        rem_len = seq_len - sink_size
        
        kv_chunks = kv_rem.view(bs, -1, self.config.seq_chunk_size, dim)

        use_seq_ref = self.config.seq_chunk_size > 1

        seq_ref = None
        if use_seq_ref:
            if self.ref_mode == 'avg':
                x = kv_rem.transpose(1, 2)
                window_size = self.config.seq_chunk_size
                x_padded = F.pad(x, (window_size - 1, 0))
                y = F.avg_pool1d(x_padded, kernel_size=window_size, stride=1)
                y = y.transpose(1, 2)
                seq_ref = y.view(bs, -1, self.config.seq_chunk_size, dim)
            elif self.ref_mode == 'first':
                seq_ref = kv_chunks[:, :, :1]
            else:
                raise ValueError(f"Unknown ref_mode: {self.ref_mode}")

        if use_seq_ref:
            comp_kv = compress_down(kv_chunks) - compress_down(seq_ref)
            recon_kv = compress_up(comp_kv) + seq_ref
        else:
            # 纯自编码模式
            log_once('纯自编码模式', 'INFO')
            comp_kv = compress_down(kv_chunks)
            recon_kv = compress_up(comp_kv)

        recon_kv_rem = recon_kv.reshape(bs, rem_len, dim)
        return torch.cat([kv_sink, recon_kv_rem], dim=1)

    def comp_then_reconstruct(self, key_states, value_states):
        bs, seq_len, k_dim = key_states.shape

        if not self.config.split_kv:
            _kv_flat = torch.cat([key_states, value_states], dim=-1)
            recon_kv = self._single_tensor_comp_then_reconstruct(
                _kv_flat, self.compress_down, self.compress_up
            )
            self.buffer_recon_kv = recon_kv
            return torch.split(recon_kv, k_dim, dim=-1)
        else:
            recon_k = self._single_tensor_comp_then_reconstruct(
                key_states, self.k_compress_down, self.k_compress_up
            )
            recon_v = self._single_tensor_comp_then_reconstruct(
                value_states, self.v_compress_down, self.v_compress_up
            )
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

        if self.config.collect_kv_before_rope:
            key_states, value_states = self.comp_then_reconstruct(key_states, value_states)

        query_states = query_states.view(hidden_shape).transpose(1, 2)
        key_states = key_states.view(hidden_shape).transpose(1, 2)
        value_states = value_states.view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if not self.config.collect_kv_before_rope:
            raise NotImplementedError

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


class LlamaLayerKVCompress(LlamaDecoderLayer):
    def __init__(self, config: KVLlamaConfig, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = LlamaAttnKVCompress(config=config, layer_idx=layer_idx)


class LlamaModelKVCompress(LlamaModel):
    def __init__(self, config: KVLlamaConfig):
        super().__init__(config)
        self.layers = nn.ModuleList(
            [LlamaLayerKVCompress(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.post_init()


class LlamaKVCompress(LlamaForCausalLM):
    def __init__(self, config: KVLlamaConfig):
        super().__init__(config)
        self.model = LlamaModelKVCompress(config)
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
                if isinstance(mod, LlamaAttnKVCompress):
                    mse_loss = mse_loss + self.mse(mod.buffer_recon_kv, mod.buffer_raw_kv)

            CUR_STEP += 1
            if CUR_STEP % MODEL_LOG_STEPS == 0 and accelerator.is_main_process and wandb.run is not None:
                log_data = {"train/mse_loss": mse_loss.item(), "train/ntp_loss": loss.item()}
                wandb.log(log_data, step=CUR_STEP)

        return CausalLMOutputWithPast(
            loss=loss + mse_loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )