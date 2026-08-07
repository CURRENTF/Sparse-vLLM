from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn
from transformers import DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_vl.modeling_qwen3_vl import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3VLCausalLMOutputWithPast,
    Qwen3VLForConditionalGeneration,
    Qwen3VLModel,
    Qwen3VLModelOutputWithPast,
    Qwen3VLPreTrainedModel,
    Qwen3VLTextAttention,
    Qwen3VLTextDecoderLayer,
    Qwen3VLTextModel,
    apply_rotary_pos_emb,
    create_causal_mask,
    eager_attention_forward,
)

from deltakv.configs.model_config_cls import KVQwen3VLTextConfig
from deltakv.modeling.compressor import create_compressor, reshape_and_apply_qk_norm


def build_qwen3vl_text_deltakv_config(config) -> KVQwen3VLTextConfig:
    if config.text_config.model_type != "qwen3_vl_text":
        raise ValueError(f"Qwen3-VL DeltaKV expects qwen3_vl_text, got {config.text_config.model_type}.")
    text_config = KVQwen3VLTextConfig(**config.text_config.to_dict())
    infer_config = getattr(config, "deltakv_infer_config", None) or {}
    text_config.set_infer_args(**infer_config)
    return text_config


_RUN_MODE = {"value": "comp"}


class Qwen3VLTextAttnKVClusterCompress(Qwen3VLTextAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        if getattr(config, "split_kv", False):
            self.k_compress_down = create_compressor(is_down=True, config=config)
            self.k_compress_up = create_compressor(is_down=False, config=config)
            self.v_compress_down = create_compressor(is_down=True, config=config)
            self.v_compress_up = create_compressor(is_down=False, config=config)
        else:
            self.compress_down = create_compressor(is_down=True, config=config)
            self.compress_up = create_compressor(is_down=False, config=config)
        self.cluster_metric = config.cluster_metric
        self.cluster_on_kv = config.cluster_on_kv
        self.buffer_raw_kv = None
        self.buffer_comp_kv = None
        self.buffer_ideal_res = None
        self.buffer_recon_kv = None

    def _scores(self, a, b):
        if self.cluster_metric == "l2":
            return -torch.cdist(a, b)
        if self.cluster_metric == "dot":
            return torch.matmul(a, b.transpose(-1, -2))
        if self.cluster_metric == "cosine":
            return torch.matmul(F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1).transpose(-1, -2))
        raise ValueError(f"Unknown cluster_metric: {self.cluster_metric}")

    def _refs(self, scores, prototypes):
        bs, seq_len, _ = scores.shape
        token_dim = prototypes.shape[-1]
        k = min(self.config.get_cluster_neighbor_count(), prototypes.shape[1])
        topk_scores, topk_indices = torch.topk(scores, k=k, dim=-1)
        indices = topk_indices.reshape(bs, -1)[:, :, None].expand(-1, -1, token_dim)
        gathered = prototypes.gather(dim=1, index=indices).view(bs, seq_len, k, token_dim)
        if self.config.cluster_soft_assignment:
            weights = F.softmax(topk_scores / self.config.cluster_temp, dim=-1).to(prototypes.dtype)
            return (gathered * weights.unsqueeze(-1)).sum(dim=2)
        return gathered.mean(dim=2)

    def _plan(self, states):
        sink_size = 16
        if states.shape[1] <= sink_size:
            raise ValueError(f"cluster_e2e_big training requires seq_len > {sink_size}; got {states.shape[1]}.")
        step = max(1, int(1 / self.config.cluster_ratio))
        sink = torch.arange(0, sink_size, device=states.device)
        rem = torch.arange(sink_size, states.shape[1], step, device=states.device)
        centers = torch.cat([sink, rem])
        rows = torch.arange(states.shape[1] - sink_size, device=states.device).view(-1, 1) + sink_size
        mask = centers.view(1, -1) <= rows
        return sink_size, centers, mask

    def comp_then_reconstruct(self, key_states, value_states):
        _, _, k_dim = key_states.shape
        kv_flat = torch.cat([key_states, value_states], dim=-1)
        sink_size, centers, mask = self._plan(key_states)
        if not getattr(self.config, "split_kv", False):
            kv_sink = kv_flat[:, :sink_size]
            kv_rem = kv_flat[:, sink_size:]
            feat = kv_flat if self.cluster_on_kv else key_states
            scores = self._scores(feat[:, sink_size:], feat[:, centers]).masked_fill(~mask.unsqueeze(0), float("-inf"))
            refs = self._refs(scores, kv_flat[:, centers])
            comp = self.compress_down(kv_rem) - self.compress_down(refs)
            self.buffer_comp_kv = comp
            self.buffer_ideal_res = kv_rem - refs
            recon_rem = (self.compress_up(comp) + refs).to(kv_sink.dtype)
            recon = torch.cat([kv_sink, recon_rem], dim=1)
            self.buffer_recon_kv = recon
            return (*torch.split(recon, k_dim, dim=-1), F.mse_loss(recon, self.buffer_raw_kv))

        k_sink, v_sink = key_states[:, :sink_size], value_states[:, :sink_size]
        k_rem, v_rem = key_states[:, sink_size:], value_states[:, sink_size:]
        k_scores = self._scores(k_rem, key_states[:, centers]).masked_fill(~mask.unsqueeze(0), float("-inf"))
        v_feat = value_states if self.cluster_on_kv else key_states
        v_centers = value_states[:, centers] if self.cluster_on_kv else key_states[:, centers]
        v_scores = self._scores(v_feat[:, sink_size:], v_centers).masked_fill(~mask.unsqueeze(0), float("-inf"))
        ref_k = self._refs(k_scores, key_states[:, centers])
        ref_v = self._refs(v_scores, value_states[:, centers])
        comp_k = self.k_compress_down(k_rem) - self.k_compress_down(ref_k)
        comp_v = self.v_compress_down(v_rem) - self.v_compress_down(ref_v)
        self.buffer_comp_kv = torch.cat([comp_k, comp_v], dim=-1)
        self.buffer_ideal_res = torch.cat([k_rem - ref_k, v_rem - ref_v], dim=-1)
        recon_k = torch.cat([k_sink, (self.k_compress_up(comp_k) + ref_k).to(k_sink.dtype)], dim=1)
        recon_v = torch.cat([v_sink, (self.v_compress_up(comp_v) + ref_v).to(v_sink.dtype)], dim=1)
        recon = torch.cat([recon_k, recon_v], dim=-1)
        self.buffer_recon_kv = recon
        return recon_k, recon_v, F.mse_loss(recon, self.buffer_raw_kv)

    def _shape_qkv(self, query_states, key_states, value_states, input_shape):
        hidden_shape = (*input_shape, -1, self.head_dim)
        query_states, key_states = reshape_and_apply_qk_norm(self, query_states, key_states, hidden_shape)
        value_states = value_states.view(hidden_shape).transpose(1, 2)
        return query_states, key_states, value_states

    def _attention_out(self, input_shape, query_states, key_states, value_states, attention_mask, kwargs, raw=False):
        attention_interface = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                self.config._attn_implementation, eager_attention_forward
            )
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
        mse = torch.tensor(0.0, device=query_states.device) if raw else None
        return self.o_proj(attn_output.reshape(*input_shape, -1).contiguous()), attn_weights, mse

    def raw_forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        self.buffer_raw_kv = torch.cat([key_states, value_states], dim=-1)
        query_states, key_states, value_states = self._shape_qkv(query_states, key_states, value_states, input_shape)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        return self._attention_out(input_shape, query_states, key_states, value_states, attention_mask, kwargs, raw=True)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        if _RUN_MODE["value"] == "raw":
            return self.raw_forward(hidden_states, position_embeddings, attention_mask, past_key_values, **kwargs)
        input_shape = hidden_states.shape[:-1]
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if not self.config.collect_kv_before_rope:
            raise NotImplementedError("cluster_e2e_big requires collect_kv_before_rope=True.")
        key_states, value_states, mse = self.comp_then_reconstruct(key_states, value_states)
        query_states, key_states, value_states = self._shape_qkv(query_states, key_states, value_states, input_shape)
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)
        out, weights, _ = self._attention_out(input_shape, query_states, key_states, value_states, attention_mask, kwargs)
        return out, weights, mse


class Qwen3VLTextLayerKVClusterCompress(Qwen3VLTextDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3VLTextAttnKVClusterCompress(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states,
        position_embeddings,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=False,
        **kwargs,
    ):
        del position_ids, use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights, mse_loss = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + self.mlp(hidden_states)
        return hidden_states, self_attn_weights, mse_loss


class Qwen3VLTextModelKVClusterCompress(Qwen3VLTextModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([Qwen3VLTextLayerKVClusterCompress(config, i) for i in range(config.num_hidden_layers)])
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        visual_pos_masks=None,
        deepstack_visual_embeds=None,
        output_hidden_states=None,
        output_attentions=None,
        **kwargs,
    ):
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False
        if not isinstance(past_key_values, (type(None), DynamicCache)):
            raise ValueError("cluster_e2e_big training expects a DynamicCache or None.")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
        elif position_ids.ndim == 2:
            position_ids = position_ids[None, ...].expand(4, position_ids.shape[0], -1)

        if position_ids.ndim == 3 and position_ids.shape[0] == 4:
            text_position_ids = position_ids[0]
            rope_position_ids = position_ids[1:]
        else:
            text_position_ids = None
            rope_position_ids = position_ids

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=text_position_ids,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, rope_position_ids)
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        mse_loss = torch.tensor(0.0, device=inputs_embeds.device)

        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states, self_attn_weights, layer_mse = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                **kwargs,
            )
            mse_loss = mse_loss + layer_mse
            if output_attentions:
                all_self_attns += (self_attn_weights,)
            if deepstack_visual_embeds is not None and layer_idx in range(len(deepstack_visual_embeds)):
                hidden_states = self._deepstack_process(
                    hidden_states,
                    visual_pos_masks,
                    deepstack_visual_embeds[layer_idx],
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        ), mse_loss


class Qwen3VLDeltaKVTrainingModel(Qwen3VLModel):
    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = self._get_default_vision_model(config)
        text_config = build_qwen3vl_text_deltakv_config(config)
        self.deltakv_text_config = text_config
        self.language_model = Qwen3VLTextModelKVClusterCompress(text_config)
        self.rope_deltas = None
        self.post_init()

    @staticmethod
    def _get_default_vision_model(config):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        return Qwen3VLVisionModel._from_config(config.vision_config)

    def forward(self, *args, **kwargs):
        input_ids = kwargs.get("input_ids", args[0] if args else None)
        attention_mask = kwargs.get("attention_mask")
        position_ids = kwargs.get("position_ids")
        past_key_values = kwargs.get("past_key_values")
        inputs_embeds = kwargs.get("inputs_embeds")
        pixel_values = kwargs.get("pixel_values")
        pixel_values_videos = kwargs.get("pixel_values_videos")
        image_grid_thw = kwargs.get("image_grid_thw")
        video_grid_thw = kwargs.get("video_grid_thw")
        mm_token_type_ids = kwargs.get("mm_token_type_ids")

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_image_embeds = image_outputs.deepstack_features
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True)
            video_embeds = torch.cat(video_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_video_embeds = video_outputs.deepstack_features
            _, video_mask = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, video_features=video_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

        visual_pos_masks = None
        deepstack_visual_embeds = None
        if image_mask is not None and video_mask is not None:
            image_mask = image_mask[..., 0]
            video_mask = video_mask[..., 0]
            visual_pos_masks = image_mask | video_mask
            deepstack_visual_embeds = []
            image_mask_joint = image_mask[visual_pos_masks]
            video_mask_joint = video_mask[visual_pos_masks]
            for img_embed, vid_embed in zip(deepstack_image_embeds, deepstack_video_embeds):
                embed_joint = img_embed.new_zeros(visual_pos_masks.sum(), img_embed.shape[-1]).to(img_embed.device)
                embed_joint[image_mask_joint, :] = img_embed
                embed_joint[video_mask_joint, :] = vid_embed
                deepstack_visual_embeds.append(embed_joint)
        elif image_mask is not None:
            visual_pos_masks = image_mask[..., 0]
            deepstack_visual_embeds = deepstack_image_embeds
        elif video_mask is not None:
            visual_pos_masks = video_mask[..., 0]
            deepstack_visual_embeds = deepstack_video_embeds

        if position_ids is None:
            position_ids = self.compute_3d_position_ids(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                mm_token_type_ids=mm_token_type_ids,
            )

        _RUN_MODE["value"] = "raw"
        with torch.no_grad():
            self.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
        _RUN_MODE["value"] = "comp"
        try:
            outputs, mse_loss = self.language_model(
                input_ids=None,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                visual_pos_masks=visual_pos_masks,
                deepstack_visual_embeds=deepstack_visual_embeds,
            )
        finally:
            _RUN_MODE["value"] = "comp"
        self._last_mse_loss = mse_loss.detach()
        return Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLDeltaKVForCompressorTraining(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLDeltaKVTrainingModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def forward(self, *args, **kwargs):
        outputs = super().forward(*args, **kwargs)
        mse_loss = getattr(self.model, "_last_mse_loss", None)
        if mse_loss is None:
            raise RuntimeError("Qwen3-VL compressor training did not produce mse_loss.")
        self._last_ntp_loss = outputs.loss.detach() if outputs.loss is not None else None
        self._last_mse_loss = mse_loss.detach()
        total_loss = (outputs.loss + mse_loss) if outputs.loss is not None else mse_loss
        if outputs.loss is not None and os.getenv("MSE_DETACH"):
            total_loss = outputs.loss + mse_loss.detach()
        elif outputs.loss is not None and os.getenv("NTP_DETACH"):
            total_loss = outputs.loss.detach() + mse_loss
        return Qwen3VLCausalLMOutputWithPast(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=outputs.rope_deltas,
        )


__all__ = [
    "Qwen3VLDeltaKVForCompressorTraining",
    "Qwen3VLTextModelKVClusterCompress",
    "build_qwen3vl_text_deltakv_config",
]
