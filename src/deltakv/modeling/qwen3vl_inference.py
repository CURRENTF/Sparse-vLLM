from __future__ import annotations

import json
import os
from typing import Optional, Union

import torch
from torch import nn
from safetensors.torch import load_file
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

from deltakv.configs.model_config_cls import KVQwen3VLTextConfig, parse_full_attn_layers
from deltakv.modeling.cache_factory import create_hf_sparse_cache, is_hf_sparse_cache_instance
from deltakv.modeling.compressor import create_compressor, reshape_and_apply_qk_norm
from deltakv.modeling.hf_common import assert_hf_bs1
from deltakv.modeling.token_select import omnikv_token_selection


def build_qwen3vl_text_deltakv_config(config) -> KVQwen3VLTextConfig:
    if config.text_config.model_type != "qwen3_vl_text":
        raise ValueError(f"Qwen3-VL DeltaKV expects qwen3_vl_text, got {config.text_config.model_type}.")
    text_config = KVQwen3VLTextConfig(**config.text_config.to_dict())
    infer_config = getattr(config, "deltakv_infer_config", None) or {}
    if getattr(config, "deltakv_infer_config_is_native", False):
        text_config.set_native_args(**infer_config)
        text_config.finalize_cluster_args()
    else:
        text_config.set_infer_args(**infer_config)
    return text_config


def load_deltakv_compressor_into_qwen3vl(
    model: nn.Module,
    compressor_path: str,
    device: Union[str, torch.device] = "cpu",
):
    state_dict = load_file(os.path.join(compressor_path, "model.safetensors"), device=str(device))
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.language_model."):
            mapped_key = key
        elif key.startswith("language_model."):
            mapped_key = "model." + key
        elif key.startswith("layers.") or key.startswith("norm."):
            mapped_key = "model.language_model." + key
        elif key.startswith("model."):
            mapped_key = "model.language_model." + key[len("model.") :]
        else:
            mapped_key = "model.language_model." + key
        mapped_state_dict[mapped_key] = value
    incompatible = model.load_state_dict(mapped_state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected DeltaKV compressor keys for Qwen3-VL: {unexpected[:8]}")
    return incompatible


def _top_tokens(config, obs_index: Optional[int], is_prefill: bool):
    value = config.num_top_tokens_in_prefill if is_prefill else config.num_top_tokens
    if isinstance(value, (list, tuple)):
        return value[obs_index]
    if isinstance(value, str) and "," in value:
        return [float(part.strip()) for part in value.split(",")][obs_index]
    return value


class Qwen3VLTextAttnKVCompress(Qwen3VLTextAttention):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        full_layers = parse_full_attn_layers(config.full_attn_layers)
        config.full_attn_layers = full_layers
        self.is_obs_layer = bool(full_layers) and layer_idx in full_layers and (layer_idx + 1) not in full_layers
        self.obs_index = (
            sorted(idx for idx in full_layers if (idx + 1) not in full_layers).index(layer_idx)
            if self.is_obs_layer
            else None
        )
        self.compress_down = create_compressor(is_down=True, config=config)
        self.compress_up = create_compressor(is_down=False, config=config)
        self.config = config
        self.layer_idx = layer_idx

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values=None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ):
        visual_mask = kwargs.pop("deltakv_visual_token_mask", None)
        input_shape = hidden_states.shape[:-1]
        bs, q_len, _ = hidden_states.shape
        assert_hf_bs1((bs, q_len), None)

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)
        if not self.config.collect_kv_before_rope:
            raise NotImplementedError("DeltaKV HF inference expects collect_kv_before_rope=True.")
        key_states, value_states, full_idx = past_key_values.update(
            key_states,
            value_states,
            self.layer_idx,
            {"cache_position": cache_position, "deltakv_visual_token_mask": visual_mask},
            compressor_down=self.compress_down,
            compressor_up=self.compress_up,
        )

        query_shape = (bs, q_len, -1, self.head_dim)
        key_shape = (bs, -1, self.config.num_key_value_heads, self.head_dim)
        query_states, key_states = reshape_and_apply_qk_norm(self, query_states, key_states, query_shape, key_shape)
        value_states = value_states.view(key_shape).transpose(1, 2)

        cur_cos, cur_sin = position_embeddings
        query_states, _ = apply_rotary_pos_emb(query_states, query_states, cur_cos, cur_sin)
        safe_full_idx = full_idx.clamp(min=0, max=past_key_values.cos.shape[1] - 1)
        k_cos = past_key_values.cos.gather(1, safe_full_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
        k_sin = past_key_values.sin.gather(1, safe_full_idx.unsqueeze(-1).expand(-1, -1, self.head_dim))
        _, key_states = apply_rotary_pos_emb(key_states, key_states, k_cos, k_sin)

        attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,
            eager_attention_forward,
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

        if self.is_obs_layer:
            compressed_len = past_key_values.get_observable_compressed_length(q_len)
        else:
            compressed_len = past_key_values.get_compressed_length(self.layer_idx)
        do_obs = (
            bool(self.config.deltakv_use_omnikv_selection)
            and self.is_obs_layer
            and compressed_len > 0
            and (self.config.chunk_prefill_accel_omnikv or q_len == 1)
        )
        if do_obs:
            start = self.config.num_sink_tokens
            candidate_key = key_states[:, :, start : start + compressed_len, :]
            top_token_idx, token_scores = omnikv_token_selection(
                self,
                query_states,
                candidate_key,
                self.scaling,
                _top_tokens(self.config, self.obs_index, q_len > 1),
                pool_kernel_size=self.config.pool_kernel_size,
                last_token_scores=past_key_values.token_scores.get(self.layer_idx),
                score_method=self.config.omnikv_score_method,
            )
            past_key_values.token_scores[self.layer_idx] = token_scores
            past_key_values.top_token_idx[self.layer_idx] = top_token_idx
        if os.getenv("DEBUG") and self.layer_idx == 0:
            print(f"[Qwen3-VL DeltaKV HF] key_states={tuple(key_states.shape)} do_obs={do_obs} q_len={q_len}", flush=True)

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        return self.o_proj(attn_output), attn_weights


class Qwen3VLTextLayerKVCompress(Qwen3VLTextDecoderLayer):
    def __init__(self, config, layer_idx: int):
        super().__init__(config, layer_idx)
        self.self_attn = Qwen3VLTextAttnKVCompress(config=config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        use_cache: bool | None = False,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        del position_ids, use_cache
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            cache_position=cache_position,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class Qwen3VLTextModelKVCompress(Qwen3VLTextModel):
    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([Qwen3VLTextLayerKVCompress(config, i) for i in range(config.num_hidden_layers)])
        self.config = config
        self.post_init()

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        visual_pos_masks: Optional[torch.Tensor] = None,
        deepstack_visual_embeds: Optional[list[torch.Tensor]] = None,
        output_hidden_states: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        return_full_hidden = bool(kwargs.pop("deltakv_return_full_hidden", False))
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if output_attentions:
            raise NotImplementedError("Qwen3-VL HF DeltaKV does not return attention weights.")
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            assert_hf_bs1(tuple(input_ids.shape), attention_mask)
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            assert_hf_bs1(tuple(inputs_embeds.shape[:2]), attention_mask)
        if self.gradient_checkpointing and self.training and use_cache:
            raise RuntimeError("Qwen3-VL HF DeltaKV inference model should not be used with gradient checkpointing.")
        if not is_hf_sparse_cache_instance(past_key_values, self.config):
            raise TypeError("Qwen3VLTextModelKVCompress expects an HF sparse cache created by create_hf_sparse_cache().")
        if cache_position is None:
            past_seen = past_key_values.get_seq_length()
            cache_position = torch.arange(past_seen, past_seen + inputs_embeds.shape[1], device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.view(1, 1, -1).expand(4, inputs_embeds.shape[0], -1)
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
        cos, sin = position_embeddings
        past_key_values.cos = cos if past_key_values.cos is None else torch.cat([past_key_values.cos, cos], dim=1)
        past_key_values.sin = sin if past_key_values.sin is None else torch.cat([past_key_values.sin, sin], dim=1)

        all_hidden_states = () if output_hidden_states else None
        for layer_idx, decoder_layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values,
                cache_position=cache_position,
                deltakv_visual_token_mask=visual_pos_masks,
                **kwargs,
            )
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
            last_hidden_state=hidden_states if return_full_hidden else hidden_states[:, -1:],
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class Qwen3VLDeltaKVModel(Qwen3VLModel):
    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.visual = self._get_default_vision_model(config)
        text_config = build_qwen3vl_text_deltakv_config(config)
        self.deltakv_text_config = text_config
        self.language_model = Qwen3VLTextModelKVCompress(text_config)
        self.rope_deltas = None
        self.post_init()

    @staticmethod
    def _get_default_vision_model(config):
        from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

        return Qwen3VLVisionModel._from_config(config.vision_config)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        **kwargs,
    ) -> Qwen3VLModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_mask = None
        video_mask = None
        deepstack_image_embeds = None
        deepstack_video_embeds = None
        if pixel_values is not None:
            image_outputs = self.get_image_features(pixel_values, image_grid_thw, return_dict=True, **kwargs)
            image_embeds = torch.cat(image_outputs.pooler_output, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            deepstack_image_embeds = image_outputs.deepstack_features
            image_mask, _ = self.get_placeholder_mask(input_ids, inputs_embeds=inputs_embeds, image_features=image_embeds)
            inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        if pixel_values_videos is not None:
            video_outputs = self.get_video_features(pixel_values_videos, video_grid_thw, return_dict=True, **kwargs)
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

        use_cache = kwargs.get("use_cache", None)
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        if use_cache and not is_hf_sparse_cache_instance(past_key_values, self.deltakv_text_config):
            past_key_values = create_hf_sparse_cache(self.deltakv_text_config)

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

        outputs = self.language_model(
            input_ids=None,
            position_ids=position_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )
        return Qwen3VLModelOutputWithPast(
            **outputs,
            rope_deltas=self.rope_deltas,
        )


class Qwen3VLDeltaKVForConditionalGeneration(Qwen3VLForConditionalGeneration):
    def __init__(self, config):
        Qwen3VLPreTrainedModel.__init__(self, config)
        self.model = Qwen3VLDeltaKVModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep=0,
        **kwargs,
    ) -> Union[tuple, Qwen3VLCausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )


def compressor_config_from_checkpoint(compressor_path: Union[str, os.PathLike]) -> dict:
    with open(os.path.join(compressor_path, "config.json"), "r", encoding="utf-8") as handle:
        return json.load(handle)


__all__ = [
    "Qwen3VLDeltaKVForConditionalGeneration",
    "Qwen3VLDeltaKVModel",
    "Qwen3VLTextModelKVCompress",
    "build_qwen3vl_text_deltakv_config",
    "compressor_config_from_checkpoint",
    "load_deltakv_compressor_into_qwen3vl",
]
