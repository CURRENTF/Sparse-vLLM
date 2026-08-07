import math
import os
from typing import Optional, Union

import torch
from torch import nn
from safetensors.torch import load_file
from transformers import AutoModel
from transformers.models.llava_onevision import modeling_llava_onevision as llava_ov_modeling
from transformers.models.llava_onevision.modeling_llava_onevision import (
    FlashAttentionKwargs,
    LlavaOnevisionCausalLMOutputWithPast,
    LlavaOnevisionForConditionalGeneration,
    LlavaOnevisionModel,
    LlavaOnevisionModelOutputWithPast,
    LlavaOnevisionMultiModalProjector,
    LlavaOnevisionPreTrainedModel,
    Unpack,
)
from transformers.utils import is_torchdynamo_compiling

KwargsForCausalLM = getattr(
    llava_ov_modeling,
    "KwargsForCausalLM",
    llava_ov_modeling.TransformersKwargs,
)

from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.cache_factory import create_hf_sparse_cache, is_hf_sparse_cache_instance
from deltakv.modeling.cache_pipeline import SnapKVCache
from deltakv.modeling.qwen2_inference import Qwen2ModelKVCompress
from deltakv.modeling.qwen2.qwen2_snapkv import Qwen2SnapKVModel


def build_llava_text_deltakv_config(config) -> KVQwen2Config:
    if config.text_config.model_type != "qwen2":
        raise ValueError(f"LLaVA-OneVision DeltaKV currently supports qwen2 text backbones, got {config.text_config.model_type}.")

    text_config = KVQwen2Config(**config.text_config.to_dict())
    infer_config = getattr(config, "deltakv_infer_config", None) or {}
    if getattr(config, "deltakv_infer_config_is_native", False):
        text_config.set_native_args(**infer_config)
        text_config.finalize_cluster_args()
    else:
        text_config.set_infer_args(**infer_config)
    return text_config


def load_deltakv_compressor_into_llava(model: nn.Module, compressor_path: str, device: Union[str, torch.device] = "cpu"):
    state_dict = load_file(os.path.join(compressor_path, "model.safetensors"), device=str(device))
    mapped_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("model.language_model.model."):
            mapped_key = "model.language_model." + key[len("model.language_model.model."):]
        elif key.startswith("model.language_model."):
            mapped_key = key
        elif key.startswith("language_model.model."):
            mapped_key = "model.language_model." + key[len("language_model.model."):]
        elif key.startswith("model.model."):
            mapped_key = "model.language_model." + key[len("model.model."):]
        elif key.startswith("model."):
            mapped_key = "model.language_model." + key[len("model."):]
        elif key.startswith("language_model.layers.") or key.startswith("language_model.norm."):
            mapped_key = "model.language_model." + key[len("language_model."):]
        elif key.startswith("language_model."):
            mapped_key = "model.language_model." + key[len("language_model."):]
        elif key.startswith("layers.") or key.startswith("norm."):
            mapped_key = "model.language_model." + key
        else:
            mapped_key = "model.language_model." + key
        mapped_state_dict[mapped_key] = value

    incompatible = model.load_state_dict(mapped_state_dict, strict=False)
    unexpected = list(incompatible.unexpected_keys)
    if unexpected:
        raise RuntimeError(f"Unexpected DeltaKV compressor keys for LLaVA-OneVision: {unexpected[:8]}")
    return incompatible


class LlavaOnevisionDeltaKVModel(LlavaOnevisionModel):
    def __init__(self, config):
        LlavaOnevisionPreTrainedModel.__init__(self, config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)

        text_config = build_llava_text_deltakv_config(config)
        self.deltakv_text_config = text_config
        self.deltakv_text_model_kind = str(getattr(config, "deltakv_text_model_kind", "deltakv")).strip().lower()
        if self.deltakv_text_model_kind == "snapkv":
            self.language_model = Qwen2SnapKVModel(text_config)
        elif self.deltakv_text_model_kind in {"deltakv", "omnikv", "hf_sparse"}:
            self.language_model = Qwen2ModelKVCompress(text_config)
        else:
            raise ValueError(f"Unsupported LLaVA-OneVision text sparse model kind: {self.deltakv_text_model_kind!r}")

        embed_std = 1 / math.sqrt(text_config.hidden_size)
        self.image_newline = nn.Parameter(torch.randn(text_config.hidden_size, dtype=self.dtype) * embed_std)
        self.vocab_size = text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

    def _is_expected_cache(self, past_key_values) -> bool:
        if self.deltakv_text_model_kind == "snapkv":
            return isinstance(past_key_values, SnapKVCache)
        return is_hf_sparse_cache_instance(past_key_values, self.deltakv_text_config)

    def _create_sparse_cache(self):
        if self.deltakv_text_model_kind == "snapkv":
            return SnapKVCache(self.deltakv_text_config)
        return create_hf_sparse_cache(self.deltakv_text_config)

    def _forward_snapkv_language_model(
        self,
        *,
        inputs_embeds: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        position_ids: Optional[torch.LongTensor],
        past_key_values: SnapKVCache,
        use_cache: Optional[bool],
        output_attentions: Optional[bool],
        output_hidden_states: Optional[bool],
        cache_position: Optional[torch.LongTensor],
        created_cache: bool,
        kwargs: dict,
    ):
        q_len = int(inputs_embeds.shape[1])
        window = int(self.deltakv_text_config.snapkv_window_size)
        chunk_size = max(1, int(self.deltakv_text_config.chunk_prefill_size))
        should_chunk_prefill = bool(use_cache) and created_cache and q_len > window
        if not should_chunk_prefill:
            return self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                **kwargs,
            )

        past_key_values.num_prompt_tokens = q_len
        embed_chunks = list(inputs_embeds[:, :-window].split(chunk_size, dim=1)) + [inputs_embeds[:, -window:]]
        if position_ids is None:
            position_chunks = [None] * len(embed_chunks)
        else:
            position_chunks = list(position_ids[:, :-window].split(chunk_size, dim=1)) + [position_ids[:, -window:]]
        if cache_position is None:
            cache_position_chunks = [None] * len(embed_chunks)
        else:
            cache_position_chunks = list(cache_position[:-window].split(chunk_size, dim=0)) + [cache_position[-window:]]

        outputs = None
        for chunk_embeds, chunk_position_ids, chunk_cache_position in zip(
            embed_chunks, position_chunks, cache_position_chunks
        ):
            outputs = self.language_model(
                attention_mask=None,
                position_ids=chunk_position_ids,
                past_key_values=past_key_values,
                inputs_embeds=chunk_embeds,
                use_cache=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=chunk_cache_position,
                **kwargs,
            )
            past_key_values = outputs.past_key_values
        return outputs

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_sizes_videos: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        vision_aspect_ratio: Optional[str] = None,
        batch_num_images: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[tuple, LlavaOnevisionModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        vision_aspect_ratio = vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if (pixel_values is not None or pixel_values_videos is not None) and inputs_embeds is not None:
            raise ValueError(
                "You cannot specify both `pixel_values`/`pixel_values_videos` and `inputs_embeds` at the same time."
            )

        visual_token_mask = None
        if input_ids is not None:
            visual_token_mask = input_ids == self.config.image_token_id

        if inputs_embeds is None:
            inputs_embeds = self.get_input_embeddings()(input_ids)

        image_features = None
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values,
                image_sizes,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
                batch_num_images=batch_num_images,
            )
            image_features = torch.cat(image_features, dim=0)

            special_image_mask = (input_ids == self.config.image_token_id).unsqueeze(-1)
            special_image_mask = special_image_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and inputs_embeds[special_image_mask].numel() != image_features.numel():
                n_image_tokens = (input_ids == self.config.image_token_id).sum()
                n_image_features = image_features.shape[0]
                raise ValueError(
                    f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                )
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        video_features = None
        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_newline = (
                self.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1).to(video_features.device)
            )
            video_features = torch.cat((video_features, image_newline), dim=1)
            video_features = video_features.flatten(0, 1)

            special_video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1)
            special_video_mask = special_video_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and inputs_embeds[special_video_mask].numel() != video_features.numel():
                n_video_tokens = (input_ids == self.config.video_token_id).sum()
                n_video_features = video_features.shape[0]
                raise ValueError(
                    f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                )
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)
            if input_ids is not None:
                visual_token_mask = visual_token_mask | (input_ids == self.config.video_token_id)

        created_cache = False
        if use_cache and not self._is_expected_cache(past_key_values):
            past_key_values = self._create_sparse_cache()
            created_cache = True

        if self.deltakv_text_model_kind == "snapkv":
            outputs = self._forward_snapkv_language_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                created_cache=created_cache,
                kwargs=kwargs,
            )
        else:
            outputs = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=True,
                cache_position=cache_position,
                deltakv_visual_token_mask=visual_token_mask,
                **kwargs,
            )

        return LlavaOnevisionModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
            video_hidden_states=video_features,
        )


class LlavaOnevisionDeltaKVForConditionalGeneration(LlavaOnevisionForConditionalGeneration):
    def __init__(self, config):
        LlavaOnevisionPreTrainedModel.__init__(self, config)
        self.model = LlavaOnevisionDeltaKVModel(config)
        self.lm_head = nn.Linear(config.text_config.hidden_size, config.text_config.vocab_size, bias=False)
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        image_sizes: Optional[torch.LongTensor] = None,
        pixel_values_videos: torch.FloatTensor = None,
        image_sizes_videos: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[list[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[Union[int, list[int]]] = None,
        vision_feature_select_strategy: Optional[str] = None,
        vision_aspect_ratio: Optional[str] = None,
        batch_num_images: Optional[torch.LongTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[tuple, LlavaOnevisionCausalLMOutputWithPast]:
        return super().forward(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_sizes=image_sizes,
            pixel_values_videos=pixel_values_videos,
            image_sizes_videos=image_sizes_videos,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            vision_feature_layer=vision_feature_layer,
            vision_feature_select_strategy=vision_feature_select_strategy,
            vision_aspect_ratio=vision_aspect_ratio,
            batch_num_images=batch_num_images,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            logits_to_keep=logits_to_keep,
            **kwargs,
        )
