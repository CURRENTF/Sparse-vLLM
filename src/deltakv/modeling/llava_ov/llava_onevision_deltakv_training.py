from __future__ import annotations

from typing import Optional, Union

import torch
from torch import nn
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

from deltakv.modeling.llava_ov.llava_onevision_deltakv import build_llava_text_deltakv_config
from deltakv.modeling.qwen2_training import Qwen2ModelKVClusterCompress


class LlavaOnevisionDeltaKVTrainingModel(LlavaOnevisionModel):
    def __init__(self, config):
        LlavaOnevisionPreTrainedModel.__init__(self, config)
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)

        text_config = build_llava_text_deltakv_config(config)
        self.deltakv_text_config = text_config
        self.language_model = Qwen2ModelKVClusterCompress(text_config)

        embed_std = 1 / (text_config.hidden_size ** 0.5)
        self.image_newline = nn.Parameter(torch.randn(text_config.hidden_size, dtype=self.dtype) * embed_std)
        self.vocab_size = text_config.vocab_size
        self.pad_token_id = self.config.pad_token_id if self.config.pad_token_id is not None else -1
        self.post_init()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.language_model.set_input_embeddings(value)

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
        kwargs.pop("logits_to_keep", None)
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
            raise ValueError("You cannot specify visual tensors and inputs_embeds at the same time.")

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
                raise ValueError(f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}")
            image_features = image_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        video_features = None
        if pixel_values_videos is not None:
            video_features = self.get_video_features(
                pixel_values_videos,
                vision_feature_layer=vision_feature_layer,
                vision_feature_select_strategy=vision_feature_select_strategy,
            )
            image_newline = self.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1).to(video_features.device)
            video_features = torch.cat((video_features, image_newline), dim=1).flatten(0, 1)
            special_video_mask = (input_ids == self.config.video_token_id).unsqueeze(-1)
            special_video_mask = special_video_mask.expand_as(inputs_embeds).to(inputs_embeds.device)
            if not is_torchdynamo_compiling() and inputs_embeds[special_video_mask].numel() != video_features.numel():
                n_video_tokens = (input_ids == self.config.video_token_id).sum()
                n_video_features = video_features.shape[0]
                raise ValueError(f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}")
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

        run_mode = getattr(self.language_model, "_deltakv_run_mode", None)
        if run_mode is None:
            raise RuntimeError("LLaVA compressor training requires a DeltaKV training language model with run_mode.")
        run_mode["value"] = "raw"
        with torch.no_grad():
            self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )
        run_mode["value"] = "comp"
        try:
            outputs, mse_loss = self.language_model(
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )
        finally:
            run_mode["value"] = "comp"
        self._last_mse_loss = mse_loss.detach()

        return LlavaOnevisionModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_features,
            video_hidden_states=video_features,
        )


class LlavaOnevisionDeltaKVForCompressorTraining(LlavaOnevisionForConditionalGeneration):
    def __init__(self, config):
        LlavaOnevisionPreTrainedModel.__init__(self, config)
        self.model = LlavaOnevisionDeltaKVTrainingModel(config)
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
        outputs = super().forward(
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
        mse_loss = getattr(self.model, "_last_mse_loss", None)
        if mse_loss is None:
            raise RuntimeError("LLaVA compressor training did not produce mse_loss.")
        self._last_ntp_loss = outputs.loss.detach() if outputs.loss is not None else None
        self._last_mse_loss = mse_loss.detach()
        total_loss = (outputs.loss + mse_loss) if outputs.loss is not None else mse_loss
        return LlavaOnevisionCausalLMOutputWithPast(
            loss=total_loss,
            logits=outputs.logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=outputs.image_hidden_states,
            video_hidden_states=outputs.video_hidden_states,
        )
