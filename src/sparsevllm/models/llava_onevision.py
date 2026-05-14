import math
from typing import Any

import torch
from torch import nn
from transformers import AutoModel
from transformers.models.llava_onevision.modeling_llava_onevision import (
    LlavaOnevisionMultiModalProjector,
    get_anyres_image_grid_shape,
    image_size_to_num_patches,
    unpad_image,
)

from sparsevllm.models.qwen2 import Qwen2ForCausalLM


class LlavaOnevisionForCausalLM(nn.Module):
    """Sparse-vLLM LLaVA-OneVision wrapper.

    LLaVA-specific vision/projector work stays here. The text path is the existing
    Qwen2 Sparse-vLLM model, receiving precomputed prompt embeddings only for
    prefill and ordinary token ids during decode.
    """

    packed_modules_mapping = Qwen2ForCausalLM.packed_modules_mapping

    def __init__(self, config) -> None:
        super().__init__()
        if getattr(config, "model_type", None) != "llava_onevision":
            raise ValueError(f"LlavaOnevisionForCausalLM requires model_type='llava_onevision', got {config.model_type!r}")
        if getattr(config.text_config, "model_type", None) != "qwen2":
            raise ValueError(
                "Sparse-vLLM LLaVA-OneVision currently supports Qwen2 text backbones only, "
                f"got {getattr(config.text_config, 'model_type', None)!r}."
            )

        self.config = config
        self.vision_tower = AutoModel.from_config(config.vision_config)
        self.multi_modal_projector = LlavaOnevisionMultiModalProjector(config)
        self.image_newline = nn.Parameter(torch.empty(config.text_config.hidden_size))
        self.language_model = Qwen2ForCausalLM(config.text_config)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.language_model(input_ids, positions, inputs_embeds=inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.language_model.compute_logits(hidden_states)

    def get_image_features(
        self,
        pixel_values: torch.Tensor,
        image_sizes: torch.Tensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
        vision_aspect_ratio: str | None = None,
        batch_num_images: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        vision_aspect_ratio = vision_aspect_ratio if vision_aspect_ratio is not None else self.config.vision_aspect_ratio

        if batch_num_images is None:
            need_patching = [True] * len(image_sizes)
        else:
            need_patching = [int(n) == 1 for n in batch_num_images for _ in range(int(n))]
        image_num_patches = [
            image_size_to_num_patches(
                image_size=imsize,
                grid_pinpoints=self.config.image_grid_pinpoints,
                patch_size=self.config.vision_config.image_size,
            )
            if should_patch
            else 1
            for imsize, should_patch in zip(image_sizes, need_patching)
        ]
        if pixel_values.dim() == 5:
            pixel_values = torch.cat(
                [pix_val[:num_patch] for pix_val, num_patch in zip(pixel_values, image_num_patches)],
                dim=0,
            )
        elif pixel_values.dim() != 4:
            raise ValueError(f"pixel_values must have 4 or 5 dimensions, got shape={tuple(pixel_values.shape)}")

        image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        if isinstance(vision_feature_layer, int):
            selected = image_outputs.hidden_states[vision_feature_layer]
        else:
            selected = torch.cat([image_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer], dim=-1)

        if vision_feature_select_strategy == "default":
            selected = selected[:, 1:]
        elif vision_feature_select_strategy != "full":
            raise ValueError(f"Unsupported vision_feature_select_strategy={vision_feature_select_strategy!r}")

        image_features = self.multi_modal_projector(selected)
        image_features = torch.split(image_features, image_num_patches, dim=0)
        image_features, _ = self.pack_image_features(
            image_features,
            image_sizes,
            image_newline=self.image_newline,
            vision_aspect_ratio=vision_aspect_ratio,
        )
        return image_features

    def get_video_features(
        self,
        pixel_values: torch.Tensor,
        vision_feature_layer: int | list[int] | None = None,
        vision_feature_select_strategy: str | None = None,
    ) -> torch.Tensor:
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )
        batch_size, frames, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * frames, channels, height, width)
        video_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
        if isinstance(vision_feature_layer, int):
            selected = video_outputs.hidden_states[vision_feature_layer]
        else:
            selected = torch.cat([video_outputs.hidden_states[layer_idx] for layer_idx in vision_feature_layer], dim=-1)

        if vision_feature_select_strategy == "default":
            selected = selected[:, 1:]
        elif vision_feature_select_strategy != "full":
            raise ValueError(f"Unsupported vision_feature_select_strategy={vision_feature_select_strategy!r}")

        video_features = self.multi_modal_projector(selected)
        video_features = self.apply_pooling(video_features)
        video_features = video_features.reshape(batch_size, frames * video_features.shape[1], -1)
        return video_features

    def apply_pooling(self, image_features: torch.Tensor) -> torch.Tensor:
        height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
        batch_frames, _, dim = image_features.shape
        image_features = image_features.view(batch_frames, height, width, -1)
        image_features = image_features.permute(0, 3, 1, 2).contiguous()
        scaled_shape = [math.ceil(image_features.shape[2] / 2), math.ceil(image_features.shape[3] / 2)]
        image_features = nn.functional.interpolate(image_features, size=scaled_shape, mode="bilinear")
        image_features = image_features.permute(0, 2, 3, 1)
        return image_features.view(batch_frames, -1, dim)

    def pack_image_features(
        self,
        image_features: tuple[torch.Tensor, ...],
        image_sizes: torch.Tensor,
        image_newline: torch.Tensor | None = None,
        vision_aspect_ratio: str = "anyres_max_9",
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        new_image_features = []
        feature_lens = []
        for image_idx, image_feature in enumerate(image_features):
            if image_feature.shape[0] > 1:
                base_image_feature = image_feature[0]
                image_feature = image_feature[1:]
                height = width = self.config.vision_config.image_size // self.config.vision_config.patch_size
                if height * width != base_image_feature.shape[0]:
                    raise ValueError("The number of patches is not consistent with the image size.")
                num_patch_height, num_patch_width = get_anyres_image_grid_shape(
                    image_sizes[image_idx],
                    self.config.image_grid_pinpoints,
                    self.config.vision_config.image_size,
                )
                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                image_feature = unpad_image(image_feature, image_sizes[image_idx])
                max_num_patches = int(vision_aspect_ratio.strip("anyres_max_"))
                channels, curr_height, curr_width = image_feature.shape
                ratio = math.sqrt(curr_height * curr_width / (max_num_patches * height**2))
                if ratio > 1.1:
                    image_feature = nn.functional.interpolate(
                        image_feature[None],
                        [int(curr_height // ratio), int(curr_width // ratio)],
                        mode="bilinear",
                    )[0]
                if image_newline is not None:
                    image_feature = torch.cat(
                        (
                            image_feature,
                            image_newline[:, None, None]
                            .expand(*image_feature.shape[:-1], 1)
                            .to(image_feature.device, image_feature.dtype),
                        ),
                        dim=-1,
                    )
                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                image_feature = torch.cat((base_image_feature, image_feature), dim=0)
            else:
                image_feature = image_feature[0]
                if image_newline is not None:
                    image_feature = torch.cat((image_feature, image_newline[None].to(image_feature)), dim=0)
                image_feature = image_feature.flatten(0, 1)
            new_image_features.append(image_feature)
            feature_lens.append(image_feature.size(0))
        feature_lens = torch.tensor(feature_lens, dtype=torch.long, device=image_features[0].device)
        return new_image_features, feature_lens

    @torch.inference_mode()
    def prepare_prompt_embeds(self, model_inputs: dict[str, Any]) -> tuple[list[list[int]], list[torch.Tensor]]:
        input_ids = model_inputs.get("input_ids")
        if input_ids is None:
            raise ValueError("LLaVA-OneVision multimodal generation requires input_ids.")
        attention_mask = model_inputs.get("attention_mask")
        if input_ids.dim() != 2:
            raise ValueError(f"input_ids must be a 2D padded batch, got shape={tuple(input_ids.shape)}")
        if attention_mask is not None and attention_mask.shape != input_ids.shape:
            raise ValueError(
                f"attention_mask shape must match input_ids: {tuple(attention_mask.shape)} != {tuple(input_ids.shape)}"
            )

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        pixel_values = model_inputs.get("pixel_values")
        if pixel_values is not None:
            image_features = self.get_image_features(
                pixel_values,
                model_inputs.get("image_sizes"),
                batch_num_images=model_inputs.get("batch_num_images"),
            )
            image_features = torch.cat(image_features, dim=0).to(inputs_embeds.device, inputs_embeds.dtype)
            special_image_mask = (input_ids == int(self.config.image_token_id)).unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[special_image_mask].numel() != image_features.numel():
                raise ValueError(
                    "Image features and image tokens do not match: "
                    f"tokens={(input_ids == int(self.config.image_token_id)).sum().item()} "
                    f"features={image_features.shape[0]}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(special_image_mask, image_features)

        pixel_values_videos = model_inputs.get("pixel_values_videos")
        if pixel_values_videos is not None:
            video_features = self.get_video_features(pixel_values_videos)
            image_newline = self.image_newline[None, None, :].repeat(video_features.shape[0], 1, 1)
            image_newline = image_newline.to(video_features.device, video_features.dtype)
            video_features = torch.cat((video_features, image_newline), dim=1).flatten(0, 1)
            video_features = video_features.to(inputs_embeds.device, inputs_embeds.dtype)
            special_video_mask = (input_ids == int(self.config.video_token_id)).unsqueeze(-1).expand_as(inputs_embeds)
            if inputs_embeds[special_video_mask].numel() != video_features.numel():
                raise ValueError(
                    "Video features and video tokens do not match: "
                    f"tokens={(input_ids == int(self.config.video_token_id)).sum().item()} "
                    f"features={video_features.shape[0]}"
                )
            inputs_embeds = inputs_embeds.masked_scatter(special_video_mask, video_features)

        token_id_lists: list[list[int]] = []
        prompt_embeds: list[torch.Tensor] = []
        for row_idx in range(input_ids.shape[0]):
            if attention_mask is None:
                valid_mask = torch.ones_like(input_ids[row_idx], dtype=torch.bool)
            else:
                valid_mask = attention_mask[row_idx].to(dtype=torch.bool)
            row_token_ids = input_ids[row_idx, valid_mask].detach().cpu().tolist()
            if not row_token_ids:
                raise ValueError(f"Multimodal request {row_idx} has no valid prompt tokens.")
            token_id_lists.append([int(token_id) for token_id in row_token_ids])
            prompt_embeds.append(inputs_embeds[row_idx, valid_mask].contiguous())

        return token_id_lists, prompt_embeds
