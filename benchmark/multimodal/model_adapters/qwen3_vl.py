from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


SUPPORTED_METHODS = {"vanilla", "divprune", "divprune_official", "fastv", "fastvid"}

def require_qwen3_vl_transformers():
    try:
        from transformers import AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "Qwen3-VL evaluation requires a Transformers build with "
            "Qwen3VLForConditionalGeneration. Install a recent/source Transformers "
            "in the evaluation environment before running --model_family qwen3_vl."
        ) from exc
    return AutoProcessor, Qwen3VLConfig, Qwen3VLForConditionalGeneration


def ensure_left_padding(processor: Any) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def iter_requested_methods(methods: str):
    for raw_method in [part.strip() for part in methods.split(",") if part.strip()]:
        method = raw_method.lower()
        if method in SUPPORTED_METHODS:
            yield raw_method, method
            continue
        raise ValueError(f"Qwen3-VL adapter supports only {sorted(SUPPORTED_METHODS)}. Unsupported method={raw_method!r}.")


@dataclass
class Qwen3VLRuntime:
    model: Any
    processor: Any
    method: str = "vanilla"

    @property
    def supports_batch_generation(self) -> bool:
        return False


def load_processor(model_path: str):
    AutoProcessor, _, _ = require_qwen3_vl_transformers()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    ensure_left_padding(processor)
    return processor


def load_vanilla_model(args: Any, dtype: torch.dtype, device: torch.device):
    _, _, model_cls = require_qwen3_vl_transformers()
    model = model_cls.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()
    return model, None, "vanilla"


def load_model_for_method(method_kind: str, args: Any, dtype: torch.dtype, device: torch.device):
    if method_kind == "vanilla":
        return load_vanilla_model(args, dtype, device)
    if method_kind in {"divprune", "divprune_official", "fastvid"}:
        model, _, _ = load_vanilla_model(args, dtype, device)
        from benchmark.multimodal.model_adapters.qwen3_vl_pruning import (
            Qwen3VLPruningConfig,
            apply_qwen3_vl_prefill_pruning,
        )

        policy = apply_qwen3_vl_prefill_pruning(
            model,
            Qwen3VLPruningConfig(method=method_kind, keep_ratio=float(args.visual_keep_ratio)),
        )
        return model, policy, method_kind
    if method_kind == "fastv":
        model, _, _ = load_vanilla_model(args, dtype, device)
        from benchmark.multimodal.model_adapters.qwen3_vl_pruning import (
            Qwen3VLPruningConfig,
            apply_qwen3_vl_fastv,
        )

        policy = apply_qwen3_vl_fastv(
            model,
            Qwen3VLPruningConfig(method=method_kind, keep_ratio=float(args.visual_keep_ratio)),
        )
        return model, policy, method_kind
    raise AssertionError(f"Unhandled Qwen3-VL method kind: {method_kind}")


def _message_content(kind: str, media: Any, text: str) -> list[dict[str, Any]]:
    if kind == "image":
        media_items = media if isinstance(media, list) else [media]
        content = [{"type": "image", "image": item} for item in media_items]
    elif kind == "video":
        content = [{"type": "video", "video": media}]
    else:
        raise ValueError(f"Unsupported Qwen3-VL media kind: {kind}")
    content.append({"type": "text", "text": text})
    return content


def prepare_inputs(
    processor: Any,
    *,
    text: str,
    media_kind: str,
    media: Any,
    device: torch.device | None,
    dtype: torch.dtype,
) -> tuple[dict[str, Any], int, int]:
    messages = [{"role": "user", "content": _message_content(media_kind, media, text)}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    if device is None:
        input_len = int(inputs["input_ids"].shape[1])
        visual_tokens = infer_visual_token_count(inputs)
        return inputs, input_len, visual_tokens
    if hasattr(inputs, "to"):
        inputs = inputs.to(device)
    else:
        for key, value in list(inputs.items()):
            if torch.is_tensor(value):
                inputs[key] = value.to(device=device, dtype=dtype) if value.is_floating_point() else value.to(device=device)
    input_len = int(inputs["input_ids"].shape[1])
    visual_tokens = infer_visual_token_count(inputs)
    return inputs, input_len, visual_tokens


def infer_visual_token_count(inputs: dict[str, Any]) -> int:
    if "image_grid_thw" in inputs and torch.is_tensor(inputs["image_grid_thw"]):
        grid = inputs["image_grid_thw"]
        return int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
    if "video_grid_thw" in inputs and torch.is_tensor(inputs["video_grid_thw"]):
        grid = inputs["video_grid_thw"]
        return int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
    return 0


def decode_generated(processor: Any, output_ids: torch.Tensor, input_ids: torch.Tensor) -> list[str]:
    generated_ids = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(input_ids, output_ids)]
    return processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
