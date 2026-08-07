from __future__ import annotations

import torch

SUPPORTED_PRUNING_METHODS = {
    "divprune",
    "divprune_official",
    "fastv",
    "fastvid_official_repo",
    "pact_official_repo",
    "visionzip",
}


def batch_to_device(inputs, device, dtype):
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            if value.is_floating_point():
                inputs[key] = value.to(device=device, dtype=dtype)
            else:
                inputs[key] = value.to(device=device)
    return inputs


def ensure_left_padding(processor) -> None:
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        return
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token


def load_vanilla_model(args, dtype, device):
    try:
        from transformers import LlavaOnevisionForConditionalGeneration
    except ImportError as exc:
        raise RuntimeError(
            "LLaVA-OneVision evaluation requires a Transformers build with "
            "LlavaOnevisionForConditionalGeneration."
        ) from exc
    return LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=str(device),
        attn_implementation=args.attn_implementation,
        trust_remote_code=True,
    ).eval()


def iter_requested_methods(methods: str, *, allow_fastvid: bool = True):
    supported = {"vanilla", *SUPPORTED_PRUNING_METHODS}
    for raw_method in [part.strip() for part in methods.split(",") if part.strip()]:
        method = raw_method.lower()
        if method == "vanilla":
            yield raw_method, "vanilla"
        elif method in SUPPORTED_PRUNING_METHODS:
            if method == "fastvid_official_repo" and not allow_fastvid:
                raise ValueError("LLaVA-OV fastvid_official_repo is video-only; use divprune or fastv for image QA.")
            yield raw_method, method
        elif method in {"fastvid", "fastvid_official"}:
            raise ValueError(
                "The local LLaVA-OV FastVID HF ports were removed. Use method='fastvid_official_repo' "
                "to run the FastVID repository implementation."
            )
        else:
            raise ValueError(f"LLaVA-OV adapter supports {sorted(supported)}. Unsupported method={raw_method!r}.")


def load_model_for_method(method_kind: str, args, dtype, device):
    if method_kind == "vanilla":
        return load_vanilla_model(args, dtype, device), None, "vanilla"
    if method_kind in {"divprune", "divprune_official"}:
        model = load_vanilla_model(args, dtype, device)
        from benchmark.multimodal.model_adapters.llava_onevision_pruning import (
            LlavaOneVisionPruningConfig,
            apply_llava_onevision_prefill_pruning,
        )

        policy = apply_llava_onevision_prefill_pruning(
            model,
            LlavaOneVisionPruningConfig(method=method_kind, keep_ratio=float(args.visual_keep_ratio)),
        )
        return model, policy, method_kind
    if method_kind == "visionzip":
        model = load_vanilla_model(args, dtype, device)
        from benchmark.multimodal.model_adapters.llava_onevision_pruning import (
            LlavaOneVisionPruningConfig,
            apply_llava_onevision_visionzip,
        )

        policy = apply_llava_onevision_visionzip(
            model,
            LlavaOneVisionPruningConfig(method=method_kind, keep_ratio=float(args.visual_keep_ratio)),
        )
        return model, policy, method_kind
    if method_kind == "fastvid_official_repo":
        from benchmark.multimodal.model_adapters.fastvid_official_repo import load_fastvid_official_repo_model

        model, policy = load_fastvid_official_repo_model(args, device)
        return model, policy, policy["method"]
    if method_kind == "pact_official_repo":
        from benchmark.multimodal.model_adapters.pact_official_repo import load_pact_official_repo_model

        model, policy = load_pact_official_repo_model(args, device)
        return model, policy, policy["method"]
    if method_kind == "fastv":
        model = load_vanilla_model(args, dtype, device)
        from benchmark.multimodal.model_adapters.llava_onevision_pruning import (
            LlavaOneVisionPruningConfig,
            apply_llava_onevision_fastv,
        )

        policy = apply_llava_onevision_fastv(
            model,
            LlavaOneVisionPruningConfig(method=method_kind, keep_ratio=float(args.visual_keep_ratio)),
        )
        return model, policy, method_kind
    raise AssertionError(f"Unhandled LLaVA-OV method kind: {method_kind}")


__all__ = [
    "batch_to_device",
    "ensure_left_padding",
    "iter_requested_methods",
    "load_model_for_method",
    "load_vanilla_model",
]
