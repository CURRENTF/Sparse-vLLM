from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REQUIRED_METHODS = {
    "vanilla",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "omnikv",
    "quest",
    "rkv",
    "skipkv",
    "deltakv",
    "deltakv-less-memory",
    "deltakv-less-memory-cudagraph",
}

REQUIRED_MODELS = {
    "qwen25_7b",
    "qwen25_32b",
    "qwen3_4b",
    "llama31_8b",
}

REQUIRED_ARTIFACTS = [
    "resolved_manifest.json",
    "raw_outputs.jsonl",
    "parsed_outputs.jsonl",
    "sample_results.jsonl",
    "metrics.json",
    "logits_alignment.json",
    "perf.jsonl",
    "memory.json",
    "stress.json",
    "stress_v2.json",
    "scbench.json",
    "grade_summary.json",
]


class ManifestError(ValueError):
    pass


def load_manifest(path: str | Path | None = None) -> dict[str, Any]:
    manifest_path = Path(path) if path else Path(__file__).with_name("manifest.json")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_manifest(manifest)
    return manifest


def validate_manifest(manifest: dict[str, Any]) -> None:
    if not isinstance(manifest, dict):
        raise ManifestError("manifest must be a JSON object.")
    for key in ("models", "methods", "quality", "logits", "performance", "stress", "stress_v2", "scbench", "outputs"):
        if key not in manifest:
            raise ManifestError(f"manifest is missing required key: {key}")

    models = manifest["models"]
    methods = manifest["methods"]
    if not isinstance(models, dict) or not isinstance(methods, dict):
        raise ManifestError("manifest models and methods must be JSON objects.")

    missing_models = sorted(REQUIRED_MODELS - set(models))
    missing_methods = sorted(REQUIRED_METHODS - set(methods))
    if missing_models:
        raise ManifestError(f"manifest is missing required models: {missing_models}")
    if missing_methods:
        raise ManifestError(f"manifest is missing required methods: {missing_methods}")

    quality = manifest["quality"]
    if not isinstance(quality, dict):
        raise ManifestError("manifest quality must be a JSON object.")
    minimum_vanilla_score = quality.get("minimum_vanilla_score")
    if not isinstance(minimum_vanilla_score, (int, float)) or minimum_vanilla_score <= 0:
        raise ManifestError("quality minimum_vanilla_score must be a positive number.")

    for model_id, model in models.items():
        if "model_path_env" not in model:
            raise ManifestError(f"model {model_id!r} is missing model_path_env.")
        if "tokenizer_path_env" not in model:
            raise ManifestError(f"model {model_id!r} is missing tokenizer_path_env.")
        compressor_env = model.get("compressor_path_env")
        if compressor_env is not None and not isinstance(compressor_env, str):
            raise ManifestError(f"model {model_id!r} compressor_path_env must be a string.")
        mixed_attention = model.get("mixed_attention", False)
        if not isinstance(mixed_attention, bool):
            raise ManifestError(f"model {model_id!r} mixed_attention must be a boolean.")

    for method_id, method in methods.items():
        if "sparse_method" not in method:
            raise ManifestError(f"method {method_id!r} is missing sparse_method.")
        if "config" not in method or not isinstance(method["config"], dict):
            raise ManifestError(f"method {method_id!r} must define config object.")
        model_configs = method.get("model_configs")
        if model_configs is not None:
            if not isinstance(model_configs, dict):
                raise ManifestError(f"method {method_id!r} model_configs must be a JSON object.")
            unknown_model_configs = sorted(set(model_configs) - set(models))
            if unknown_model_configs:
                raise ManifestError(
                    f"method {method_id!r} model_configs references unknown models: {unknown_model_configs}"
                )
            for model_id, override in model_configs.items():
                if not isinstance(override, dict):
                    raise ManifestError(
                        f"method {method_id!r} model_configs[{model_id!r}] must be a JSON object."
                    )
        for bool_key in ("requires_compressor", "hf_logits_reference"):
            if bool_key not in method or not isinstance(method[bool_key], bool):
                raise ManifestError(f"method {method_id!r} must define boolean {bool_key}.")
        compressor_env = method.get("compressor_path_env")
        if compressor_env is not None and not isinstance(compressor_env, str):
            raise ManifestError(f"method {method_id!r} compressor_path_env must be a string.")
        if method["requires_compressor"] and "compressor_path_env" not in method:
            model_specific = [model_id for model_id, model in models.items() if model.get("compressor_path_env")]
            if not model_specific:
                raise ManifestError(
                    f"method {method_id!r} requires compressor but no model or method defines compressor_path_env."
                )

    outputs = manifest["outputs"]
    missing_artifacts = sorted(set(REQUIRED_ARTIFACTS) - set(outputs))
    if missing_artifacts:
        raise ManifestError(f"manifest outputs missing required artifacts: {missing_artifacts}")

    scbench = manifest["scbench"]
    if not isinstance(scbench, dict):
        raise ManifestError("manifest scbench must be a JSON object.")
    if scbench.get("model") not in models:
        raise ManifestError(f"scbench model must reference a known model, got {scbench.get('model')!r}.")
    scbench_methods = scbench.get("methods")
    if not isinstance(scbench_methods, list) or not scbench_methods:
        raise ManifestError("scbench methods must be a non-empty list.")
    unknown_scbench_methods = sorted(set(scbench_methods) - set(methods))
    if unknown_scbench_methods:
        raise ManifestError(f"scbench methods reference unknown methods: {unknown_scbench_methods}")
    scbench_tasks = scbench.get("tasks")
    if not isinstance(scbench_tasks, list) or not scbench_tasks:
        raise ManifestError("scbench tasks must be a non-empty list.")
    for int_key in ("num_eval_examples", "max_turns", "max_seq_length", "batch_size"):
        value = scbench.get(int_key)
        if not isinstance(value, int) or value <= 0:
            raise ManifestError(f"scbench {int_key} must be a positive integer.")


def select_entries(manifest: dict[str, Any], models: list[str] | None, methods: list[str] | None):
    model_ids = models or list(manifest["models"])
    method_ids = methods or list(manifest["methods"])
    unknown_models = sorted(set(model_ids) - set(manifest["models"]))
    unknown_methods = sorted(set(method_ids) - set(manifest["methods"]))
    if unknown_models:
        raise ManifestError(f"Unknown model ids: {unknown_models}")
    if unknown_methods:
        raise ManifestError(f"Unknown method ids: {unknown_methods}")
    return model_ids, method_ids


def resolve_manifest_paths(manifest: dict[str, Any]) -> dict[str, Any]:
    resolved = json.loads(json.dumps(manifest))
    for model in resolved["models"].values():
        model["model_path"] = os.getenv(model["model_path_env"])
        tokenizer_env = model["tokenizer_path_env"]
        model["tokenizer_path"] = os.getenv(tokenizer_env) or model["model_path"]
        compressor_env = model.get("compressor_path_env")
        model["compressor_path"] = os.getenv(compressor_env) if compressor_env else None
    for method in resolved["methods"].values():
        env_key = method.get("compressor_path_env")
        method["compressor_path"] = os.getenv(env_key) if env_key else None
    return resolved


def compressor_path_for(model: dict[str, Any], method: dict[str, Any]) -> str | None:
    if not method.get("requires_compressor"):
        return None
    if model.get("compressor_path_env"):
        return model.get("compressor_path")
    return model.get("compressor_path") or method.get("compressor_path")


def compressor_env_for(model: dict[str, Any], method: dict[str, Any]) -> str:
    return model.get("compressor_path_env") or method.get("compressor_path_env") or "compressor_path_env"


def missing_runtime_inputs(resolved: dict[str, Any], model_id: str, method_id: str) -> list[str]:
    missing: list[str] = []
    model = resolved["models"][model_id]
    method = resolved["methods"][method_id]
    if not model.get("model_path"):
        missing.append(model["model_path_env"])
    elif not Path(model["model_path"]).exists():
        missing.append(f"{model['model_path_env']}={model['model_path']}")
    tokenizer_path = model.get("tokenizer_path")
    if tokenizer_path and not Path(tokenizer_path).exists():
        missing.append(f"{model['tokenizer_path_env']}={tokenizer_path}")
    if method.get("requires_compressor"):
        compressor_path = compressor_path_for(model, method)
        compressor_env = compressor_env_for(model, method)
        if not compressor_path:
            missing.append(compressor_env)
        elif not Path(compressor_path).exists():
            missing.append(f"{compressor_env}={compressor_path}")
    return missing
