from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedRuntimeParams:
    """Normalized runtime parameters plus optional top-level routing fields."""

    infer_config: dict[str, Any]
    model_cls: str | None = None
    compressor_path: str | None = None
    warnings: tuple[str, ...] = ()


_COMMON_ALIASES: dict[str, str] = {
    # Accuracy-affecting sparse token budgets.
    "decode_keep_tokens": "num_top_tokens",
    "prefill_keep_tokens": "num_top_tokens_in_prefill",
    "sink_keep_tokens": "num_sink_tokens",
    "recent_keep_tokens": "num_recent_tokens",
    # Layer routing.
    "full_attention_layers": "full_attn_layers",
    # DeltaKV compression / clustering semantics.
    "deltakv_center_ratio": "cluster_ratio",
    "deltakv_latent_dim": "kv_compressed_size",
    "deltakv_latent_quant_bits": "kv_quant_bits",
}

_BACKEND_ALIASES: dict[str, dict[str, str]] = {
    "hf": {
        "hf_prefill_chunk_size": "chunk_prefill_size",
        "model_prefill_chunk_size": "chunk_prefill_size",
        "deltakv_neighbor_count": "k_neighbors",
        # Multimodal visual-token pruning aliases. The legacy names used
        # "deltakv" and "compress" even for the no-compressor uniform-prune path.
        "deltakv_visual_compress_only": "visual_token_prune_only",
        "deltakv_visual_keep_ratio": "visual_token_keep_ratio",
    },
    "sparsevllm": {
        "engine_prefill_chunk_size": "chunk_prefill_size",
        "sparsevllm_prefill_chunk_size": "chunk_prefill_size",
        "deltakv_neighbor_count": "deltakv_k_neighbors",
        "observation_layers": "obs_layer_ids",
    },
}

_SPARSE_METHOD_TO_HF_MODEL_CLS: dict[str, str] = {
    "": "auto",
    "vanilla": "auto",
    "deltakv": "deltakv",
    "deltakv-triton": "deltakv",
    "deltakv-triton-v2": "deltakv",
    "deltakv-triton-v3": "deltakv",
    "deltakv-triton-v4": "deltakv",
    "deltakv-triton-v3-offload": "deltakv",
    "deltakv-triton-v3-cuda-offload": "deltakv",
    "deltakv-standalone": "deltakv",
    "deltakv-snapkv": "deltasnapkv",
    "snapkv": "snapkv",
    "pyramidkv": "pyramidkv",
    "omnikv": "omnikv",
    "quest": "quest",
    "streamingllm": "streamingllm",
    "attention-sink": "streamingllm",
    "attention_sink": "streamingllm",
}


def _canonical_backend(backend: str | None) -> str | None:
    if backend is None:
        return None
    backend = str(backend).strip().lower()
    if backend in ("sparse-vllm", "sparse_vllm"):
        return "sparsevllm"
    if backend in ("hf", "sparsevllm"):
        return backend
    raise ValueError(f"Unknown runtime parameter backend: {backend!r}")


def _set_alias(
    params: dict[str, Any],
    *,
    alias: str,
    target: str,
    warnings: list[str],
):
    if alias not in params:
        return

    value = params.pop(alias)
    if target in params:
        if params[target] != value:
            raise ValueError(
                f"Conflicting runtime parameters: `{alias}`={value!r} maps to "
                f"`{target}`, but `{target}`={params[target]!r} was also provided."
            )
        warnings.append(f"`{alias}` duplicates `{target}`; using `{target}`.")
        return

    params[target] = value
    warnings.append(f"`{alias}` was normalized to `{target}`.")


def _normalize_aliases(params: dict[str, Any], backend: str | None, warnings: list[str]):
    aliases = dict(_COMMON_ALIASES)
    if backend is not None:
        aliases.update(_BACKEND_ALIASES.get(backend, {}))

    for alias, target in aliases.items():
        _set_alias(params, alias=alias, target=target, warnings=warnings)


def _validate_sparsevllm_token_budgets(params: dict[str, Any]):
    for key in ("num_top_tokens", "num_top_tokens_in_prefill"):
        value = params.get(key)
        if isinstance(value, float) and value <= 1.0:
            raise ValueError(
                f"Sparse-vLLM `{key}` must be an explicit token count, got ratio-style "
                f"value {value!r}. Convert the ratio using the target context length before "
                "running Sparse-vLLM, or use backend='hf' for ratio semantics."
            )


def normalize_runtime_params(
    params: dict[str, Any] | None,
    *,
    backend: str | None = None,
    model_cls: str | None = None,
    compressor_path: str | None = None,
) -> NormalizedRuntimeParams:
    """Normalize user-facing runtime params to backend-native legacy fields.

    The canonical aliases are intentionally explicit:

    - `decode_keep_tokens` -> `num_top_tokens`
    - `prefill_keep_tokens` -> `num_top_tokens_in_prefill`
    - `engine_prefill_chunk_size` -> Sparse-vLLM `chunk_prefill_size`
    - `hf_prefill_chunk_size` -> HF DeltaKV `chunk_prefill_size`
    - `deltakv_checkpoint_path` -> Sparse-vLLM `deltakv_path` or HF `compressor_path`

    Existing legacy keys continue to work. Supplying both a canonical key and its
    legacy target with different values raises, because that changes accuracy or
    speed semantics silently.
    """

    backend = _canonical_backend(backend)
    normalized = dict(params or {})
    warnings: list[str] = []

    # Accept these top-level fields inside JSON configs because many benchmark
    # launchers already pass a single hyper_param blob.
    json_model_cls = normalized.pop("model_cls", None)
    if json_model_cls is not None:
        if model_cls is not None and str(model_cls) != str(json_model_cls):
            raise ValueError(
                f"Conflicting model_cls values: argument={model_cls!r}, JSON={json_model_cls!r}."
            )
        model_cls = str(json_model_cls)

    json_compressor_path = normalized.pop("compressor_path", None)
    if json_compressor_path is not None:
        if compressor_path is not None and compressor_path != json_compressor_path:
            raise ValueError(
                "Conflicting compressor_path values: "
                f"argument={compressor_path!r}, JSON={json_compressor_path!r}."
            )
        compressor_path = str(json_compressor_path)

    sparse_method = normalized.pop("sparse_method", None)
    if sparse_method is not None:
        sparse_method = str(sparse_method)
        if backend == "sparsevllm":
            sparsevllm_method = "" if sparse_method == "vanilla" else sparse_method
            if "vllm_sparse_method" in normalized and normalized["vllm_sparse_method"] != sparsevllm_method:
                raise ValueError(
                    "Conflicting runtime parameters: `sparse_method` maps to "
                    f"`vllm_sparse_method`, but `vllm_sparse_method`={normalized['vllm_sparse_method']!r}."
                )
            normalized["vllm_sparse_method"] = sparsevllm_method
            warnings.append("`sparse_method` was normalized to `vllm_sparse_method`.")
        elif backend == "hf":
            mapped_model_cls = _SPARSE_METHOD_TO_HF_MODEL_CLS.get(sparse_method, sparse_method)
            if model_cls is not None and model_cls != mapped_model_cls:
                raise ValueError(
                    f"Conflicting method selectors: model_cls={model_cls!r}, "
                    f"sparse_method={sparse_method!r} -> {mapped_model_cls!r}."
                )
            model_cls = mapped_model_cls
            warnings.append("`sparse_method` was normalized to HF `model_cls`.")
        else:
            normalized["sparse_method"] = sparse_method

    checkpoint_path = normalized.pop("deltakv_checkpoint_path", None)
    if backend == "sparsevllm":
        if checkpoint_path is not None:
            if "deltakv_path" in normalized and normalized["deltakv_path"] != checkpoint_path:
                raise ValueError(
                    "Conflicting DeltaKV checkpoint paths: "
                    f"deltakv_checkpoint_path={checkpoint_path!r}, "
                    f"deltakv_path={normalized['deltakv_path']!r}."
                )
            normalized["deltakv_path"] = checkpoint_path
            warnings.append("`deltakv_checkpoint_path` was normalized to `deltakv_path`.")
        elif compressor_path is not None:
            if "deltakv_path" in normalized:
                if normalized["deltakv_path"] != compressor_path:
                    raise ValueError(
                        "Conflicting DeltaKV checkpoint paths: "
                        f"compressor_path={compressor_path!r}, deltakv_path={normalized['deltakv_path']!r}."
                    )
            else:
                normalized["deltakv_path"] = compressor_path
                warnings.append("top-level `compressor_path` was normalized to `deltakv_path`.")
        compressor_path = None
    elif backend == "hf":
        if checkpoint_path is not None:
            if compressor_path is not None and compressor_path != checkpoint_path:
                raise ValueError(
                    "Conflicting DeltaKV checkpoint paths: "
                    f"deltakv_checkpoint_path={checkpoint_path!r}, compressor_path={compressor_path!r}."
                )
            compressor_path = str(checkpoint_path)
            warnings.append("`deltakv_checkpoint_path` was normalized to HF `compressor_path`.")
        elif "deltakv_path" in normalized:
            deltakv_path = str(normalized.pop("deltakv_path"))
            if compressor_path is not None and compressor_path != deltakv_path:
                raise ValueError(
                    "Conflicting DeltaKV checkpoint paths: "
                    f"compressor_path={compressor_path!r}, deltakv_path={deltakv_path!r}."
                )
            compressor_path = deltakv_path
            warnings.append("`deltakv_path` was normalized to HF `compressor_path`.")
    elif checkpoint_path is not None:
        normalized["deltakv_checkpoint_path"] = checkpoint_path

    _normalize_aliases(normalized, backend, warnings)
    if backend == "sparsevllm":
        _validate_sparsevllm_token_budgets(normalized)

    return NormalizedRuntimeParams(
        infer_config=normalized,
        model_cls=model_cls,
        compressor_path=compressor_path,
        warnings=tuple(warnings),
    )
