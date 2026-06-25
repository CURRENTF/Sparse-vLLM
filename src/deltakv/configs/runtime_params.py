from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class NormalizedRuntimeParams:
    """Normalized runtime parameters plus optional top-level routing fields."""

    infer_config: dict[str, Any]
    hf_model_cls: str | None = None
    hf_deltakv_checkpoint_path: str | None = None
    warnings: tuple[str, ...] = ()


_COMMON_ALIASES: dict[str, str] = {
    # Accuracy-affecting sparse token budgets.
    "sink_keep_tokens": "num_sink_tokens",
    "recent_keep_tokens": "num_recent_tokens",
    # Layer routing.
    "full_attention_layers": "full_attn_layers",
    # DeltaKV compression / clustering semantics.
    "deltakv_center_ratio": "cluster_ratio",
    "deltakv_latent_dim": "kv_compressed_size",
    "deltakv_latent_quant_bits": "kv_quant_bits",
    "deltakv_latent_quant_group_size": "kv_quant_group_size",
}

_BACKEND_ALIASES: dict[str, dict[str, str]] = {
    "hf": {
        "decode_keep_tokens": "num_top_tokens",
        "prefill_keep_tokens": "num_top_tokens_in_prefill",
        "hf_prefill_chunk_size": "chunk_prefill_size",
    },
    "sparsevllm": {
        "engine_prefill_chunk_size": "chunk_prefill_size",
        "deltakv_neighbor_count": "deltakv_k_neighbors",
        "observation_layers": "obs_layer_ids",
    },
}

_LEGACY_RUNTIME_KEYS: dict[str, str] = {
    # Routing/checkpoint names.
    "model_cls": "sparse_method",
    "vllm_sparse_method": "sparse_method",
    "compressor_path": "deltakv_checkpoint_path",
    "deltakv_path": "deltakv_checkpoint_path",
    # Token budgets.
    "num_top_tokens": "decode_keep_tokens",
    "num_top_tokens_in_prefill": "prefill_keep_tokens",
    "num_sink_tokens": "sink_keep_tokens",
    "num_recent_tokens": "recent_keep_tokens",
    "tail_token_size": "recent_keep_tokens",
    # Layer routing.
    "full_attn_layers": "full_attention_layers",
    "obs_layer_ids": "observation_layers",
    # DeltaKV naming.
    "k_neighbors": "deltakv_neighbor_count",
    "deltakv_k_neighbors": "deltakv_neighbor_count",
    "seq_chunk_size": "removed; use deltakv_neighbor_count for cluster reference top-k",
    "compressor_token_group_size": "removed; use deltakv_neighbor_count for cluster reference top-k",
    "ref_mode": "removed; cluster_e2e_big always uses cluster-derived references",
    "cluster_ratio": "deltakv_center_ratio",
    "kv_compressed_size": "deltakv_latent_dim",
    "kv_quant_bits": "deltakv_latent_quant_bits",
    "kv_quant_group_size": "deltakv_latent_quant_group_size",
    # Prefill chunking must be backend-specific.
    "chunk_prefill_size": "hf_prefill_chunk_size or engine_prefill_chunk_size",
    "model_prefill_chunk_size": "hf_prefill_chunk_size",
    "sparsevllm_prefill_chunk_size": "engine_prefill_chunk_size",
    # LLaVA visual path.
    "deltakv_visual_compress_only": "visual_token_prune_only",
    "deltakv_visual_keep_ratio": "visual_token_keep_ratio",
}

_SPARSE_METHOD_TO_HF_MODEL_CLS: dict[str, str] = {
    "": "auto",
    "vanilla": "auto",
    "deltakv": "deltakv",
    "deltakv-less-memory": "deltakv",
    "deltakv-less-memory-cudagraph": "deltakv",
    "delta_compressed_quant_kivi_full_fp8_ref": "delta_compressed_quant_kivi_full_fp8_ref",
    "hf_kivi": "hf_kivi",
    "kivi_hf": "hf_kivi",
    "snapkv": "snapkv",
    "pyramidkv": "pyramidkv",
    "omnikv": "omnikv",
    "quest": "quest",
    "rkv": "rkv",
    "r-kv": "rkv",
    "r_kv": "rkv",
    "skipkv": "skipkv",
    "skip-kv": "skipkv",
    "skip_kv": "skipkv",
    "streamingllm": "streamingllm",
    "attention-sink": "streamingllm",
    "attention_sink": "streamingllm",
}

_SPARSEVLLM_METHOD_ALIASES: dict[str, str] = {
    "vanilla": "",
    "r-kv": "rkv",
    "r_kv": "rkv",
    "skip-kv": "skipkv",
    "skip_kv": "skipkv",
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


def _reject_legacy_runtime_keys(params: dict[str, Any]):
    found = sorted(key for key in params if key in _LEGACY_RUNTIME_KEYS)
    if not found:
        return
    details = ", ".join(
        f"`{key}` -> `{_LEGACY_RUNTIME_KEYS[key]}`" for key in found
    )
    raise ValueError(
        "Legacy runtime parameter names are no longer accepted. "
        f"Use the new semantic names instead: {details}."
    )


def _validate_sparsevllm_token_budgets(params: dict[str, Any]):
    for key in ("decode_keep_tokens",):
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
) -> NormalizedRuntimeParams:
    """Normalize user-facing runtime params to backend-native legacy fields.

    The canonical aliases are intentionally explicit:

    - `decode_keep_tokens` stays native for Sparse-vLLM and maps to HF `num_top_tokens`
    - `engine_prefill_chunk_size` -> Sparse-vLLM `chunk_prefill_size`
    - `hf_prefill_chunk_size` -> HF DeltaKV `chunk_prefill_size`
    - `deltakv_checkpoint_path` -> Sparse-vLLM `deltakv_path` or HF compressor path

    Legacy runtime keys are rejected at the API boundary. This function still
    maps the new semantic names to backend-native internal fields where needed.
    """

    backend = _canonical_backend(backend)
    normalized = dict(params or {})
    warnings: list[str] = []

    _reject_legacy_runtime_keys(normalized)

    hf_model_cls: str | None = None
    hf_deltakv_checkpoint_path: str | None = None

    sparse_method = normalized.pop("sparse_method", None)
    if sparse_method is not None:
        sparse_method = str(sparse_method)
        if backend == "sparsevllm":
            sparsevllm_method = _SPARSEVLLM_METHOD_ALIASES.get(sparse_method, sparse_method)
            normalized["vllm_sparse_method"] = sparsevllm_method
            warnings.append("`sparse_method` was normalized to `vllm_sparse_method`.")
        elif backend == "hf":
            mapped_model_cls = _SPARSE_METHOD_TO_HF_MODEL_CLS.get(sparse_method, sparse_method)
            if hf_model_cls is not None and hf_model_cls != mapped_model_cls:
                raise ValueError(
                    f"Conflicting method selectors: hf_model_cls={hf_model_cls!r}, "
                    f"sparse_method={sparse_method!r} -> {mapped_model_cls!r}."
                )
            hf_model_cls = mapped_model_cls
            warnings.append("`sparse_method` was normalized to HF backend class.")
        else:
            normalized["sparse_method"] = sparse_method

    checkpoint_path = normalized.pop("deltakv_checkpoint_path", None)
    if backend == "sparsevllm":
        if checkpoint_path is not None:
            normalized["deltakv_path"] = checkpoint_path
            warnings.append("`deltakv_checkpoint_path` was normalized to `deltakv_path`.")
        hf_deltakv_checkpoint_path = None
    elif backend == "hf":
        if checkpoint_path is not None:
            hf_deltakv_checkpoint_path = str(checkpoint_path)
            warnings.append("`deltakv_checkpoint_path` was normalized for the HF DeltaKV loader.")
    elif checkpoint_path is not None:
        normalized["deltakv_checkpoint_path"] = checkpoint_path

    _normalize_aliases(normalized, backend, warnings)
    if backend == "sparsevllm":
        _validate_sparsevllm_token_budgets(normalized)

    return NormalizedRuntimeParams(
        infer_config=normalized,
        hf_model_cls=hf_model_cls,
        hf_deltakv_checkpoint_path=hf_deltakv_checkpoint_path,
        warnings=tuple(warnings),
    )
