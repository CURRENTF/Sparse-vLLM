from __future__ import annotations

PREFILL_POLICY_ALL_CHUNKED = "all_chunked"
PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH = "long_bs1full_short_batch"
PREFILL_POLICY_AUTO = "auto"

SUPPORTED_PREFILL_POLICIES = {
    PREFILL_POLICY_ALL_CHUNKED,
    PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
}

METHOD_ALIASES = {
    None: "",
    "": "",
    "vanilla": "",
    "attention-sink": "streamingllm",
    "attention_sink": "streamingllm",
}

CANONICAL_SPARSE_METHODS = {
    "",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "omnikv",
    "quest",
    "deltakv",
    "deltakv-triton",
    "deltakv-triton-v2",
    "deltakv-triton-v3",
    "deltakv-triton-v4",
    "deltakv-delta-quant",
    "deltakv_delta_quant",
    "deltakv-standalone",
    "deltakv-snapkv",
}

SUPPORTED_SPARSE_METHODS = set(CANONICAL_SPARSE_METHODS)

DECODE_CUDA_GRAPH_SUPPORTED_METHODS = {
    method for method in CANONICAL_SPARSE_METHODS if not method.startswith("deltakv")
}

_DEFAULT_PREFILL_POLICY_BY_METHOD = {
    "": PREFILL_POLICY_ALL_CHUNKED,
    "streamingllm": PREFILL_POLICY_ALL_CHUNKED,
    "snapkv": PREFILL_POLICY_ALL_CHUNKED,
    "pyramidkv": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "omnikv": PREFILL_POLICY_ALL_CHUNKED,
    "quest": PREFILL_POLICY_ALL_CHUNKED,
    "deltakv": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv-triton": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv-triton-v2": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv-triton-v3": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv-triton-v4": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv-delta-quant": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv_delta_quant": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv-standalone": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv-snapkv": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
}

PREFILL_POLICY_BY_METHOD = {
    **_DEFAULT_PREFILL_POLICY_BY_METHOD,
    "vanilla": PREFILL_POLICY_ALL_CHUNKED,
    "attention-sink": PREFILL_POLICY_ALL_CHUNKED,
    "attention_sink": PREFILL_POLICY_ALL_CHUNKED,
}


def normalize_sparse_method(method: str | None) -> str:
    if method is None:
        return ""
    normalized = str(method).strip().lower()
    return METHOD_ALIASES.get(normalized, normalized)


def is_deltakv_method(method: str | None) -> bool:
    return normalize_sparse_method(method).startswith("deltakv")


def is_decode_cuda_graph_supported(method: str | None) -> bool:
    return normalize_sparse_method(method) in DECODE_CUDA_GRAPH_SUPPORTED_METHODS


def get_default_prefill_schedule_policy(method: str | None) -> str:
    normalized = normalize_sparse_method(method)
    if normalized not in _DEFAULT_PREFILL_POLICY_BY_METHOD:
        supported = ", ".join(repr(name) for name in sorted(CANONICAL_SPARSE_METHODS) if name)
        raise ValueError(
            f"Unsupported vllm_sparse_method={method!r}. Supported methods: '', {supported}."
        )
    return _DEFAULT_PREFILL_POLICY_BY_METHOD[normalized]


def resolve_prefill_schedule_policy(method: str | None, policy: str | None) -> str:
    default_policy = get_default_prefill_schedule_policy(method)
    if policy is None:
        return default_policy

    requested = str(policy).strip().lower()
    if requested in {"", PREFILL_POLICY_AUTO}:
        return default_policy
    if requested not in SUPPORTED_PREFILL_POLICIES:
        supported = ", ".join(repr(name) for name in sorted(SUPPORTED_PREFILL_POLICIES))
        raise ValueError(
            f"Unsupported prefill_schedule_policy={policy!r}. Supported policies: {supported}, "
            f"or {PREFILL_POLICY_AUTO!r}."
        )
    if requested != default_policy:
        raise ValueError(
            "prefill_schedule_policy must match the registry default for reproducibility. "
            f"method={normalize_sparse_method(method)!r} requested={requested!r} default={default_policy!r}."
        )
    return requested
