from __future__ import annotations

from dataclasses import dataclass

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
    "r-kv": "rkv",
    "r_kv": "rkv",
    "skip-kv": "skipkv",
    "skip_kv": "skipkv",
    # DeltaKV now has one public runtime.  The old names stay as aliases so old
    # config files still load, but all code routes through vllm_sparse_method="deltakv".
    "deltakv-less-memory": "deltakv",
    "deltakv_less_memory": "deltakv",
    "deltakv-less-memory-cudagraph": "deltakv",
    "deltakv_less_memory_cudagraph": "deltakv",
}

CANONICAL_SPARSE_METHODS = {
    "",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "omnikv",
    "quest",
    "rkv",
    "skipkv",
    "deltakv",
}

SUPPORTED_SPARSE_METHODS = set(CANONICAL_SPARSE_METHODS)
SUPPORTED_SPARSE_METHOD_ALIASES = {str(k) for k in METHOD_ALIASES if k is not None and str(k)}

PREFIX_CACHE_SUPPORTED_METHODS = {"", "omnikv", "quest"}


@dataclass(frozen=True)
class ModelRuntimeCompatibility:
    parallel_mode: str
    sparse_methods: frozenset[str]
    prefix_cache_methods: frozenset[str]
    requires_eager: bool = True
    decode_cuda_graph_methods: frozenset[str] = frozenset()


QWEN3_MOE_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    parallel_mode="ep_replicated_kv",
    sparse_methods=frozenset(
        {
            "",
            "streamingllm",
            "snapkv",
            "pyramidkv",
            "omnikv",
            "quest",
            "rkv",
        }
    ),
    prefix_cache_methods=frozenset({"", "omnikv", "quest"}),
    requires_eager=False,
    decode_cuda_graph_methods=frozenset({""}),
)

MINIMAX_M2_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    parallel_mode="ep_replicated_kv",
    sparse_methods=frozenset(
        {
            "",
            "streamingllm",
            "snapkv",
            "pyramidkv",
            "omnikv",
            "quest",
            "rkv",
        }
    ),
    prefix_cache_methods=frozenset({"", "omnikv", "quest"}),
    requires_eager=False,
    decode_cuda_graph_methods=frozenset({""}),
)

MODEL_RUNTIME_COMPATIBILITY = {
    "qwen3_moe": QWEN3_MOE_EP_COMPATIBILITY,
    "minimax_m2": MINIMAX_M2_EP_COMPATIBILITY,
}

# All shipped cache managers now expose a graph-stable decode preparation path.
DECODE_CUDA_GRAPH_SUPPORTED_METHODS = set(CANONICAL_SPARSE_METHODS)
TP_DECODE_CUDA_GRAPH_SUPPORTED_METHODS = {
    "",
    "streamingllm",
    "snapkv",
    "pyramidkv",
    "omnikv",
    "quest",
    "rkv",
    "skipkv",
}

_DEFAULT_PREFILL_POLICY_BY_METHOD = {
    "": PREFILL_POLICY_ALL_CHUNKED,
    "streamingllm": PREFILL_POLICY_ALL_CHUNKED,
    "snapkv": PREFILL_POLICY_ALL_CHUNKED,
    "pyramidkv": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "omnikv": PREFILL_POLICY_ALL_CHUNKED,
    "quest": PREFILL_POLICY_ALL_CHUNKED,
    "rkv": PREFILL_POLICY_ALL_CHUNKED,
    "skipkv": PREFILL_POLICY_ALL_CHUNKED,
    "deltakv": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
}

PREFILL_POLICY_BY_METHOD = {
    **_DEFAULT_PREFILL_POLICY_BY_METHOD,
    "vanilla": PREFILL_POLICY_ALL_CHUNKED,
    "attention-sink": PREFILL_POLICY_ALL_CHUNKED,
    "attention_sink": PREFILL_POLICY_ALL_CHUNKED,
    "r-kv": PREFILL_POLICY_ALL_CHUNKED,
    "r_kv": PREFILL_POLICY_ALL_CHUNKED,
    "skip-kv": PREFILL_POLICY_ALL_CHUNKED,
    "skip_kv": PREFILL_POLICY_ALL_CHUNKED,
    "deltakv-less-memory": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv_less_memory": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv-less-memory-cudagraph": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
    "deltakv_less_memory_cudagraph": PREFILL_POLICY_LONG_BS1FULL_SHORT_BATCH,
}


def normalize_sparse_method(method: str | None) -> str:
    if method is None:
        return ""
    normalized = str(method).strip().lower()
    return METHOD_ALIASES.get(normalized, normalized)


def is_deltakv_method(method: str | None) -> bool:
    return normalize_sparse_method(method) == "deltakv"


def is_decode_cuda_graph_supported(method: str | None) -> bool:
    return normalize_sparse_method(method) in DECODE_CUDA_GRAPH_SUPPORTED_METHODS


def is_tp_decode_cuda_graph_supported(method: str | None) -> bool:
    return normalize_sparse_method(method) in TP_DECODE_CUDA_GRAPH_SUPPORTED_METHODS


def validate_model_runtime_compatibility(
    *,
    model_type: str,
    sparse_method: str | None,
    tensor_parallel_size: int,
    expert_parallel_size: int,
    data_parallel_size: int,
    enforce_eager: bool,
    decode_cuda_graph: bool,
    enable_prefix_caching: bool,
) -> ModelRuntimeCompatibility | None:
    model_type = str(model_type or "").strip().lower()
    compatibility = MODEL_RUNTIME_COMPATIBILITY.get(model_type)
    if compatibility is None:
        return None

    method = normalize_sparse_method(sparse_method)
    if int(tensor_parallel_size) != 1 or int(data_parallel_size) != 1:
        raise ValueError(
            f"{model_type} {compatibility.parallel_mode} requires TP=1 and DP=1, got "
            f"TP={tensor_parallel_size}, EP={expert_parallel_size}, DP={data_parallel_size}."
        )
    if int(expert_parallel_size) <= 0:
        raise ValueError(
            f"{model_type} requires a positive expert_parallel_size, got {expert_parallel_size}."
        )
    if compatibility.requires_eager and not bool(enforce_eager):
        raise ValueError(f"{model_type} v1 requires enforce_eager=True.")
    if bool(decode_cuda_graph) and method not in compatibility.decode_cuda_graph_methods:
        supported = ", ".join(
            "'vanilla'" if item == "" else repr(item)
            for item in sorted(compatibility.decode_cuda_graph_methods)
        )
        raise ValueError(
            f"{model_type} v1 decode_cuda_graph is validated only for {supported}; "
            f"got method={method!r}."
        )
    if model_type == "qwen3_moe" and method == "skipkv":
        raise NotImplementedError(
            "Qwen3MoE + SkipKV requires a Qwen3MoE-matched steering asset and validation; "
            "no compatible asset is currently registered."
        )
    if model_type == "qwen3_moe" and method == "deltakv":
        raise NotImplementedError(
            "Qwen3MoE + DeltaKV is not part of the validated v1 compatibility matrix."
        )
    if method not in compatibility.sparse_methods:
        supported = ", ".join(
            "'vanilla'" if item == "" else repr(item)
            for item in sorted(compatibility.sparse_methods)
        )
        raise ValueError(
            f"Unsupported {model_type} {compatibility.parallel_mode} sparse method "
            f"{method!r}; validated methods: {supported}."
        )
    if bool(enable_prefix_caching) and method not in compatibility.prefix_cache_methods:
        supported = ", ".join(
            "'vanilla'" if item == "" else repr(item)
            for item in sorted(compatibility.prefix_cache_methods)
        )
        raise ValueError(
            f"{model_type} prefix caching is validated only for {supported}; got method={method!r}."
        )
    return compatibility


def get_default_prefill_schedule_policy(method: str | None) -> str:
    normalized = normalize_sparse_method(method)
    if normalized not in _DEFAULT_PREFILL_POLICY_BY_METHOD:
        supported = ", ".join(repr(name) for name in sorted(CANONICAL_SPARSE_METHODS) if name)
        aliases = ", ".join(repr(name) for name in sorted(SUPPORTED_SPARSE_METHOD_ALIASES))
        raise ValueError(
            f"Unsupported vllm_sparse_method={method!r}. Supported methods: '', {supported}. "
            f"Supported aliases: {aliases}."
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
