from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from sparsevllm.distributed.topology import ParallelMode, ParallelTopology
from sparsevllm.models.spec import MODEL_SPECS
from sparsevllm.operators.attention_capabilities import AttentionScoreKind

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
    # config files still load, but all code routes through sparse_method="deltakv".
    "deltakv-less-memory": "deltakv",
    "deltakv_less_memory": "deltakv",
    "deltakv-less-memory-cudagraph": "deltakv",
    "deltakv_less_memory_cudagraph": "deltakv",
}

CANONICAL_SPARSE_METHODS = {
    "",
    "streamingllm",
    "snapkv",
    "h2o",
    "pyramidkv",
    "omnikv",
    "quest",
    "rkv",
    "skipkv",
    "deltakv",
}

SUPPORTED_SPARSE_METHODS = set(CANONICAL_SPARSE_METHODS)
SUPPORTED_SPARSE_METHOD_ALIASES = {str(k) for k in METHOD_ALIASES if k is not None and str(k)}

PREFIX_CACHE_SUPPORTED_METHODS = {
    "",
    "streamingllm",
    "omnikv",
    "quest",
    "snapkv",
    "h2o",
    "pyramidkv",
    "rkv",
    "skipkv",
}

H2O_SUPPORTED_MODEL_TYPES = frozenset(MODEL_SPECS) - {"gemma4"}

SKIPKV_ASSET_MODEL_NAMES = frozenset(
    {
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Qwen-14B",
    }
)


@dataclass(frozen=True)
class ModelRuntimeCompatibility:
    sparse_methods: frozenset[str]
    prefix_cache_methods: frozenset[str]
    decode_graph_methods: frozenset[str] = frozenset()


class PrefillScoreCollectionKind(Enum):
    NONE = auto()
    METHOD_OWNED_POSTHOC_REDUCED = auto()
    MAIN_ATTENTION_REDUCED = auto()


@dataclass(frozen=True)
class SparsePrefillAttentionContract:
    main_score_kind: AttentionScoreKind
    score_collection: PrefillScoreCollectionKind


_PREFILL_POSTHOC_SCORE_METHODS = frozenset(
    {"snapkv", "pyramidkv", "h2o", "rkv"}
)

# These methods can request a score-producing decode launch on at least one
# layer or decode step.  The answer is deliberately static so Provider
# selection happens before CUDA Graph capture and never changes in run().
_DECODE_ATTENTION_SCORE_METHODS = frozenset(
    {"pyramidkv", "omnikv", "skipkv", "deltakv"}
)


def sparse_prefill_attention_contract(
    method: str | None,
    *,
    sparse_prefill_score_mode: str = "probability",
    h2o_prefill_score_window: int = 0,
) -> SparsePrefillAttentionContract:
    normalized = normalize_sparse_method(method)
    if normalized not in CANONICAL_SPARSE_METHODS:
        raise ValueError(f"Unknown sparse method {normalized!r}.")
    fused_h2o_score = (
        normalized == "h2o"
        and str(sparse_prefill_score_mode).strip().lower() == "logits"
        and int(h2o_prefill_score_window) == 0
    )
    if fused_h2o_score:
        return SparsePrefillAttentionContract(
            main_score_kind=AttentionScoreKind.RAW_QK_REDUCED,
            score_collection=PrefillScoreCollectionKind.MAIN_ATTENTION_REDUCED,
        )
    collection = (
        PrefillScoreCollectionKind.METHOD_OWNED_POSTHOC_REDUCED
        if normalized in _PREFILL_POSTHOC_SCORE_METHODS
        else PrefillScoreCollectionKind.NONE
    )
    return SparsePrefillAttentionContract(
        main_score_kind=AttentionScoreKind.NONE,
        score_collection=collection,
    )


def h2o_uses_fused_prefill_score(config) -> bool:
    return (
        normalize_sparse_method(getattr(config, "sparse_method", None)) == "h2o"
        and str(getattr(config, "sparse_prefill_score_mode", "probability")).strip().lower()
        == "logits"
        and int(getattr(config, "h2o_prefill_score_window", 0)) == 0
    )


def sparse_decode_attention_requires_scores(method: str | None) -> bool:
    """Return whether a prepared decode implementation must support scores."""

    normalized = normalize_sparse_method(method)
    if normalized not in CANONICAL_SPARSE_METHODS:
        raise ValueError(f"Unknown sparse method {normalized!r}.")
    return normalized in _DECODE_ATTENTION_SCORE_METHODS


_MOE_SPARSE_METHODS = frozenset(
    {"", "streamingllm", "snapkv", "h2o", "pyramidkv", "omnikv", "quest", "rkv"}
)

DENSE_MODEL_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=frozenset(CANONICAL_SPARSE_METHODS),
    prefix_cache_methods=frozenset(PREFIX_CACHE_SUPPORTED_METHODS),
    decode_graph_methods=frozenset(CANONICAL_SPARSE_METHODS),
)

QWEN3_MOE_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=_MOE_SPARSE_METHODS,
    prefix_cache_methods=frozenset(
        {"", "omnikv", "quest", "snapkv", "h2o", "pyramidkv", "rkv"}
    ),
    decode_graph_methods=_MOE_SPARSE_METHODS,
)

QWEN3_MOE_TP_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=_MOE_SPARSE_METHODS,
    prefix_cache_methods=frozenset({""}),
    decode_graph_methods=_MOE_SPARSE_METHODS,
)

QWEN3_MOE_TP_COMPATIBILITY = QWEN3_MOE_TP_EP_COMPATIBILITY

QWEN35_MOE_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=QWEN3_MOE_TP_EP_COMPATIBILITY.sparse_methods,
    prefix_cache_methods=frozenset({""}),
    decode_graph_methods=(
        QWEN3_MOE_TP_EP_COMPATIBILITY.decode_graph_methods
    ),
)

MINIMAX_M2_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=_MOE_SPARSE_METHODS,
    prefix_cache_methods=frozenset({"", "omnikv", "quest"}),
    decode_graph_methods=_MOE_SPARSE_METHODS,
)

MINIMAX_M2_TP_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=MINIMAX_M2_EP_COMPATIBILITY.sparse_methods,
    prefix_cache_methods=MINIMAX_M2_EP_COMPATIBILITY.prefix_cache_methods,
    decode_graph_methods=MINIMAX_M2_EP_COMPATIBILITY.decode_graph_methods,
)

GLM4_MOE_LITE_EP_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=frozenset(
        {"", "streamingllm", "snapkv", "h2o", "omnikv", "rkv"}
    ),
    prefix_cache_methods=frozenset(
        {"", "streamingllm", "snapkv", "h2o", "omnikv", "rkv"}
    ),
    decode_graph_methods=frozenset(
        {"", "streamingllm", "snapkv", "h2o", "omnikv", "rkv"}
    ),
)

GEMMA4_COMPATIBILITY = ModelRuntimeCompatibility(
    sparse_methods=frozenset({"", "streamingllm", "omnikv"}),
    prefix_cache_methods=frozenset({"", "streamingllm", "omnikv"}),
    decode_graph_methods=frozenset({"", "streamingllm", "omnikv"}),
)

MODEL_RUNTIME_COMPATIBILITY = {
    **{
        (model_type, ParallelMode.STANDARD): DENSE_MODEL_COMPATIBILITY
        for model_type in ("qwen2", "qwen3", "qwen3_5", "llama")
    },
    ("qwen3_moe", ParallelMode.STANDARD): QWEN3_MOE_EP_COMPATIBILITY,
    ("qwen3_moe", ParallelMode.OUTER_TP_MOE): QWEN3_MOE_TP_EP_COMPATIBILITY,
    ("qwen3_5_moe", ParallelMode.STANDARD): QWEN35_MOE_COMPATIBILITY,
    ("qwen3_5_moe", ParallelMode.OUTER_TP_MOE): QWEN35_MOE_COMPATIBILITY,
    ("minimax_m2", ParallelMode.STANDARD): MINIMAX_M2_EP_COMPATIBILITY,
    ("minimax_m2", ParallelMode.OUTER_TP_MOE): MINIMAX_M2_TP_EP_COMPATIBILITY,
    ("glm4_moe_lite", ParallelMode.STANDARD): GLM4_MOE_LITE_EP_COMPATIBILITY,
    ("glm4_moe_lite", ParallelMode.OUTER_TP_MOE): GLM4_MOE_LITE_EP_COMPATIBILITY,
    ("gemma4", ParallelMode.STANDARD): GEMMA4_COMPATIBILITY,
    ("gemma4", ParallelMode.OUTER_TP_MOE): GEMMA4_COMPATIBILITY,
}

# All shipped cache managers now expose a graph-stable decode preparation path.
DECODE_CUDA_GRAPH_SUPPORTED_METHODS = set(CANONICAL_SPARSE_METHODS)
TP_DECODE_CUDA_GRAPH_SUPPORTED_METHODS = {
    "",
    "streamingllm",
    "snapkv",
    "h2o",
    "pyramidkv",
    "omnikv",
    "quest",
    "rkv",
    "skipkv",
}


def decode_sparse_long_text_threshold(
    method: str,
    *,
    num_sink_tokens: int,
    decode_keep_tokens: int,
    num_recent_tokens: int,
) -> int:
    """Return the shared decode boundary between short and sparse graph families."""
    method = str(method or "")
    if not method:
        return 0
    if method in {"streamingllm", "attention-sink", "attention_sink"}:
        return int(num_sink_tokens) + int(num_recent_tokens)
    return (
        int(num_sink_tokens)
        + int(decode_keep_tokens)
        + int(num_recent_tokens)
    )


def decode_graph_path_id(method: str, is_long_text: bool) -> str:
    """Identify one graph-stable decode topology family."""
    method = str(method or "")
    if not method:
        return "dense"
    return "long" if is_long_text else "short"


_DEFAULT_PREFILL_POLICY_BY_METHOD = {
    "": PREFILL_POLICY_ALL_CHUNKED,
    "streamingllm": PREFILL_POLICY_ALL_CHUNKED,
    "snapkv": PREFILL_POLICY_ALL_CHUNKED,
    "h2o": PREFILL_POLICY_ALL_CHUNKED,
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


def validate_sparse_method_assets(method: str | None, model_path: str) -> None:
    if normalize_sparse_method(method) != "skipkv":
        return
    model_name = str(model_path).rstrip("/").split("/")[-1]
    if model_name not in SKIPKV_ASSET_MODEL_NAMES:
        supported = ", ".join(sorted(SKIPKV_ASSET_MODEL_NAMES))
        raise ValueError(
            "SkipKV is supported only for models with released steering assets: "
            f"{supported}. Got model basename {model_name!r}."
        )


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
    topology: ParallelTopology,
    decode_graph: bool,
    enable_prefix_caching: bool,
) -> ModelRuntimeCompatibility:
    model_type = str(model_type or "").strip().lower()
    method = normalize_sparse_method(sparse_method)
    compatibility = MODEL_RUNTIME_COMPATIBILITY.get((model_type, topology.mode))
    if compatibility is None:
        raise NotImplementedError(
            f"Unsupported Sparse-vLLM model_type={model_type!r} with "
            f"parallel mode={topology.mode.value!r}."
        )

    if bool(decode_graph) and method not in compatibility.decode_graph_methods:
        supported = ", ".join(
            "'vanilla'" if item == "" else repr(item)
            for item in sorted(compatibility.decode_graph_methods)
        )
        raise ValueError(
            f"{model_type} v1 decode_graph is validated only for {supported}; "
            f"got method={method!r}."
        )
    if method not in compatibility.sparse_methods:
        supported = ", ".join(
            "'vanilla'" if item == "" else repr(item)
            for item in sorted(compatibility.sparse_methods)
        )
        raise ValueError(
            f"Unsupported {model_type} {topology.mode.value} sparse method "
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
            f"Unsupported sparse_method={method!r}. Supported methods: '', {supported}. "
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
