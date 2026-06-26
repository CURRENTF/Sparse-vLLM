import os
from dataclasses import dataclass
from typing import Any, Union

from transformers import AutoConfig

from sparsevllm.method_registry import (
    DECODE_CUDA_GRAPH_SUPPORTED_METHODS,
    PREFILL_POLICY_AUTO,
    PREFIX_CACHE_SUPPORTED_METHODS,
    SUPPORTED_SPARSE_METHODS,
    is_decode_cuda_graph_supported,
    is_tp_decode_cuda_graph_supported,
    normalize_sparse_method,
    resolve_prefill_schedule_policy,
)
from sparsevllm.engine.prefix_cache import resolve_prefix_cache_block_size
from sparsevllm.utils.log import logger, log_once

try:
    from transformers import Qwen3Config
except ImportError:
    Qwen3Config = AutoConfig


def _coerce_bool_config(name: str, value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"1", "true", "yes", "on"}:
            return True
        if raw in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean or explicit true/false string, got {value!r}.")


def _coerce_optional_positive_int(name: str, value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw.isdecimal():
            raise ValueError(f"{name} must be a positive integer, got {value!r}.")
        parsed = int(raw)
    else:
        raise ValueError(f"{name} must be a positive integer, got {value!r}.")
    if parsed <= 0:
        raise ValueError(f"{name} must be > 0 when set, got {parsed}.")
    return parsed


SUPPORTED_SKIPKV_MODEL_NAMES = frozenset(
    {
        "DeepSeek-R1-Distill-Llama-8B",
        "DeepSeek-R1-Distill-Qwen-7B",
        "DeepSeek-R1-Distill-Qwen-14B",
    }
)


def _model_path_basename(model_path: str) -> str:
    return str(model_path).rstrip("/").split("/")[-1]


def _default_decode_cuda_graph_capture_sizes(max_decoding_seqs: int) -> list[int]:
    max_decoding_seqs = int(max_decoding_seqs)
    if max_decoding_seqs <= 0:
        raise ValueError(f"max_decoding_seqs must be > 0, got {max_decoding_seqs}.")

    sizes: list[int] = []
    size = 1
    while size < max_decoding_seqs:
        sizes.append(size)
        size *= 2
    sizes.append(size)
    return sizes


def _resolve_decode_cuda_graph_capture_sizes(
    value: str | int | list[int] | tuple[int, ...] | None,
    max_decoding_seqs: int,
) -> list[int]:
    if value is None:
        sizes = _default_decode_cuda_graph_capture_sizes(max_decoding_seqs)
    elif isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "auto"}:
            sizes = _default_decode_cuda_graph_capture_sizes(max_decoding_seqs)
        else:
            try:
                sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError(
                    "decode_cuda_graph_capture_sizes must be 'auto' or a comma-separated "
                    f"integer list, got {value!r}."
                ) from exc
    elif isinstance(value, int):
        sizes = [int(value)]
    elif isinstance(value, (list, tuple)):
        sizes = [int(item) for item in value]
    else:
        raise ValueError(
            "decode_cuda_graph_capture_sizes must be 'auto', an int, a list/tuple of ints, "
            f"or None, got {type(value).__name__}."
        )

    sizes = sorted(set(sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"decode_cuda_graph_capture_sizes must contain positive integers, got {sizes}.")
    if sizes[-1] < int(max_decoding_seqs):
        raise ValueError(
            "decode_cuda_graph_capture_sizes must cover max_decoding_seqs: "
            f"max capture size {sizes[-1]} < max_decoding_seqs {int(max_decoding_seqs)}."
        )
    return sizes


def _default_decode_cuda_graph_context_sizes(max_model_len: int) -> list[int]:
    """Default decode graph context buckets: 1k, 2k, 4k, ... up to max_model_len."""
    max_model_len = int(max_model_len)
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be > 0, got {max_model_len}.")

    size = min(1024, max_model_len)
    sizes: list[int] = []
    while size < max_model_len:
        sizes.append(size)
        size *= 2
    sizes.append(size)
    return sorted(set(sizes))


def _resolve_decode_cuda_graph_context_sizes(
    value: str | int | list[int] | tuple[int, ...] | None,
    max_model_len: int,
) -> list[int]:
    if value is None:
        sizes = _default_decode_cuda_graph_context_sizes(max_model_len)
    elif isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"", "auto"}:
            sizes = _default_decode_cuda_graph_context_sizes(max_model_len)
        else:
            try:
                sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
            except ValueError as exc:
                raise ValueError(
                    "decode_cuda_graph_context_sizes must be 'auto' or a comma-separated "
                    f"integer list, got {value!r}."
                ) from exc
    elif isinstance(value, int):
        sizes = [int(value)]
    elif isinstance(value, (list, tuple)):
        sizes = [int(item) for item in value]
    else:
        raise ValueError(
            "decode_cuda_graph_context_sizes must be 'auto', an int, a list/tuple of ints, "
            f"or None, got {type(value).__name__}."
        )

    sizes = sorted(set(sizes))
    if not sizes or any(size <= 0 for size in sizes):
        raise ValueError(f"decode_cuda_graph_context_sizes must contain positive integers, got {sizes}.")
    return sizes


def _normalize_decode_cuda_graph_context_policy(value: str | None) -> str:
    policy = str(value or "current").strip().lower()
    if policy in {"cur", "now"}:
        return "current"
    if policy in {"request", "final"}:
        return "requested"
    if policy not in {"current", "requested"}:
        raise ValueError(
            "decode_cuda_graph_context_policy must be 'current' or 'requested', "
            f"got {policy!r}."
        )
    return policy


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs_in_batch: int = 32  # 不能设置太大
    max_model_len: int = 128_000
    max_decoding_seqs: int = 64

    chunk_prefill_size: int = 8192
    mlp_chunk_size: int = 16384
    prefill_schedule_policy: str = PREFILL_POLICY_AUTO
    gpu_memory_utilization: float = 0.8
    device_memory_utilization: float | None = None
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    hf_config: Union[Qwen3Config, AutoConfig] | None = None
    eos: int = -1
    num_kvcache_slots: int | list = -1

    # Sparse Attention Config
    vllm_sparse_method: str = ""  # "", "streamingllm", "snapkv", "pyramidkv", "omnikv", "quest", "rkv", "skipkv", "deltakv"; legacy deltakv-less-memory aliases normalize to deltakv.

    # Prefix Cache Config
    enable_prefix_caching: bool = False
    prefix_cache_block_size: int | None = None
    prefix_cache_max_blocks: int | None = None
    prefix_cache_salt: str = ""

    # General Sparse Config
    num_sink_tokens: int = 64
    num_recent_tokens: int = 512
    decode_keep_tokens: int = 4096

    # OmniKV Config
    obs_layer_ids: list[int] = None  # None means auto-calculate based on full_attn_layers (useful for omnikv)
    full_attn_layers: str | list[int] = "0" # useful for omnikv
    decode_cuda_graph: bool = False
    decode_cuda_graph_capture_sampling: bool = False
    decode_cuda_graph_capture_sizes: str | int | list[int] | tuple[int, ...] | None = "auto"
    # Static decode/CUDA graph context buckets.  The default produces
    # 1024, 2048, 4096, ... up to max_model_len so a short request never
    # inherits a previously captured 128k graph.
    decode_cuda_graph_context_sizes: str | int | list[int] | tuple[int, ...] | None = "auto"
    # "current" buckets by the real current decode length; "requested" keeps
    # the old final-length capacity behavior for compatibility/debugging.
    decode_cuda_graph_context_policy: str = "current"
    # Optional LRU cap for captured graphs.  None keeps all bucketed graphs.
    decode_cuda_graph_max_cached_graphs: int | None = None
    omnikv_decode_cuda_graph: bool = False
    sparse_attn_score_dtype: str = "float32"
    decode_graph: bool | None = None
    decode_graph_capture_sampling: bool | None = None
    decode_graph_capture_sizes: str | int | list[int] | tuple[int, ...] | None = None
    omnikv_decode_graph: bool | None = None

    # QuEST Config
    quest_chunk_size: int = 16
    quest_token_budget: int = 1024
    quest_skip_layers: int = 2

    # SnapKV Config
    snapkv_window_size: int = 32
    snapkv_num_full_layers: int = 0  # 前多少层不进行驱逐

    # R-KV Config
    rkv_compression_interval: int = 128
    rkv_observation_tokens: int = 8
    rkv_alpha: float = 0.1
    rkv_similarity_threshold: float = 0.8
    rkv_recent_similar_keep: int = 1
    rkv_max_redundancy_tokens: int = 4096
    # 0 means score the full candidate set, matching R-KV's budget-candidate selection.
    # Positive values are an explicit speed/quality approximation.
    rkv_redundancy_window: int = 0

    # SkipKV Config.
    skipkv_compression_interval: int = 128
    skipkv_alpha: float = 0.1
    skipkv_similarity_threshold: float = 0.95
    skipkv_segment_size: int = 32
    skipkv_max_redundancy_tokens: int = 4096
    skipkv_redundancy_window: int = 64
    skipkv_enable_sentence_scoring: bool = True
    skipkv_sentence_score_weight: float = 1.0
    skipkv_sentence_min_tokens: int = 4
    skipkv_sentence_max_tokens: int = 256
    skipkv_sentence_embedding_layer: int = -1
    skipkv_max_tracked_sentences: int = 256
    skipkv_enable_activation_steering: bool = False
    skipkv_steering_vector_path: str | None = None
    skipkv_steering_layer: int = -1
    skipkv_steering_alpha: float = 0.0
    skipkv_steering_alpha_increment: float = 0.0
    skipkv_steering_alpha_max: float = 0.0
    
    # PyramidKV Config
    pyramid_layer_ratios: list[float] | None = None  # 每层的 KV budget 比例
    pyramidkv_start_layer: int = 0
    pyramidkv_start_ratio: float = 0.6
    pyramidkv_least_layer: int | None = None
    pyramidkv_least_ratio: float = 0.01

    # DeltaKV Config
    deltakv_path: str | None = None
    deltakv_k_neighbors: int = 4
    cluster_ratio: float = 0.1
    cluster_metric: str = 'l2'  # 'l2', 'dot', 'cosine', 'fastdot' (approx; fastest)
    cluster_on_kv: bool = True
    use_compression: bool = True
    kv_compressed_size: int = 128
    kv_quant_bits: int = 4
    kv_quant_group_size: int = 0
    enable_sparse_ref_fp8: bool = False
    # Optional residual quantization for layers listed in full_attn_layers.
    # 0 keeps the previous BF16/FP16 full-layer KV behavior. 2/4 store old
    # full-layer tokens as DeltaKV-style residuals and reconstruct a dense view
    # before full attention.
    full_layer_kv_quant_bits: int = 0
    full_layer_cluster_ratio: float = 0.0  # <=0 reuses cluster_ratio
    full_layer_kivi_group_size: int = 32
    full_layer_kivi_residual_length: int = 32
    full_layer_kivi_decode_block_seq: int = 256
    full_layer_kivi_decode_block_n: int = 16
    full_layer_kivi_decode_num_warps: int = 2
    full_layer_kivi_decode_num_stages: int = 3
    enable_full_layer_kivi_quant: bool = True
    enable_full_layer_kivi_fused_decode: bool = False
    enable_full_layer_kivi_grouped_decode: bool = False
    # Legacy config compatibility only. Full-layer KIVI decode uses the direct
    # packed backend whenever KIVI quantization is enabled.
    enable_full_layer_kivi_dense_decode: bool = False
    pool_kernel_size: int = 1
    # Legacy symmetric compressor controls (for backward compatibility).
    use_nonlinear_compressor: bool = True
    compressor_intermediate_size: int = 2048
    compressor_linear_bias: bool = True
    # New directional compressor controls (match latest DeltaKV training code).
    # Values: auto|linear|mlp_gelu|mlp_swiglu
    compressor_down_type: str = "auto"
    compressor_up_type: str = "auto"
    compressor_down_intermediate_size: int = -1
    compressor_up_intermediate_size: int = -1
    # DeltaKV memory split: reserve a fraction of available KV memory for the sparse full-KV pool
    # (centers + buffer + temp reconstruction slots). Larger values reduce full/latent capacity
    # but improve robustness at large batch sizes / long contexts.
    deltakv_full_pool_reserve_ratio: float = 0.1
    # Cap DeltaKV packed-cache capacity to the configured resident
    # token budget instead of consuming all available memory. This keeps long
    # context experiments reproducible and leaves workspace memory for kernels.
    deltakv_cache_capacity_margin: float = 1.05
    # Extra center slots above the exact fixed-stride center-policy estimate.
    # This avoids allocating cluster_ratio * max_tokens plus no scheduling margin.
    deltakv_center_capacity_margin: float = 1.5
    # Triton kernels: group multiple KV heads per program to reduce redundant loads.
    deltakv_triton_gather_heads_per_program: int = 4
    deltakv_triton_reconstruct_heads_per_program: int = 4
    deltakv_cluster_gather_chunk_size: int = 16384
    
    enable_profiler: bool = False
    throughput_log_interval_s: float = 10.0
    allow_missing_deltakv_path: bool = False
    allow_unknown_config_keys: bool = False

    def _normalize_platform_aliases(self):
        if self.device_memory_utilization is not None:
            self.gpu_memory_utilization = float(self.device_memory_utilization)
        self.device_memory_utilization = float(self.gpu_memory_utilization)

        if self.decode_graph is not None:
            self.decode_cuda_graph = _coerce_bool_config("decode_graph", self.decode_graph)
        else:
            self.decode_cuda_graph = _coerce_bool_config("decode_cuda_graph", self.decode_cuda_graph)
        self.decode_graph = bool(self.decode_cuda_graph)

        if self.decode_graph_capture_sampling is not None:
            self.decode_cuda_graph_capture_sampling = _coerce_bool_config(
                "decode_graph_capture_sampling",
                self.decode_graph_capture_sampling,
            )
        else:
            self.decode_cuda_graph_capture_sampling = _coerce_bool_config(
                "decode_cuda_graph_capture_sampling",
                self.decode_cuda_graph_capture_sampling,
            )
        self.decode_graph_capture_sampling = bool(self.decode_cuda_graph_capture_sampling)

        if self.decode_graph_capture_sizes is not None:
            self.decode_cuda_graph_capture_sizes = self.decode_graph_capture_sizes

        if self.omnikv_decode_graph is not None:
            self.omnikv_decode_cuda_graph = _coerce_bool_config(
                "omnikv_decode_graph",
                self.omnikv_decode_graph,
            )
        else:
            self.omnikv_decode_cuda_graph = _coerce_bool_config(
                "omnikv_decode_cuda_graph",
                self.omnikv_decode_cuda_graph,
            )
        self.omnikv_decode_graph = bool(self.omnikv_decode_cuda_graph)

    def __post_init__(self):
        if os.getenv("PROFILER_SVLLM"):
            self.enable_profiler = True

        raw_sparse_method = self.vllm_sparse_method
        raw_sparse_method_normalized = "" if raw_sparse_method is None else str(raw_sparse_method).strip().lower()
        legacy_deltakv_graph_method = raw_sparse_method_normalized in {
            "deltakv-less-memory-cudagraph",
            "deltakv_less_memory_cudagraph",
        }

        self.vllm_sparse_method = normalize_sparse_method(self.vllm_sparse_method)
        if self.vllm_sparse_method not in SUPPORTED_SPARSE_METHODS:
            supported = ", ".join(repr(method) for method in sorted(SUPPORTED_SPARSE_METHODS) if method)
            raise ValueError(
                f"Unsupported vllm_sparse_method={self.vllm_sparse_method!r}. "
                f"Supported methods: '', {supported}."
            )
        self.enable_prefix_caching = _coerce_bool_config("enable_prefix_caching", self.enable_prefix_caching)
        self.prefix_cache_block_size = _coerce_optional_positive_int(
            "prefix_cache_block_size",
            self.prefix_cache_block_size,
        )
        self.prefix_cache_max_blocks = _coerce_optional_positive_int(
            "prefix_cache_max_blocks",
            self.prefix_cache_max_blocks,
        )
        if self.enable_prefix_caching and self.vllm_sparse_method not in PREFIX_CACHE_SUPPORTED_METHODS:
            raise ValueError("prefix caching only supports vanilla, omnikv, quest.")
        self.prefix_cache_salt = str(self.prefix_cache_salt or "")
        self.prefill_schedule_policy = resolve_prefill_schedule_policy(
            self.vllm_sparse_method,
            self.prefill_schedule_policy,
        )
        
        if int(self.mlp_chunk_size) <= 0:
            raise ValueError(f"mlp_chunk_size must be > 0, got {self.mlp_chunk_size}.")
        self.mlp_chunk_size = int(self.mlp_chunk_size)
        if int(self.deltakv_cluster_gather_chunk_size) <= 0:
            raise ValueError(
                "deltakv_cluster_gather_chunk_size must be > 0, "
                f"got {self.deltakv_cluster_gather_chunk_size}."
            )
        self.deltakv_cluster_gather_chunk_size = int(self.deltakv_cluster_gather_chunk_size)
        self.sparse_attn_score_dtype = str(self.sparse_attn_score_dtype or "float32").strip().lower()
        if self.sparse_attn_score_dtype not in {"float32", "bfloat16", "float16"}:
            raise ValueError(
                "sparse_attn_score_dtype must be 'float32', 'bfloat16', or 'float16', "
                f"got {self.sparse_attn_score_dtype!r}."
            )
        self.full_layer_kv_quant_bits = int(self.full_layer_kv_quant_bits or 0)
        if self.full_layer_kv_quant_bits not in (0, 2, 4):
            raise ValueError(
                "full_layer_kv_quant_bits must be 0, 2, or 4, "
                f"got {self.full_layer_kv_quant_bits}."
            )
        self.full_layer_cluster_ratio = float(self.full_layer_cluster_ratio or 0.0)
        if self.full_layer_cluster_ratio < 0.0:
            raise ValueError(f"full_layer_cluster_ratio must be >= 0, got {self.full_layer_cluster_ratio}.")
        self.kv_quant_bits = int(self.kv_quant_bits or 0)
        if self.kv_quant_bits not in (0, 2, 4):
            raise ValueError(f"kv_quant_bits must be 0, 2, or 4, got {self.kv_quant_bits}.")
        self.kv_quant_group_size = int(self.kv_quant_group_size or 0)
        if self.kv_quant_group_size < 0:
            raise ValueError(f"kv_quant_group_size must be >= 0, got {self.kv_quant_group_size}.")
        self.full_layer_kivi_group_size = int(self.full_layer_kivi_group_size or 32)
        if self.full_layer_kivi_group_size <= 0:
            raise ValueError(
                "full_layer_kivi_group_size must be > 0, "
                f"got {self.full_layer_kivi_group_size}."
            )
        self.full_layer_kivi_residual_length = int(
            self.full_layer_kivi_residual_length or self.full_layer_kivi_group_size
        )
        if self.full_layer_kivi_residual_length <= 0:
            raise ValueError(
                "full_layer_kivi_residual_length must be > 0, "
                f"got {self.full_layer_kivi_residual_length}."
            )
        self.full_layer_kivi_decode_block_seq = int(self.full_layer_kivi_decode_block_seq or 256)
        if self.full_layer_kivi_decode_block_seq <= 0 or self.full_layer_kivi_decode_block_seq % 16 != 0:
            raise ValueError(
                "full_layer_kivi_decode_block_seq must be a positive multiple of 16, "
                f"got {self.full_layer_kivi_decode_block_seq}."
            )
        self.full_layer_kivi_decode_block_n = int(self.full_layer_kivi_decode_block_n or 16)
        if self.full_layer_kivi_decode_block_n <= 0 or self.full_layer_kivi_decode_block_n % 16 != 0:
            raise ValueError(
                "full_layer_kivi_decode_block_n must be a positive multiple of 16, "
                f"got {self.full_layer_kivi_decode_block_n}."
            )
        self.full_layer_kivi_decode_num_warps = int(self.full_layer_kivi_decode_num_warps or 2)
        if self.full_layer_kivi_decode_num_warps not in {1, 2, 4, 8}:
            raise ValueError(
                "full_layer_kivi_decode_num_warps must be one of 1, 2, 4, or 8, "
                f"got {self.full_layer_kivi_decode_num_warps}."
            )
        self.full_layer_kivi_decode_num_stages = int(self.full_layer_kivi_decode_num_stages or 3)
        if self.full_layer_kivi_decode_num_stages <= 0:
            raise ValueError(
                "full_layer_kivi_decode_num_stages must be > 0, "
                f"got {self.full_layer_kivi_decode_num_stages}."
            )
        self.enable_full_layer_kivi_fused_decode = bool(self.enable_full_layer_kivi_fused_decode)
        self.enable_full_layer_kivi_grouped_decode = bool(self.enable_full_layer_kivi_grouped_decode)
        self.enable_full_layer_kivi_dense_decode = bool(self.enable_full_layer_kivi_dense_decode)
        if self.enable_full_layer_kivi_fused_decode:
            raise ValueError(
                "enable_full_layer_kivi_fused_decode was removed; full-layer KIVI decode now "
                "uses the direct packed backend."
            )
        if self.enable_full_layer_kivi_grouped_decode:
            raise ValueError(
                "enable_full_layer_kivi_grouped_decode was removed; full-layer KIVI decode now "
                "uses the direct packed backend."
            )
        self.deltakv_full_pool_reserve_ratio = float(self.deltakv_full_pool_reserve_ratio or 0.0)
        if self.deltakv_full_pool_reserve_ratio < 0.0 or self.deltakv_full_pool_reserve_ratio >= 1.0:
            raise ValueError(
                "deltakv_full_pool_reserve_ratio must be in [0, 1), "
                f"got {self.deltakv_full_pool_reserve_ratio}."
            )
        self.deltakv_cache_capacity_margin = float(self.deltakv_cache_capacity_margin or 1.0)
        if self.deltakv_cache_capacity_margin < 1.0:
            raise ValueError(
                "deltakv_cache_capacity_margin must be >= 1.0, "
                f"got {self.deltakv_cache_capacity_margin}."
            )
        self.deltakv_center_capacity_margin = float(self.deltakv_center_capacity_margin or 1.0)
        if self.deltakv_center_capacity_margin < 1.0:
            raise ValueError(
                "deltakv_center_capacity_margin must be >= 1.0, "
                f"got {self.deltakv_center_capacity_margin}."
            )

        if not os.path.isdir(self.model):
            raise FileNotFoundError(f"Model directory does not exist: {self.model}")
        if self.vllm_sparse_method == "skipkv":
            model_name = _model_path_basename(self.model)
            if model_name not in SUPPORTED_SKIPKV_MODEL_NAMES:
                supported = ", ".join(sorted(SUPPORTED_SKIPKV_MODEL_NAMES))
                raise ValueError(
                    "SkipKV is supported only for the official models with released steering vectors: "
                    f"{supported}. Got model basename {model_name!r} from model path {self.model!r}."
                )
        if int(self.max_decoding_seqs) <= 0:
            raise ValueError(f"max_decoding_seqs must be > 0, got {self.max_decoding_seqs}.")
        self.max_decoding_seqs = int(self.max_decoding_seqs)
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError(f"tensor_parallel_size must be in [1, 8], got {self.tensor_parallel_size}.")
        self._normalize_platform_aliases()
        if legacy_deltakv_graph_method:
            self.decode_cuda_graph = True
            self.decode_graph = True
        if self.decode_cuda_graph_max_cached_graphs is not None:
            self.decode_cuda_graph_max_cached_graphs = int(self.decode_cuda_graph_max_cached_graphs)
            if self.decode_cuda_graph_max_cached_graphs <= 0:
                raise ValueError(
                    "decode_cuda_graph_max_cached_graphs must be a positive integer or None, "
                    f"got {self.decode_cuda_graph_max_cached_graphs}."
                )
        if self.omnikv_decode_cuda_graph:
            if self.vllm_sparse_method != "omnikv":
                raise ValueError(
                    "omnikv_decode_cuda_graph is only valid with vllm_sparse_method='omnikv'."
                )
            self.decode_cuda_graph = True
        if self.decode_cuda_graph_capture_sampling and not self.decode_cuda_graph:
            raise ValueError("decode_cuda_graph_capture_sampling requires decode_cuda_graph=True.")
        self.decode_cuda_graph_context_policy = _normalize_decode_cuda_graph_context_policy(
            self.decode_cuda_graph_context_policy
        )
        if self.decode_cuda_graph:
            if self.enable_prefix_caching:
                if self.decode_cuda_graph_capture_sampling:
                    raise ValueError(
                        "prefix caching with decode_cuda_graph does not support "
                        "decode_cuda_graph_capture_sampling=True yet."
                    )
                if self.tensor_parallel_size > 1:
                    raise ValueError(
                        "prefix caching with decode_cuda_graph supports tensor_parallel_size=1 only."
                    )
            if self.tensor_parallel_size > 1:
                if self.decode_cuda_graph_capture_sampling:
                    raise ValueError(
                        "decode_cuda_graph_capture_sampling is disabled when tensor_parallel_size > 1 "
                        "because TP workers do not materialize rank-0 gathered logits."
                    )
                if not is_tp_decode_cuda_graph_supported(self.vllm_sparse_method):
                    supported = ", ".join(
                        repr(method)
                        for method in sorted(DECODE_CUDA_GRAPH_SUPPORTED_METHODS)
                        if method and is_tp_decode_cuda_graph_supported(method)
                    )
                    raise ValueError(
                        "decode_cuda_graph with tensor_parallel_size > 1 supports these methods only: "
                        f"'', {supported}. v1 excludes DeltaKV and QuEST."
                    )
                log_once(
                    "decode_cuda_graph with tensor_parallel_size > 1 uses TP-local sparse selection: "
                    "each rank selects sparse tokens from its local heads/KV heads without cross-rank "
                    "sparse-index aggregation, so sparse behavior is not guaranteed equivalent to TP=1 "
                    "or global-head sparse selection.",
                    level="WARNING",
                )
            elif not is_decode_cuda_graph_supported(self.vllm_sparse_method):
                supported = ", ".join(
                    repr(method) for method in sorted(DECODE_CUDA_GRAPH_SUPPORTED_METHODS) if method
                )
                raise ValueError(f"decode_cuda_graph supports these methods only: '', {supported}.")
            self.decode_cuda_graph_capture_sizes = _resolve_decode_cuda_graph_capture_sizes(
                self.decode_cuda_graph_capture_sizes,
                self.max_decoding_seqs,
            )
            self.decode_cuda_graph_context_sizes = _resolve_decode_cuda_graph_context_sizes(
                self.decode_cuda_graph_context_sizes,
                self.max_model_len,
            )
        self.decode_graph = bool(self.decode_cuda_graph)
        self.decode_graph_capture_sampling = bool(self.decode_cuda_graph_capture_sampling)
        self.decode_graph_capture_sizes = self.decode_cuda_graph_capture_sizes
        self.omnikv_decode_graph = bool(self.omnikv_decode_cuda_graph)
        if isinstance(self.deltakv_path, str):
            deltakv_path = self.deltakv_path.strip()
            self.deltakv_path = None if deltakv_path.lower() in {"", "none", "null"} else deltakv_path
        try:
            self.hf_config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                "AutoConfig.from_pretrained failed. Refusing to silently fall back to raw "
                f"`config.json`. model={self.model} error={type(e).__name__}: {e}"
            ) from e
        if getattr(self.hf_config, "model_type", "") in {"deepseek_v2", "deepseek_v32"}:
            raise NotImplementedError(
                f"Unsupported Sparse-vLLM model_type={self.hf_config.model_type!r}. "
                "Supported model types: qwen2, qwen3, llama."
            )
        if self.max_model_len > self.hf_config.max_position_embeddings:
            logger.warning('max_model_len > model.max_position_embeddings 输出可能不正常')
            self.hf_config.max_position_embeddings = self.max_model_len

        if self.max_num_seqs_in_batch > 32:
            logger.warning('max_num_seqs_in_batch 过大或许会占用太多显存')

        if isinstance(self.full_attn_layers, str):
            layers = self.full_attn_layers.strip()
            self.full_attn_layers = [] if not layers else [int(x) for x in layers.split(",")]

        if self.quest_chunk_size <= 0:
            raise ValueError("quest_chunk_size 必须 > 0")
        if self.quest_token_budget <= 0:
            raise ValueError("quest_token_budget 必须 > 0")
        if self.quest_skip_layers < 0:
            raise ValueError("quest_skip_layers 不能 < 0")
        self.rkv_compression_interval = int(self.rkv_compression_interval or 0)
        if self.rkv_compression_interval <= 0:
            raise ValueError(
                f"rkv_compression_interval must be > 0, got {self.rkv_compression_interval}."
            )
        self.rkv_observation_tokens = int(self.rkv_observation_tokens or 0)
        if self.rkv_observation_tokens <= 0:
            raise ValueError(
                f"rkv_observation_tokens must be > 0, got {self.rkv_observation_tokens}."
            )
        if self.rkv_observation_tokens > 128:
            raise ValueError(
                "rkv_observation_tokens must be <= 128 because the prefill score kernel "
                f"supports at most 128 query tokens, got {self.rkv_observation_tokens}."
            )
        if self.rkv_observation_tokens > self.rkv_compression_interval:
            raise ValueError(
                "rkv_observation_tokens must be <= rkv_compression_interval so the query cache "
                "can be refreshed between decode evictions, "
                f"got observation={self.rkv_observation_tokens} interval={self.rkv_compression_interval}."
            )
        self.rkv_alpha = float(self.rkv_alpha)
        if not 0.0 <= self.rkv_alpha <= 1.0:
            raise ValueError(f"rkv_alpha must be in [0, 1], got {self.rkv_alpha}.")
        self.rkv_similarity_threshold = float(self.rkv_similarity_threshold)
        if not 0.0 <= self.rkv_similarity_threshold <= 1.0:
            raise ValueError(
                "rkv_similarity_threshold must be in [0, 1], "
                f"got {self.rkv_similarity_threshold}."
            )
        self.rkv_recent_similar_keep = int(self.rkv_recent_similar_keep)
        if self.rkv_recent_similar_keep < 0:
            raise ValueError(
                f"rkv_recent_similar_keep must be >= 0, got {self.rkv_recent_similar_keep}."
            )
        self.rkv_max_redundancy_tokens = int(self.rkv_max_redundancy_tokens or 0)
        if self.rkv_max_redundancy_tokens <= 0:
            raise ValueError(
                f"rkv_max_redundancy_tokens must be > 0, got {self.rkv_max_redundancy_tokens}."
            )
        self.rkv_redundancy_window = int(self.rkv_redundancy_window or 0)
        if self.rkv_redundancy_window < 0:
            raise ValueError(
                f"rkv_redundancy_window must be >= 0, got {self.rkv_redundancy_window}."
            )
        if 0 < self.rkv_redundancy_window > self.rkv_max_redundancy_tokens:
            raise ValueError(
                "rkv_redundancy_window must be <= rkv_max_redundancy_tokens, "
                f"got window={self.rkv_redundancy_window} max={self.rkv_max_redundancy_tokens}."
            )
        if self.vllm_sparse_method == "rkv":
            log_once(
                "R-KV support is an approximation of the official implementation: "
                "Sparse-VLLM uses one shared physical token index set across KV heads, "
                "so official per-KV-head token selection is not fully reproduced. "
                f"rkv_redundancy_window={self.rkv_redundancy_window}; values > 0 score "
                "redundancy only over the trailing candidate tokens.",
                level="WARNING",
            )
        self.skipkv_compression_interval = int(self.skipkv_compression_interval or 0)
        if self.skipkv_compression_interval <= 0:
            raise ValueError(
                "skipkv_compression_interval must be > 0, "
                f"got {self.skipkv_compression_interval}."
            )
        self.skipkv_alpha = float(self.skipkv_alpha)
        if self.skipkv_alpha < 0.0:
            raise ValueError(f"skipkv_alpha must be >= 0, got {self.skipkv_alpha}.")
        self.skipkv_similarity_threshold = float(self.skipkv_similarity_threshold)
        if not 0.0 <= self.skipkv_similarity_threshold <= 1.0:
            raise ValueError(
                "skipkv_similarity_threshold must be in [0, 1], "
                f"got {self.skipkv_similarity_threshold}."
            )
        self.skipkv_segment_size = int(self.skipkv_segment_size or 0)
        if self.skipkv_segment_size <= 0:
            raise ValueError(f"skipkv_segment_size must be > 0, got {self.skipkv_segment_size}.")
        self.skipkv_max_redundancy_tokens = int(self.skipkv_max_redundancy_tokens or 0)
        if self.skipkv_max_redundancy_tokens <= 0:
            raise ValueError(
                "skipkv_max_redundancy_tokens must be > 0, "
                f"got {self.skipkv_max_redundancy_tokens}."
            )
        self.skipkv_redundancy_window = int(self.skipkv_redundancy_window or 0)
        if self.skipkv_redundancy_window <= 0:
            raise ValueError(
                "skipkv_redundancy_window must be > 0, "
                f"got {self.skipkv_redundancy_window}."
            )
        if self.skipkv_redundancy_window > self.skipkv_max_redundancy_tokens:
            raise ValueError(
                "skipkv_redundancy_window must be <= skipkv_max_redundancy_tokens, "
                f"got window={self.skipkv_redundancy_window} max={self.skipkv_max_redundancy_tokens}."
            )
        self.skipkv_enable_sentence_scoring = bool(self.skipkv_enable_sentence_scoring)
        self.skipkv_sentence_score_weight = float(self.skipkv_sentence_score_weight)
        if self.skipkv_sentence_score_weight < 0.0:
            raise ValueError(
                "skipkv_sentence_score_weight must be >= 0, "
                f"got {self.skipkv_sentence_score_weight}."
            )
        self.skipkv_sentence_min_tokens = int(self.skipkv_sentence_min_tokens or 0)
        if self.skipkv_sentence_min_tokens <= 0:
            raise ValueError(
                "skipkv_sentence_min_tokens must be > 0, "
                f"got {self.skipkv_sentence_min_tokens}."
            )
        self.skipkv_sentence_max_tokens = int(self.skipkv_sentence_max_tokens or 0)
        if self.skipkv_sentence_max_tokens < self.skipkv_sentence_min_tokens:
            raise ValueError(
                "skipkv_sentence_max_tokens must be >= skipkv_sentence_min_tokens, "
                f"got max={self.skipkv_sentence_max_tokens} min={self.skipkv_sentence_min_tokens}."
            )
        self.skipkv_sentence_embedding_layer = int(self.skipkv_sentence_embedding_layer)
        self.skipkv_max_tracked_sentences = int(self.skipkv_max_tracked_sentences or 0)
        if self.skipkv_max_tracked_sentences <= 0:
            raise ValueError(
                "skipkv_max_tracked_sentences must be > 0, "
                f"got {self.skipkv_max_tracked_sentences}."
            )
        self.skipkv_enable_activation_steering = bool(self.skipkv_enable_activation_steering)
        self.skipkv_steering_layer = int(self.skipkv_steering_layer)
        self.skipkv_steering_alpha = float(self.skipkv_steering_alpha)
        self.skipkv_steering_alpha_increment = float(self.skipkv_steering_alpha_increment)
        self.skipkv_steering_alpha_max = float(self.skipkv_steering_alpha_max)
        if self.skipkv_enable_activation_steering and not self.skipkv_steering_vector_path:
            raise ValueError(
                "skipkv_enable_activation_steering=True requires skipkv_steering_vector_path. "
                "Official SkipKV support is limited to the released steering vectors for "
                f"{', '.join(sorted(SUPPORTED_SKIPKV_MODEL_NAMES))}."
            )
        self.prefix_cache_block_size = resolve_prefix_cache_block_size(self)

        # Normalize compressor type strings.
        for attr in ("compressor_down_type", "compressor_up_type"):
            v = getattr(self, attr, "auto")
            if v is None:
                v = "auto"
            v = str(v).strip().lower()
            setattr(self, attr, v if v else "auto")

        if self.vllm_sparse_method == "deltakv":
            log_once(
                "DeltaKV support in Sparse-vLLM is still experimental and not fully mature; "
                "verify results carefully before treating them as final.",
                level="WARNING",
            )
            if not bool(getattr(self, "use_compression", True)):
                raise ValueError("DeltaKV runtime is compressor-only; set use_compression=True.")
            if bool(getattr(self, "enable_sparse_ref_fp8", False)):
                raise ValueError("enable_sparse_ref_fp8 was removed from the slim DeltaKV runtime.")
            if self.deltakv_path is None and not self.allow_missing_deltakv_path:
                raise ValueError(
                    "DeltaKV requires deltakv_path for compressor sparse layers. "
                    "Set allow_missing_deltakv_path=True only for construction-only tests."
                )
            if self.kv_quant_bits not in (0, 4):
                raise ValueError(
                    "DeltaKV slim runtime supports sparse compressor residual bits 0 or 4 only, "
                    f"got kv_quant_bits={self.kv_quant_bits}."
                )
            if self.full_layer_kv_quant_bits not in (0, 4):
                raise ValueError(
                    "DeltaKV slim runtime supports full-layer storage bits 0 or 4 only, "
                    f"got full_layer_kv_quant_bits={self.full_layer_kv_quant_bits}."
                )
            if self.kv_quant_bits == 4 and self.kv_quant_group_size == 0:
                self.kv_quant_group_size = 32
            is_bf16_full_compressor_sparse = (
                self.full_layer_kv_quant_bits == 0 and self.kv_quant_bits == 0
            )
            is_kivi4_full_int4_compressor_sparse = (
                self.full_layer_kv_quant_bits == 4
                and self.kv_quant_bits == 4
                and bool(getattr(self, "enable_full_layer_kivi_quant", True))
            )
            if not (is_bf16_full_compressor_sparse or is_kivi4_full_int4_compressor_sparse):
                raise ValueError(
                    "DeltaKV slim runtime supports exactly two paths: "
                    "(full_layer_kv_quant_bits=0, kv_quant_bits=0) and "
                    "(full_layer_kv_quant_bits=4, kv_quant_bits=4, enable_full_layer_kivi_quant=True)."
                )

        if self.obs_layer_ids is None:
            self.obs_layer_ids = []
            for l in self.full_attn_layers:
                if (l + 1) not in self.full_attn_layers:
                    self.obs_layer_ids.append(l)
        
        # 确保调度吞吐量限制不小于单次分块大小
        if self.max_num_batched_tokens < 2 * self.chunk_prefill_size:
            self.max_num_batched_tokens = 2 * self.chunk_prefill_size

        # PyramidKV 配置验证与智能生成
        if 'pyramidkv' == self.vllm_sparse_method:
            if self.pyramid_layer_ratios is None:
                num_layers = self.hf_config.num_hidden_layers
                start_l = self.pyramidkv_start_layer
                least_l = self.pyramidkv_least_layer if self.pyramidkv_least_layer is not None else num_layers - 1
                start_r = self.pyramidkv_start_ratio
                least_r = self.pyramidkv_least_ratio
                
                ratios = [1.0] * num_layers
                for i in range(start_l, num_layers):
                    if i <= least_l:
                        if least_l > start_l:
                            ratio = start_r - (start_r - least_r) * (i - start_l) / (least_l - start_l)
                        else:
                            ratio = least_r
                        ratios[i] = ratio
                    else:
                        ratios[i] = least_r
                self.pyramid_layer_ratios = ratios
                logger.info(f"PyramidKV 自动生成 layer_ratios = {[f'{r:.3f}' for r in self.pyramid_layer_ratios]}")
        
        if self.pyramid_layer_ratios is not None:
            # PyramidKV 模式自动启用 SnapKV 逻辑
            if 'pyramidkv' != self.vllm_sparse_method:
                raise ValueError('vllm_sparse_method 应为 pyramidkv')

            num_layers = self.hf_config.num_hidden_layers
            if len(self.pyramid_layer_ratios) != num_layers:
                raise ValueError(f"pyramid_layer_ratios 长度 ({len(self.pyramid_layer_ratios)}) 必须等于模型层数 ({num_layers})")

            if any(r <= 0 or r > 1.0 for r in self.pyramid_layer_ratios):
                raise ValueError("pyramid_layer_ratios 的所有值必须在 (0, 1.0] 范围内")
        
        logger.info(f"LLM Config: {self}".replace('\n', ' '))
