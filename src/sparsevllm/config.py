import os
from dataclasses import dataclass
from transformers import AutoConfig, Qwen3Config
from typing import Union
from sparsevllm.method_registry import (
    PREFILL_POLICY_AUTO,
    SUPPORTED_SPARSE_METHODS,
    normalize_sparse_method,
    resolve_prefill_schedule_policy,
)
from sparsevllm.utils.log import logger


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
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    hf_config: Union[Qwen3Config, AutoConfig] | None = None
    eos: int = -1
    num_kvcache_slots: int | list = -1

    # Sparse Attention Config
    vllm_sparse_method: str = ""  # "", "streamingllm", "attention-sink", "attention_sink", "snapkv", "omnikv", "quest", "deltakv", "deltakv-triton", "deltakv-triton-v2", "deltakv-triton-v3", "deltakv-triton-v4", "deltakv-delta-quant", "deltakv_delta_quant", "deltakv-standalone", "deltakv-snapkv", "pyramidkv"

    # General Sparse Config
    num_sink_tokens: int = 64
    num_recent_tokens: int = 512
    num_top_tokens: int = 4096

    # OmniKV Config
    obs_layer_ids: list[int] = None  # None means auto-calculate based on full_attn_layers (useful for omnikv)
    full_attn_layers: str | list[int] = "0" # useful for omnikv
    chunk_prefill_accel_omnikv: bool = False
    num_top_tokens_in_prefill: int | None = 8192
    decode_cuda_graph: bool = False
    decode_cuda_graph_capture_sampling: bool = False
    decode_cuda_graph_capture_sizes: str | int | list[int] | tuple[int, ...] | None = "auto"
    omnikv_decode_cuda_graph: bool = False

    # QuEST Config
    quest_chunk_size: int = 16
    quest_token_budget: int = 1024
    quest_skip_layers: int = 2

    # SnapKV Config
    snapkv_window_size: int = 32
    snapkv_num_full_layers: int = 0  # 前多少层不进行驱逐
    
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
    kv_compressed_size: int = 128
    kv_quant_bits: int = 4
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
    # Triton kernels: group multiple KV heads per program to reduce redundant loads.
    deltakv_triton_gather_heads_per_program: int = 4
    deltakv_triton_reconstruct_heads_per_program: int = 4
    deltakv_cluster_gather_chunk_size: int = 16384
    
    enable_profiler: bool = False
    throughput_log_interval_s: float = 10.0
    allow_missing_deltakv_path: bool = False
    allow_unknown_config_keys: bool = False


    def __post_init__(self):
        if os.getenv("PROFILER_SVLLM"):
            self.enable_profiler = True
            
        self.vllm_sparse_method = normalize_sparse_method(self.vllm_sparse_method)
        if self.vllm_sparse_method not in SUPPORTED_SPARSE_METHODS:
            supported = ", ".join(repr(method) for method in sorted(SUPPORTED_SPARSE_METHODS) if method)
            raise ValueError(
                f"Unsupported vllm_sparse_method={self.vllm_sparse_method!r}. "
                f"Supported methods: '', {supported}."
            )
        self.prefill_schedule_policy = resolve_prefill_schedule_policy(
            self.vllm_sparse_method,
            self.prefill_schedule_policy,
        )
        
        if self.num_top_tokens_in_prefill is None:
            self.num_top_tokens_in_prefill = self.num_top_tokens
        if int(self.mlp_chunk_size) <= 0:
            raise ValueError(f"mlp_chunk_size must be > 0, got {self.mlp_chunk_size}.")
        self.mlp_chunk_size = int(self.mlp_chunk_size)
        if int(self.deltakv_cluster_gather_chunk_size) <= 0:
            raise ValueError(
                "deltakv_cluster_gather_chunk_size must be > 0, "
                f"got {self.deltakv_cluster_gather_chunk_size}."
            )
        self.deltakv_cluster_gather_chunk_size = int(self.deltakv_cluster_gather_chunk_size)

        if not os.path.isdir(self.model):
            raise FileNotFoundError(f"Model directory does not exist: {self.model}")
        if int(self.max_decoding_seqs) <= 0:
            raise ValueError(f"max_decoding_seqs must be > 0, got {self.max_decoding_seqs}.")
        self.max_decoding_seqs = int(self.max_decoding_seqs)
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError(f"tensor_parallel_size must be in [1, 8], got {self.tensor_parallel_size}.")
        self.decode_cuda_graph = bool(self.decode_cuda_graph)
        self.decode_cuda_graph_capture_sampling = bool(self.decode_cuda_graph_capture_sampling)
        self.omnikv_decode_cuda_graph = bool(self.omnikv_decode_cuda_graph)
        if self.omnikv_decode_cuda_graph:
            if self.vllm_sparse_method != "omnikv":
                raise ValueError(
                    "omnikv_decode_cuda_graph is only valid with vllm_sparse_method='omnikv'."
                )
            self.decode_cuda_graph = True
        if self.decode_cuda_graph_capture_sampling and not self.decode_cuda_graph:
            raise ValueError("decode_cuda_graph_capture_sampling requires decode_cuda_graph=True.")
        if self.decode_cuda_graph:
            if self.vllm_sparse_method not in {"", "omnikv"}:
                raise ValueError("decode_cuda_graph currently supports vanilla and omnikv only.")
            if self.tensor_parallel_size != 1:
                raise ValueError("decode_cuda_graph currently supports tensor_parallel_size=1 only.")
            self.decode_cuda_graph_capture_sizes = _resolve_decode_cuda_graph_capture_sizes(
                self.decode_cuda_graph_capture_sizes,
                self.max_decoding_seqs,
            )
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
                "Supported model types: qwen2, qwen3."
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

        # Normalize compressor type strings.
        for attr in ("compressor_down_type", "compressor_up_type"):
            v = getattr(self, attr, "auto")
            if v is None:
                v = "auto"
            v = str(v).strip().lower()
            setattr(self, attr, v if v else "auto")

        if (
            isinstance(self.vllm_sparse_method, str)
            and self.vllm_sparse_method.startswith("deltakv")
            and self.vllm_sparse_method not in {
                "deltakv-standalone",
                "deltakv-snapkv",
                "deltakv-delta-quant",
                "deltakv_delta_quant",
            }
            and self.deltakv_path is None
            and not self.allow_missing_deltakv_path
        ):
            raise ValueError(
                "DeltaKV compressor mode requires deltakv_path. Pass deltakv_path explicitly, "
                "use a no-checkpoint method such as deltakv-standalone/deltakv-snapkv/"
                "deltakv-delta-quant, or set allow_missing_deltakv_path=True only for an "
                "explicitly validated ablation."
            )

        if self.vllm_sparse_method in ("deltakv-standalone", "deltakv-snapkv"):
            # Standalone DeltaKV uses all layers uniformly and does not rely on
            # OmniKV-style observation/full-layer routing.
            self.full_attn_layers = []
            self.obs_layer_ids = []
        elif self.obs_layer_ids is None:
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
