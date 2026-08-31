"""Top-level configuration composition and initialization orchestration."""

from dataclasses import dataclass, field
from typing import Any

from sparsevllm.configs.bootstrap import normalize_bootstrap
from sparsevllm.configs.cuda_graph import (
    _resolve_decode_cuda_graph_capture_sizes,
    _resolve_decode_cuda_graph_context_sizes,
    normalize_decode_cuda_graph,
)
from sparsevllm.configs.delta import (
    normalize_deltakv_storage,
    validate_deltakv_runtime,
)
from sparsevllm.configs.full_attention_profiles import (
    resolve_auto_full_attention_layers,
)
from sparsevllm.configs.groups import (
    DecodeCudaGraphConfig,
    DeltaKVConfig,
    ObservabilityConfig,
    PrefixCacheConfig,
    SparseMethodConfig,
)
from sparsevllm.configs.model import (
    AutoConfig,
    load_and_validate_model,
)
from sparsevllm.configs.platform import normalize_platform
from sparsevllm.configs.prefix_cache import (
    finalize_prefix_cache,
    normalize_prefix_cache,
)
from sparsevllm.configs.scheduling import normalize_scheduling
from sparsevllm.configs.sparse import (
    finalize_sparse_layout,
    normalize_sparse_method_name,
    normalize_sparse_methods,
)
from sparsevllm.distributed import ParallelTopology
from sparsevllm.method_registry import PREFILL_POLICY_AUTO
from sparsevllm.models.layout import RuntimeLayout
from sparsevllm.models.spec import ModelSpec
from sparsevllm.quantization import QuantizationConfig
from sparsevllm.utils.log import logger


@dataclass
class Config(
    PrefixCacheConfig,
    DecodeCudaGraphConfig,
    SparseMethodConfig,
    DeltaKVConfig,
    ObservabilityConfig,
):
    model: str
    max_num_batched_tokens: int = 65536
    max_num_seqs_in_batch: int = 32  # 不能设置太大
    max_model_len: int | None = None
    max_model_len_auto: bool = field(default=False, init=False)
    max_decoding_seqs: int = 64
    max_num_seqs_in_gpu: int | None = None

    engine_prefill_chunk_size: int | None = None
    long_prefill_offload_threshold: int = 64 * 1024
    mlp_chunk_size: int = 16384
    mla_prefill_workspace_bytes: int = 2 * 1024**3
    prefill_schedule_policy: str = PREFILL_POLICY_AUTO
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    data_parallel_size: int = 1
    # Soft host-side I/O budget shared across ranks; every rank retains at
    # least one synchronous loading path when the budget is smaller.
    weight_loading_workers: int = 1
    enable_multimodal: bool = True
    hf_config: AutoConfig | None = None
    outer_hf_config: Any | None = None
    runtime_layout: RuntimeLayout | None = None
    attention_cache_layout: str = field(default="explicit_kv", init=False)
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig.disabled)
    model_spec: ModelSpec = field(init=False, repr=False)
    parallel_topology: ParallelTopology = field(init=False, repr=False)
    tiny_random: bool = False
    tiny_random_config: str | None = None
    tiny_random_seed: int = 0
    tiny_random_overrides: dict[str, int] = field(default_factory=dict, init=False)
    eos: int = -1
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)
    num_kvcache_slots: int | list = -1

    @property
    def uses_outer_tp_moe_layout(self) -> bool:
        return self.parallel_topology.is_outer_tp_moe

    @property
    def attention_tensor_parallel_size(self) -> int:
        return self.parallel_topology.attention_tp_size

    @property
    def moe_expert_parallel_size(self) -> int:
        return int(self.expert_parallel_size)

    @property
    def moe_tensor_parallel_size(self) -> int:
        return self.parallel_topology.moe_tp_size

    @property
    def world_size(self) -> int:
        return self.parallel_topology.world_size

    @property
    def weight_loading_workers_per_rank(self) -> int:
        return max(1, self.weight_loading_workers // self.world_size)

    def limit_auto_max_model_len(self, capacity: int) -> None:
        if not self.max_model_len_auto:
            return
        resolved = min(int(self.max_model_len), int(capacity))
        if resolved <= 0:
            raise RuntimeError(f"Runtime capacity must be positive, got {capacity}.")
        if resolved == self.max_model_len:
            return
        log = (
            logger.debug
            if getattr(self, "startup_cache_phase", "production") == "profiling"
            else logger.info
        )
        log(
            "Limiting auto max_model_len from model capacity {} to runtime capacity {}.",
            self.max_model_len,
            resolved,
        )
        self.max_model_len = resolved
        if self.decode_graph and self.decode_graph_context_sizes_auto:
            self.decode_graph_context_sizes = _resolve_decode_cuda_graph_context_sizes(
                "auto", resolved
            )

    def __post_init__(self):
        normalize_bootstrap(self)
        normalize_sparse_method_name(self)
        normalize_prefix_cache(self)
        normalize_scheduling(self)
        normalize_deltakv_storage(self)
        normalize_platform(self)
        load_and_validate_model(self)
        normalize_decode_cuda_graph(self)
        normalize_sparse_methods(self)
        finalize_prefix_cache(self)
        validate_deltakv_runtime(self)
        resolve_auto_full_attention_layers(self)
        finalize_sparse_layout(self)

        logger.info(
            "Runtime config: model={} sparse_method={} tp={} ep={} dp={} "
            "max_model_len={} max_batched_tokens={} prefill_chunk={} "
            "max_prefill_batch={} max_decode_batch={} gpu_utilization={:.3f} "
            "decode_graph={} graph_policy={}.",
            self.model,
            self.sparse_method or "vanilla",
            self.tensor_parallel_size,
            self.expert_parallel_size,
            self.data_parallel_size,
            self.max_model_len,
            self.max_num_batched_tokens,
            self.engine_prefill_chunk_size,
            self.max_num_seqs_in_batch,
            self.max_decoding_seqs,
            self.gpu_memory_utilization,
            self.decode_graph,
            self.decode_graph_shape_policy,
        )
        logger.debug("Full runtime config: {}", str(self).replace("\n", " "))
        setattr(self.hf_config, "runtime_layout", self.runtime_layout)
