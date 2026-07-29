"""Top-level configuration composition and initialization orchestration."""

from dataclasses import dataclass, field
from typing import Any, Union

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
from sparsevllm.configs.groups import (
    DecodeCudaGraphConfig,
    DeltaKVConfig,
    ObservabilityConfig,
    PrefixCacheConfig,
    SparseMethodConfig,
)
from sparsevllm.configs.model import (
    AutoConfig,
    Qwen3Config,
    QuantizationConfig,
    RuntimeLayout,
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
from sparsevllm.method_registry import PREFILL_POLICY_AUTO
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
    max_model_len: int = 128_000
    max_decoding_seqs: int = 64
    max_num_seqs_in_gpu: int | None = None

    chunk_prefill_size: int | None = None
    long_prefill_offload_threshold: int = 96 * 1024
    mlp_chunk_size: int = 16384
    prefill_schedule_policy: str = PREFILL_POLICY_AUTO
    gpu_memory_utilization: float = 0.8
    device_memory_utilization: float | None = None
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    data_parallel_size: int = 1
    # Soft host-side I/O budget shared across ranks; every rank retains at
    # least one synchronous loading path when the budget is smaller.
    weight_loading_workers: int = 1
    enforce_eager: bool = True
    hf_config: Union[Qwen3Config, AutoConfig] | None = None
    outer_hf_config: Any | None = None
    runtime_layout: RuntimeLayout | None = None
    quantization_config: QuantizationConfig = field(default_factory=QuantizationConfig.disabled)
    tiny_random: bool = False
    tiny_random_config: str | None = None
    tiny_random_seed: int = 0
    tiny_random_overrides: dict[str, int] = field(default_factory=dict, init=False)
    eos: int = -1
    eos_token_ids: tuple[int, ...] = field(default_factory=tuple)
    num_kvcache_slots: int | list = -1

    @property
    def uses_outer_tp_moe_layout(self) -> bool:
        model_type = str(getattr(self.hf_config, "model_type", "") or "")
        return model_type in {"qwen3_moe", "minimax_m2"} and int(
            self.tensor_parallel_size
        ) > 1

    @property
    def attention_tensor_parallel_size(self) -> int:
        return int(self.tensor_parallel_size)

    @property
    def moe_expert_parallel_size(self) -> int:
        return int(self.expert_parallel_size)

    @property
    def moe_tensor_parallel_size(self) -> int:
        if self.uses_outer_tp_moe_layout:
            return int(self.tensor_parallel_size) // int(self.expert_parallel_size)
        return int(self.tensor_parallel_size)

    @property
    def world_size(self) -> int:
        if self.uses_outer_tp_moe_layout:
            return int(self.tensor_parallel_size) * int(self.data_parallel_size)
        return (
            int(self.tensor_parallel_size)
            * int(self.expert_parallel_size)
            * int(self.data_parallel_size)
        )

    @property
    def weight_loading_workers_per_rank(self) -> int:
        return max(1, self.weight_loading_workers // self.world_size)

    def __post_init__(self):
        normalize_bootstrap(self)
        legacy_deltakv_graph_method = normalize_sparse_method_name(self)
        normalize_prefix_cache(self)
        normalize_scheduling(self)
        normalize_deltakv_storage(self)
        normalize_platform(self)
        normalize_decode_cuda_graph(
            self,
            legacy_deltakv_graph_method=legacy_deltakv_graph_method,
        )
        is_qwen35 = load_and_validate_model(self)
        normalize_sparse_methods(self)
        finalize_prefix_cache(self, is_qwen35=is_qwen35)
        validate_deltakv_runtime(self, is_qwen35=is_qwen35)
        finalize_sparse_layout(self)

        logger.info(f"LLM Config: {self}".replace('\n', ' '))
        setattr(self.hf_config, "runtime_layout", self.runtime_layout)
