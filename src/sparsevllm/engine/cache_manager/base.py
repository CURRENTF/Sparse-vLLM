from __future__ import annotations

import os
import hashlib
from collections import deque
from dataclasses import dataclass, fields, is_dataclass
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist

from sparsevllm.config import Config
from sparsevllm.distributed import ParallelContext
from sparsevllm.engine.sequence import Sequence
from sparsevllm.method_registry import SUPPORTED_SPARSE_METHODS, normalize_sparse_method
from sparsevllm.triton_kernel.store_kvcache import store_kvcache
import sparsevllm.platforms as platforms
from sparsevllm.utils.log import logger, log_level


def _debug_tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    detached = tensor.detach().contiguous().cpu()
    raw = detached.reshape(-1).view(torch.uint8).numpy().tobytes()
    summary = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "numel": int(detached.numel()),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    if detached.numel() > 0 and not detached.is_complex():
        numeric = detached.to(torch.float64)
        summary.update(
            {
                "min": float(numeric.min().item()),
                "max": float(numeric.max().item()),
                "sum": float(numeric.sum().item()),
            }
        )
    return summary


def _debug_value_summary(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return _debug_tensor_summary(value)
    if is_dataclass(value):
        return {
            field.name: _debug_value_summary(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, dict):
        return {
            str(key): _debug_value_summary(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_debug_value_summary(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "tolist"):
        return _debug_value_summary(value.tolist())
    return repr(value)


@dataclass
class LayerBatchStates:
    """存储当前 Batch 在特定层的前向计算状态。

    仅包含与物理存储和基本前向元数据相关的字段。
    """

    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    max_context_len: int | None = None
    req_indices: torch.Tensor | None = None


@dataclass(frozen=True)
class JointPrefixCapacity:
    block_capacity: int
    kv_allocatable_bytes: int
    recurrent_capacity_bytes: int
    unallocated_bytes: int


def resolve_joint_prefix_capacity(
    *,
    available_bytes: int,
    kv_bytes_per_block: int,
    recurrent_bytes_per_block: int,
    requested_max_blocks: int | None,
) -> JointPrefixCapacity:
    available_bytes = int(available_bytes)
    kv_bytes_per_block = int(kv_bytes_per_block)
    recurrent_bytes_per_block = int(recurrent_bytes_per_block)
    if available_bytes < 0:
        raise ValueError(f"available_bytes must be >= 0, got {available_bytes}.")
    if kv_bytes_per_block <= 0 or recurrent_bytes_per_block <= 0:
        raise ValueError(
            "Joint prefix block bytes must be positive: "
            f"kv={kv_bytes_per_block} recurrent={recurrent_bytes_per_block}."
        )
    block_capacity = available_bytes // (kv_bytes_per_block + recurrent_bytes_per_block)
    if requested_max_blocks is not None:
        requested_max_blocks = int(requested_max_blocks)
        if requested_max_blocks <= 0:
            raise ValueError(
                f"requested_max_blocks must be positive when set, got {requested_max_blocks}."
            )
        block_capacity = min(block_capacity, requested_max_blocks)
    kv_allocatable_bytes = block_capacity * kv_bytes_per_block
    recurrent_capacity_bytes = block_capacity * recurrent_bytes_per_block
    return JointPrefixCapacity(
        block_capacity=int(block_capacity),
        kv_allocatable_bytes=int(kv_allocatable_bytes),
        recurrent_capacity_bytes=int(recurrent_capacity_bytes),
        unallocated_bytes=int(
            available_bytes - kv_allocatable_bytes - recurrent_capacity_bytes
        ),
    )


@dataclass
class SparseSelection:
    """Logical token selection produced by SparseController for one layer."""

    kind: str
    req_indices: torch.Tensor
    context_lens: torch.Tensor
    max_context_len: int | None = None
    attn_score: torch.Tensor | None = None
    active_indices: torch.Tensor | None = None
    active_slots: torch.Tensor | None = None
    active_compressed_indices: torch.Tensor | None = None
    global_req_indices: torch.Tensor | None = None
    chunk_lens: torch.Tensor | None = None
    release_temp_slots: bool = False


@dataclass
class DecodeComputeView:
    """Physical KV/view tensors consumed by decode attention kernels."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    active_slots: torch.Tensor
    req_indices: torch.Tensor
    context_lens: torch.Tensor
    attn_score: torch.Tensor | None = None
    max_context_len: int | None = None
    temp_slots: torch.Tensor | None = None
    backend: str = "dense"
    metadata: dict[str, Any] | None = None


@dataclass
class PrefillComputeView:
    """Physical KV/view tensors consumed by prefill attention kernels."""

    k_cache: torch.Tensor
    v_cache: torch.Tensor
    active_slots: torch.Tensor
    req_indices: torch.Tensor
    context_lens: torch.Tensor
    attn_score: torch.Tensor | None = None
    max_context_len: int | None = None
    temp_slots: torch.Tensor | None = None


class CacheManager(ABC):
    """每个 Rank 只有一个 CacheManager，内部管理所有层的物理槽位和 KV Cache。"""

    def __init__(self, config: Config, parallel_context: ParallelContext):
        self.config = config
        self.parallel_context = parallel_context
        self.rank = parallel_context.world_rank
        self.world_size = parallel_context.world_size
        self.tp_rank = parallel_context.tp_rank
        self.tp_size = parallel_context.tp_size
        self.ep_rank = parallel_context.ep_rank
        self.ep_size = parallel_context.ep_size
        self.dp_rank = parallel_context.dp_rank
        self.dp_size = parallel_context.dp_size
        self.platform = platforms.current_platform
        self.device = self.platform.get_device(self.rank)
        self.hf_config = config.hf_config
        self.num_layers = self.hf_config.num_hidden_layers
        self.runtime_layout = getattr(config, "runtime_layout", None)
        if self.runtime_layout is None:
            from sparsevllm.config import RuntimeLayout

            self.runtime_layout = RuntimeLayout.dense(self.num_layers)
        self.num_kv_layers = int(self.runtime_layout.num_kv_layers)

        self.num_kv_heads = self.hf_config.num_key_value_heads // self.tp_size
        self.head_dim = getattr(
            self.hf_config,
            "head_dim",
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
        )

        self.max_model_len = config.max_model_len
        resident_buffer_rows = int(config.max_num_seqs_in_gpu)
        recurrent_row_capacity = getattr(
            config,
            "recurrent_state_row_capacity",
            None,
        )
        has_recurrent_layers = bool(
            getattr(self.runtime_layout, "linear_attention_layer_indices", ())
        )
        if has_recurrent_layers and recurrent_row_capacity is not None:
            recurrent_row_capacity = int(recurrent_row_capacity)
            if recurrent_row_capacity != resident_buffer_rows:
                raise RuntimeError(
                    "Cache-manager and recurrent live-row capacities disagree: "
                    f"cache_rows={resident_buffer_rows} "
                    f"recurrent_rows={recurrent_row_capacity}."
                )
            self.max_buffer_rows = recurrent_row_capacity
        else:
            self.max_buffer_rows = resident_buffer_rows

        self.kv_cache = None
        self._decode_static_max_context_len: int | None = None

    def synchronize_prefix_cache_delete_plan(
        self,
        local_plan: dict[str, object],
    ) -> None:
        if int(getattr(self, "world_size", 1)) <= 1:
            return
        plans: list[dict[str, object] | None] = [None] * self.world_size
        dist.all_gather_object(
            plans,
            local_plan,
            group=self.parallel_context.world.process_group,
        )
        if any(plan != plans[0] for plan in plans[1:]):
            raise RuntimeError(
                "Prefix-cache subtree deletion plan diverged across world ranks: "
                f"plans={plans!r}."
            )

    def is_full_attention_layer(self, layer_idx: int) -> bool:
        return bool(self.runtime_layout.is_full_attention(int(layer_idx)))

    def is_linear_attention_layer(self, layer_idx: int) -> bool:
        return bool(self.runtime_layout.is_linear_attention(int(layer_idx)))

    def kv_layer_index(self, layer_idx: int) -> int:
        return int(self.runtime_layout.kv_layer_index(int(layer_idx)))

    def kv_transformer_layer_indices(self) -> tuple[int, ...]:
        return tuple(int(layer_idx) for layer_idx in self.runtime_layout.kv_idx_to_layer_idx)

    def _is_stream_capturing(self) -> bool:
        platform = getattr(self, "platform", None)
        if platform is not None:
            return platform.is_stream_capturing()
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())

    @staticmethod
    def create(config: Config, parallel_context: ParallelContext) -> "CacheManager":
        sparse_method = normalize_sparse_method(config.vllm_sparse_method)
        model_type = getattr(getattr(config, "hf_config", None), "model_type", "") or ""

        if model_type in {"deepseek_v2", "deepseek_v32"}:
            raise NotImplementedError(f"Unsupported Sparse-vLLM model_type={model_type!r}.")
        if sparse_method not in SUPPORTED_SPARSE_METHODS:
            raise ValueError(f"Unsupported vllm_sparse_method={sparse_method!r}.")
        if sparse_method == "deltakv":
            from .deltakv_runtime import DeltaKVCacheManager

            return DeltaKVCacheManager(config, parallel_context)
        if sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            from .streamingllm import StreamingLLMCacheManager

            return StreamingLLMCacheManager(config, parallel_context)
        if sparse_method in ("snapkv", "pyramidkv"):
            from .snapkv import SnapKVCacheManager

            return SnapKVCacheManager(config, parallel_context)
        if sparse_method == "rkv":
            from .rkv import RKVCacheManager

            return RKVCacheManager(config, parallel_context)
        if sparse_method == "skipkv":
            from .skipkv import SkipKVCacheManager

            return SkipKVCacheManager(config, parallel_context)
        if sparse_method == "quest":
            from .quest import QuestCacheManager

            return QuestCacheManager(config, parallel_context)
        if sparse_method == "omnikv":
            from .omnikv import OmniKVCacheManager

            return OmniKVCacheManager(config, parallel_context)

        from .standard import StandardCacheManager

        return StandardCacheManager(config, parallel_context)

    def _get_available_slots_info(self) -> tuple[int, int]:
        """返回 (可用显存字节数, 每层每 token 的字节数)"""
        config = self.config
        hf_config = config.hf_config
        free, total = self.platform.get_available_memory(self.device.index or 0)

        # 动态估计 max_num_batched_tokens
        reserved_mem = total * (1 - config.gpu_memory_utilization)
        intermediate_size = getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4)
        # Dense MLP activations are sharded only by tensor parallelism.
        intermediate_size_per_rank = intermediate_size // self.tp_size
        dtype_size = torch.tensor([], dtype=hf_config.torch_dtype).element_size()

        # Keep this heuristic conservative: large prefill batches can still peak on
        # MLP/linear-attention projections and allocator fragmentation after KV
        # cache allocation.
        estimated_max_tokens = int(reserved_mem / (intermediate_size_per_rank * dtype_size * 16))
        allow_large_prefill_chunk = os.getenv("SPARSEVLLM_ALLOW_LARGE_PREFILL_CHUNK", "0") == "1"
        prefill_policy = getattr(config, "prefill_schedule_policy", None)
        if estimated_max_tokens <= 0:
            raise RuntimeError(
                "Estimated prefill token capacity must be positive: "
                f"estimated_max_tokens={estimated_max_tokens}."
            )
        chunk_prefill_size = int(config.chunk_prefill_size)
        if prefill_policy == "long_bs1full_short_batch":
            long_prefill_offload_threshold = int(config.long_prefill_offload_threshold)
            assert long_prefill_offload_threshold == chunk_prefill_size, (
                "long_bs1full_short_batch requires long_prefill_offload_threshold "
                "to equal chunk_prefill_size: "
                f"long_prefill_offload_threshold={long_prefill_offload_threshold}, "
                f"chunk_prefill_size={chunk_prefill_size}."
            )
        if (
            prefill_policy == "long_bs1full_short_batch"
            and chunk_prefill_size > estimated_max_tokens
        ):
            msg = (
                f"chunk_prefill_size={chunk_prefill_size} > "
                f"estimated_max_tokens={estimated_max_tokens} "
                f"(prefill_schedule_policy={prefill_policy!r})"
            )
            if allow_large_prefill_chunk:
                logger.warning(
                    "{}; continuing because SPARSEVLLM_ALLOW_LARGE_PREFILL_CHUNK=1. "
                    "This is an explicit experiment override and may OOM.",
                    msg,
                )
            else:
                logger.warning(
                    "{}; capping long_prefill_offload_threshold and "
                    "chunk_prefill_size to {} to avoid OOM.",
                    msg,
                    estimated_max_tokens,
                )
                config.long_prefill_offload_threshold = estimated_max_tokens
                config.chunk_prefill_size = estimated_max_tokens

        if estimated_max_tokens < config.max_num_batched_tokens and not allow_large_prefill_chunk:
            logger.warning(
                f"Estimated max_num_batched_tokens ({estimated_max_tokens}) is smaller than config "
                f"({config.max_num_batched_tokens}). Updating to avoid OOM."
            )
            config.max_num_batched_tokens = estimated_max_tokens
        elif estimated_max_tokens < config.max_num_batched_tokens:
            logger.warning(
                "Keeping max_num_batched_tokens={} above estimated {} because "
                "SPARSEVLLM_ALLOW_LARGE_PREFILL_CHUNK=1.",
                config.max_num_batched_tokens,
                estimated_max_tokens,
            )

        logger.info(f"Set dynamically max_num_batched_tokens = {config.max_num_batched_tokens}")

        used = total - free
        allocator_stats = self.platform.get_allocator_stats(self.device)
        peak = allocator_stats.peak_allocated_bytes
        current = allocator_stats.current_allocated_bytes

        target_persistent_bytes = int(total * config.gpu_memory_utilization)
        activation_reserve_bytes = int(total - target_persistent_bytes)
        recurrent_pool_bytes = int(getattr(config, "recurrent_state_pool_bytes", 0) or 0)
        recurrent_peak_before = int(
            getattr(config, "recurrent_state_allocator_peak_before_bytes", peak) or 0
        )
        recurrent_peak_after = int(
            getattr(config, "recurrent_state_allocator_peak_after_bytes", peak) or 0
        )
        recurrent_peak_growth = min(
            recurrent_pool_bytes,
            max(0, recurrent_peak_after - recurrent_peak_before),
        )
        recurrent_explicit_deduction = max(
            0,
            recurrent_pool_bytes - recurrent_peak_growth,
        )
        available_memory = int(
            target_persistent_bytes
            - used
            - peak
            + current
            - recurrent_explicit_deduction
        )
        slot_bytes_per_layer = 2 * self.num_kv_heads * self.head_dim * dtype_size

        recurrent_bytes_per_block = int(
            getattr(config, "prefix_recurrent_bytes_per_block", 0) or 0
        )
        prefix_block_capacity = 0
        prefix_recurrent_capacity_bytes = 0
        kv_bytes_per_block = 0
        if bool(getattr(config, "enable_prefix_caching", False)) and recurrent_bytes_per_block > 0:
            kv_bytes_per_block = self._kv_allocation_bytes_per_prefix_block(
                slot_bytes_per_layer
            )
            requested_max_blocks = getattr(
                config,
                "prefix_cache_requested_max_blocks",
                getattr(config, "prefix_cache_max_blocks", None),
            )
            config.prefix_cache_requested_max_blocks = requested_max_blocks
            joint_capacity = resolve_joint_prefix_capacity(
                available_bytes=available_memory,
                kv_bytes_per_block=kv_bytes_per_block,
                recurrent_bytes_per_block=recurrent_bytes_per_block,
                requested_max_blocks=requested_max_blocks,
            )
            if joint_capacity.block_capacity <= 0:
                raise RuntimeError(
                    "Insufficient GPU memory for one mixed prefix block: "
                    f"available_bytes={available_memory} "
                    f"kv_bytes_per_block={kv_bytes_per_block} "
                    f"recurrent_bytes_per_block={recurrent_bytes_per_block}."
                )
            available_memory = joint_capacity.kv_allocatable_bytes
            prefix_block_capacity = joint_capacity.block_capacity
            prefix_recurrent_capacity_bytes = joint_capacity.recurrent_capacity_bytes
            config.prefix_cache_max_blocks = int(prefix_block_capacity)
            config.prefix_recurrent_capacity_bytes = int(
                prefix_recurrent_capacity_bytes
            )
            config.prefix_kv_bytes_per_block = int(kv_bytes_per_block)
            config.prefix_kv_block_capacity = int(prefix_block_capacity)
            config.kv_allocatable_bytes = int(available_memory)

        model_current_bytes = max(0, int(current) - recurrent_pool_bytes)
        ratio = (
            0.0
            if kv_bytes_per_block <= 0
            else float(recurrent_bytes_per_block) / float(kv_bytes_per_block)
        )
        num_kv_layers = int(
            getattr(self, "num_kv_layers", getattr(self, "num_layers", 1))
        )
        kv_slots = (
            prefix_block_capacity * int(config.prefix_cache_block_size)
            if prefix_block_capacity > 0
            else available_memory // (num_kv_layers * slot_bytes_per_layer)
        )
        logger.info(
            "Persistent GPU budget: target={} model_current={} allocator_current={} "
            "device_used={} historical_peak={} recurrent_pool={} "
            "recurrent_peak_growth={} recurrent_explicit_deduction={} "
            "prefix_recurrent_capacity={} "
            "prefix_recurrent_to_kv_ratio={:.6f} activation_reserve={} "
            "kv_allocatable={} kv_slots={} prefix_blocks={}.",
            target_persistent_bytes,
            model_current_bytes,
            current,
            used,
            peak,
            recurrent_pool_bytes,
            recurrent_peak_growth,
            recurrent_explicit_deduction,
            prefix_recurrent_capacity_bytes,
            ratio,
            activation_reserve_bytes,
            available_memory,
            kv_slots,
            prefix_block_capacity,
        )

        if log_level == "DEBUG":
            logger.debug(
                f"[DEBUG] Available Memory: {available_memory / 1024**3:.2f} GB, "
                f"Slot Bytes Per Layer: {slot_bytes_per_layer / 1024**2:.4f} MB"
            )

        return available_memory, slot_bytes_per_layer

    def _kv_allocation_bytes_per_prefix_block(
        self,
        slot_bytes_per_layer: int,
    ) -> int:
        block_size = int(self.config.prefix_cache_block_size or 0)
        if block_size <= 0:
            raise RuntimeError(
                "Mixed prefix capacity requires a positive prefix_cache_block_size."
            )
        return int(block_size * self.num_kv_layers * int(slot_bytes_per_layer))

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        if is_prefill:
            return self._prepare_prefill(seqs)
        return self._prepare_decode(seqs)

    @abstractmethod
    def allocate_kv_cache(self):
        """自动计算并物理分配 KV Cache 张量"""
        raise NotImplementedError

    @abstractmethod
    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        raise NotImplementedError

    @abstractmethod
    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def get_layer_compute_tensors(self, layer_idx: int, selection: SparseSelection | None = None):
        raise NotImplementedError

    def get_layer_store_tensors(
        self,
        layer_idx: int,
        *,
        k_post_rope: torch.Tensor,
        v: torch.Tensor,
        pre_rope_k: torch.Tensor | None = None,
        pre_rope_v: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the tensors that should be written to the layer's physical KV cache."""
        del layer_idx, pre_rope_k, pre_rope_v
        return k_post_rope, v

    def _store_layer_kv(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        k_cache, v_cache, slot_mapping = self.get_layer_store_view(layer_idx)
        if slot_mapping is None:
            raise RuntimeError(f"KV store requires slot_mapping at layer={layer_idx}.")
        if int(slot_mapping.numel()) != int(k.shape[0]):
            raise RuntimeError(
                "KV store shape mismatch: "
                f"layer={layer_idx} k={tuple(k.shape)} v={tuple(v.shape)} "
                f"slot_mapping={tuple(slot_mapping.shape)}."
            )
        store_kvcache(k, v, k_cache, v_cache, slot_mapping)
        return slot_mapping

    def save_raw_kv_if_needed(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
    ):
        """Optional pre-norm/pre-RoPE KV storage point."""
        del layer_idx, k, v
        return None

    def save_rope_kv_if_needed(
        self,
        layer_idx: int,
        k_post_rope: torch.Tensor,
        v: torch.Tensor,
    ):
        """Store the post-RoPE KV representation used by ordinary cache layouts."""
        store_k, store_v = self.get_layer_store_tensors(
            layer_idx,
            k_post_rope=k_post_rope,
            v=v,
        )
        slot_mapping = self._store_layer_kv(layer_idx, store_k, store_v)
        self.on_kv_stored(
            layer_idx,
            store_k,
            slot_mapping,
        )

    def get_layer_compute_view(
        self,
        layer_idx: int,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        selection: SparseSelection | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return KV tensors and logical view used by attention kernels."""
        try:
            k_cache, v_cache = self.get_layer_compute_tensors(layer_idx, selection)
        except NotImplementedError:
            k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, active_slots, req_indices, context_lens

    def get_prefill_compute_view(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return KV tensors and logical view used by prefill attention kernels."""
        del k_current, v_current
        return self.get_layer_compute_view(
            layer_idx,
            active_slots,
            req_indices,
            context_lens,
            selection,
        )

    def _default_active_slots_for_selection(self, layer_idx: int, selection: SparseSelection) -> torch.Tensor:
        if selection.active_slots is not None:
            return selection.active_slots
        return self.get_layer_buffer_req_to_token_slots(layer_idx)

    def build_prefill_compute_view(
        self,
        layer_idx: int,
        k_current: torch.Tensor,
        v_current: torch.Tensor,
        selection: SparseSelection,
    ) -> PrefillComputeView:
        temp_slots = None
        if self.has_prefill_staging_view(layer_idx):
            active_slots, req_indices, context_lens, temp_slots = self.get_prefill_staging_view(layer_idx)
        elif self.has_full_layer_quantized_view(layer_idx):
            active_slots, req_indices, context_lens = self.build_full_layer_quantized_view(
                layer_idx,
                selection.req_indices,
                selection.context_lens,
            )
        else:
            active_slots = self._default_active_slots_for_selection(layer_idx, selection)
            req_indices = selection.req_indices
            context_lens = selection.context_lens
        k_cache, v_cache, active_slots, req_indices, context_lens = self.get_prefill_compute_view(
            layer_idx,
            k_current,
            v_current,
            selection,
            active_slots,
            req_indices,
            context_lens,
        )
        return PrefillComputeView(
            k_cache=k_cache,
            v_cache=v_cache,
            active_slots=active_slots,
            req_indices=req_indices,
            context_lens=context_lens,
            attn_score=selection.attn_score,
            max_context_len=selection.max_context_len,
            temp_slots=temp_slots,
        )

    def collect_prefill_attention_score(
        self,
        layer_idx: int,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
    ):
        """Optional method-owned prefill score collection after attention output is computed."""
        del layer_idx, q, view, b_start_loc, chunk_lens
        return None

    def record_prefill_query(
        self,
        layer_idx: int,
        q: torch.Tensor,
        view: PrefillComputeView,
        *,
        b_start_loc: torch.Tensor,
        chunk_lens: torch.Tensor,
    ):
        """Optional method-owned prefill query cache update after attention output is computed."""
        del layer_idx, q, view, b_start_loc, chunk_lens
        return None

    def before_prefill_layer_attention(self, layer_idx: int, selection: SparseSelection):
        """Optional hook immediately before building a prefill layer compute view."""
        del selection
        coordinator = getattr(self, "prefix_cache_coordinator", None)
        if coordinator is not None:
            coordinator.before_prefill_layer_attention(int(layer_idx))
        return None

    def defer_prefill_eviction(self) -> bool:
        """Whether the current method should skip chunk-end sparse eviction."""
        return False

    def record_decode_query(self, layer_idx: int, q: torch.Tensor):
        """Optional method-owned decode query cache update after attention output is computed."""
        del layer_idx, q
        return None

    def pop_prefill_attention_score(self, layer_idx: int, seq: Sequence) -> torch.Tensor | None:
        """Return and clear a method-owned prefill score for one completed sequence."""
        del layer_idx, seq
        return None

    @abstractmethod
    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def on_kv_stored(
        self,
        layer_idx: int,
        k: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        """Optional method-specific hook after KV has been written into cache."""
        return None

    def on_pre_rope_kv_stored(
        self,
        layer_idx: int,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        slot_mapping: torch.Tensor,
    ):
        """Optional hook for methods that need RoPE-independent KV metadata."""
        return None

    def on_layer_attention_end(self, layer_idx: int):
        """Optional method-specific hook after a layer's attention has consumed KV."""
        return None

    def release_layer_temp_slots(self, layer_idx: int, temp_slots: torch.Tensor | None):
        """Release temporary physical slots returned with a layer read/compute view."""
        del layer_idx, temp_slots
        return None

    def decode_cuda_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        """Cache-manager-owned tensors captured by decode CUDA graphs."""
        return []

    def decode_cuda_graph_max_cached_graphs(self) -> int | None:
        """Optional bound for captured decode graph states.

        Applies to every sparse method; individual managers may still override it.
        """
        value = getattr(self.config, "decode_cuda_graph_max_cached_graphs", None)
        return None if value is None else int(value)

    def select_decode_cuda_graph_batch_size(
        self,
        real_batch_size: int,
        capture_sizes: list[int],
    ) -> int | None:
        """Optional method-specific graph batch-size selection.

        Return None to use the runner's standard capture-size buckets.
        """
        del real_batch_size, capture_sizes
        return None

    def decode_cuda_graph_context_capacity(
        self,
        seqs: list[Sequence],
        *,
        requested_context_capacity: int,
        current_context_capacity: int,
    ) -> tuple[int, bool] | None:
        """Optional method-specific graph context-capacity policy.

        Returns (context_capacity, allow_larger_cached_capacity), or None to use
        the runner's default requested-capacity graph policy.
        """
        del seqs, requested_context_capacity, current_context_capacity
        return None

    def decode_cuda_graph_force_eager(self) -> bool:
        """Whether this method should bypass graph replay for diagnostics."""
        return False

    def on_forward_end(self, seqs: list[Sequence], is_prefill: bool):
        """Optional hook after all layers have stored KV for a forward step."""
        return None

    def prefix_cache_inspect(
        self,
        token_ids: list[int],
        *,
        include_subtree: bool = False,
    ) -> dict[str, object]:
        del token_ids, include_subtree
        raise RuntimeError("prefix cache is not enabled or not supported by this cache manager.")

    def prefix_cache_match(self, token_ids: list[int]) -> dict[str, object]:
        del token_ids
        return {
            "supported": False,
            "enabled": False,
            "matched_tokens": 0,
            "matched_blocks": 0,
            "match_ratio": 0.0,
            "reason": "prefix cache is not supported by this cache manager.",
        }

    def prefix_cache_delete_subtree(self, token_ids: list[int]) -> dict[str, object]:
        del token_ids
        raise RuntimeError("prefix cache is not enabled or not supported by this cache manager.")

    def prefix_cache_set_eviction_priority(
        self,
        token_ids: list[int],
        *,
        priority: int,
    ) -> dict[str, object]:
        del token_ids, priority
        raise RuntimeError("prefix cache is not enabled or not supported by this cache manager.")

    def has_prefill_staging_view(self, layer_idx: int) -> bool:
        """Whether the current prefill layer should read from a temporary staging KV view."""
        return False

    def get_prefill_staging_view(
        self,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Return (active_slots, req_indices, context_lens, temp_slots) for prefill staging."""
        raise NotImplementedError

    def has_full_layer_quantized_view(self, layer_idx: int) -> bool:
        """Whether a full-attention layer should read a reconstructed quantized KV view."""
        return False

    def build_full_layer_quantized_view(
        self,
        layer_idx: int,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (active_slots, local_req_indices, context_lens) for quantized full layers."""
        raise NotImplementedError

    def build_decode_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        active_slots: torch.Tensor,
        req_indices: torch.Tensor,
        context_lens: torch.Tensor,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Optional method-specific decode-time logical view builder."""
        return active_slots, req_indices, context_lens

    def build_decode_compute_view(
        self,
        layer_idx: int,
        q: torch.Tensor,
        selection: SparseSelection,
        *,
        num_heads: int,
        num_kv_heads: int,
    ) -> DecodeComputeView:
        if self.has_full_layer_quantized_view(layer_idx):
            active_slots, req_indices, context_lens = self.build_full_layer_quantized_view(
                layer_idx,
                selection.req_indices,
                selection.context_lens,
            )
        else:
            active_slots = self._default_active_slots_for_selection(layer_idx, selection)
            active_slots, req_indices, context_lens = self.build_decode_view(
                layer_idx,
                q,
                active_slots,
                selection.req_indices,
                selection.context_lens,
                num_heads=num_heads,
                num_kv_heads=num_kv_heads,
            )
        k_cache, v_cache, active_slots, req_indices, context_lens = self.get_layer_compute_view(
            layer_idx,
            active_slots,
            req_indices,
            context_lens,
            selection,
        )
        return DecodeComputeView(
            k_cache=k_cache,
            v_cache=v_cache,
            active_slots=active_slots,
            req_indices=req_indices,
            context_lens=context_lens,
            attn_score=selection.attn_score,
            max_context_len=selection.max_context_len,
        )

    def get_decode_block_seq(self, layer_idx: int, default: int) -> int:
        """Optional per-layer decode stage block size override."""
        return int(default)

    def set_decode_static_max_context_len(self, max_context_len: int):
        """Pin graph-captured decode kernels to a fixed max context length."""
        max_context_len = int(max_context_len)
        self._decode_static_max_context_len = max_context_len
        layer_batch_state = getattr(self, "layer_batch_state", None)
        if layer_batch_state is not None:
            layer_batch_state.max_context_len = max_context_len
        layer_batch_states = getattr(self, "layer_batch_states", None)
        if layer_batch_states is not None:
            for state in layer_batch_states:
                state.max_context_len = max_context_len
        for attr_name in ("full_layer_batch_states", "deltakv_layer_batch_states"):
            state = getattr(self, attr_name, None)
            if state is not None:
                state.max_context_len = max_context_len

    @property
    @abstractmethod
    def num_free_slots(self) -> int:
        raise NotImplementedError

    def num_free_slots_full_layers(self) -> int:
        """Free slots in the KV pool that is not subject to sparse eviction.

        Default behavior: treat `num_free_slots` as the only pool.
        DeltaKV overrides this to expose the full-attention pool capacity, which
        bounds how many long prompts can be admitted without thrashing.
        """
        return self.num_free_slots

    # ---- Scheduler hooks (default implementations) ----
    def prefill_batched_tokens_margin(self) -> int:
        """Extra headroom the scheduler should leave in `max_num_batched_tokens` for this cache manager."""
        return 0

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        """Effective remaining prefill tokens for scheduling decisions."""
        virtual_prefilled = max(
            int(seq.num_prefilled_tokens),
            int(getattr(seq, "prefix_cache_hit_len", 0) or 0),
        )
        return int(seq.num_prompt_tokens - virtual_prefilled)

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], chunk_prefill_size: int) -> int:
        """Persistent slots reserved by waiting/running prefills.

        This must not include temporary staging KV or decode reconstruction scratch.
        """
        reserved = 0
        for seq in waiting_seqs:
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                reserved += int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        return reserved

    def prefill_step_free_slots(self) -> int:
        """Writable KV capacity for the current prefill step.

        Temporary pools with a different lifetime should expose their own accounting
        instead of being mixed into this persistent step capacity.
        """
        return int(self.num_free_slots)

    def should_schedule_full_prefill(self, seq: Sequence) -> bool:
        """Whether scheduler should route this first prefill as a full bs1 step."""
        return False

    def requires_full_prefill_step(self, seq: Sequence) -> bool:
        """Whether this prefill candidate must run its remaining tokens in one step."""
        return False

    def is_full_prefill_step(self, seqs: list[Sequence]) -> bool:
        """Whether the current prepared prefill step is backed by a full-prefill staging view."""
        return False

    def prefill_step_free_slots_for(self, seq: Sequence) -> int:
        """Writable KV capacity for a specific prefill candidate."""
        return int(self.prefill_step_free_slots())

    def min_final_prefill_chunk_size(self, seq: Sequence) -> int:
        """Minimum final chunk size required by method-specific prefill logic."""
        del seq
        return 0

    def requires_long_prefill_offload(self, seq: Sequence) -> bool:
        """Whether this long prefill should be internally chunked through offload staging."""
        del seq
        return False

    def prefill_step_reservation_cost(self, seq: Sequence, scheduled_tokens: int) -> int:
        """Scheduler-side capacity consumed by scheduling a prefill chunk."""
        return int(scheduled_tokens)

    def decode_step_free_slots(self) -> int:
        """Writable KV capacity for one decode step."""
        return int(self.num_free_slots)

    def decode_step_free_slots_for(self, seq: Sequence) -> int:
        """Writable KV capacity for a specific decode candidate."""
        return int(self.decode_step_free_slots())

    def decode_step_reservation_cost(self, seq: Sequence) -> int:
        """Scheduler-side capacity consumed by scheduling one decode token."""
        return 1

    def prompt_admission_free_slots(self) -> int:
        """Slots pool used to decide whether a new prompt can be admitted."""
        return int(self.num_free_slots)

    def prompt_admission_cost(self, seq: Sequence) -> int:
        """Persistent slots needed to admit a complete prompt to its final representation."""
        hit_len = int(getattr(seq, "prefix_cache_hit_len", 0) or 0)
        return int(seq.num_prompt_tokens - hit_len)

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        """Logical slots reserved when a new prompt is admitted (scheduler-side accounting)."""
        return int(self.prompt_admission_cost(seq))

    def prompt_admission_failure_action(self) -> str:
        """Action when a prompt cannot be admitted: 'raise' or 'defer'."""
        return "defer"

    def prompt_admission_budgets(
        self,
        waiting_seqs: deque[Sequence],
        chunk_prefill_size: int,
    ) -> dict[str, int]:
        """Return admission budgets used by Scheduler for new prompts.

        Default behavior merges the reserved-prefill headroom into the same
        budget that gates new-prompt admission. This keeps the first budget
        check aligned with the later logical reservation accounting.
        """
        reserved = int(self.reserved_prefill_slots(waiting_seqs, chunk_prefill_size))
        free_slots = int(self.prompt_admission_free_slots())
        return {"slots": max(0, free_slots - reserved)}

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        """Return persistent final-representation admission costs per budget."""
        return {"slots": int(self.prompt_admission_cost(seq))}

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]):
        """Hook called when Scheduler admits a new prompt."""
        return

    def refresh_prefix_cache_hit(self, seq: Sequence) -> None:
        """Populate scheduler-visible prefix hit metadata for a fresh prompt."""
        self.clear_prefix_cache_hit(seq)

    def clear_prefix_cache_hit(self, seq: Sequence) -> None:
        """Clear scheduler-visible prefix hit metadata."""
        seq.clear_prefix_cache_hit()

    def build_prefix_kv_payload(self, seq: Sequence, block_start: int, block_end: int) -> object:
        del seq, block_start, block_end
        raise RuntimeError("This cache manager does not support mixed prefix KV payloads.")

    def attach_prefix_kv_payload(self, seq: Sequence, payload: object) -> None:
        del seq, payload
        raise RuntimeError("This cache manager does not support mixed prefix KV payload attach.")

    def validate_prefix_kv_attach(self, seq: Sequence) -> bool:
        del seq
        raise RuntimeError("This cache manager does not support mixed prefix KV payload attach.")

    def rollback_prefix_kv_attach(
        self,
        seq: Sequence,
        payloads: list[object],
        *,
        row_preexisted: bool,
    ) -> None:
        del seq, payloads, row_preexisted
        raise RuntimeError("This cache manager does not support mixed prefix KV attach rollback.")

    def free_prefix_kv_payload(self, payload: object) -> None:
        del payload
        raise RuntimeError("This cache manager does not support mixed prefix KV payload free.")

    def allocate_prefix_kv_payload_device(self, payload: object) -> None:
        del payload
        raise RuntimeError("This cache manager does not support mixed prefix KV promotion.")

    def allocate_prefix_kv_payloads_device(self, payloads: list[object]) -> None:
        for payload in payloads:
            self.allocate_prefix_kv_payload_device(payload)

    def free_prefix_kv_payload_device(self, payload: object) -> None:
        del payload
        raise RuntimeError("This cache manager does not support mixed prefix KV demotion.")

    def prefix_kv_payload_nbytes(self, payload: object) -> int:
        del payload
        raise RuntimeError("This cache manager does not support mixed prefix KV payload accounting.")

    def mark_materialized_prefix_kv_payload(self, seq: Sequence, payload: object) -> None:
        del seq, payload
        return None

    def free_slot_stats(self) -> dict[str, int]:
        """Return a small set of free-slot stats for logging/debugging."""
        return {"free_slots": int(self.num_free_slots)}

    def debug_state_summary(self) -> dict[str, Any]:
        """Return a synchronized-test snapshot without touching the inference hot path."""
        live_rows = {}
        seq_id_to_row = getattr(self, "seq_id_to_row", {})
        mappings = (
            [(layer_idx, mapping) for layer_idx, mapping in enumerate(seq_id_to_row)]
            if isinstance(seq_id_to_row, list)
            else [(None, seq_id_to_row)]
        )
        for layer_idx, mapping in mappings:
            if not isinstance(mapping, dict) or not mapping:
                continue
            row_seq_lens = getattr(self, "row_seq_lens")
            token_slots = getattr(self, "buffer_req_to_token_slots")
            if layer_idx is not None:
                row_seq_lens = row_seq_lens[layer_idx]
                token_slots = token_slots[layer_idx]
            records = []
            for seq_id, row_idx in sorted(mapping.items()):
                row_len = int(row_seq_lens[row_idx])
                record = {
                    "seq_id": int(seq_id),
                    "row_idx": int(row_idx),
                    "row_len": row_len,
                    "token_slots": _debug_tensor_summary(
                        token_slots[row_idx, :row_len]
                    ),
                }
                page_slots = getattr(self, "buffer_req_to_page_slots", None)
                if layer_idx is None and page_slots is not None:
                    page_size = int(getattr(self, "page_size"))
                    num_pages = (row_len + page_size - 1) // page_size
                    record["page_slots"] = _debug_tensor_summary(
                        page_slots[row_idx, :num_pages]
                    )
                records.append(record)
            live_rows["shared" if layer_idx is None else str(layer_idx)] = records

        prefix_state: dict[str, Any] | None = None
        prefix_cache = getattr(self, "prefix_cache", None)
        if prefix_cache is not None:
            blocks = []
            for block_id, block in sorted(prefix_cache.blocks.items()):
                blocks.append(
                    {
                        "stable_block_id": block_id.hex(),
                        "parent_block_id": (
                            None
                            if block.parent_block_id is None
                            else block.parent_block_id.hex()
                        ),
                        "logical_block_idx": int(block.logical_block_idx),
                        "token_ids": [int(token_id) for token_id in block.token_ids],
                        "ref_count": int(block.ref_count),
                        "last_access": int(block.last_access),
                        "eviction_priority": int(block.eviction_priority),
                        "device_present": bool(block.residency.device_present),
                        "host_present": bool(block.residency.host_present),
                        "transfer": (
                            None
                            if block.residency.transfer is None
                            else block.residency.transfer.value
                        ),
                        "payload": _debug_value_summary(block.payload),
                    }
                )
            prefix_state = {
                "fingerprint": prefix_cache.fingerprint.hex(),
                "stats": {
                    str(key): int(value)
                    for key, value in prefix_cache.stats().items()
                },
                "blocks": blocks,
            }

        return {
            "cache_manager_class": type(self).__name__,
            "free_slot_stats": {
                str(key): int(value) for key, value in self.free_slot_stats().items()
            },
            "live_rows": live_rows,
            "prefix_cache": prefix_state,
        }

    def _cache_slot_dtype_size(self) -> int:
        dtype = getattr(self.hf_config, "torch_dtype", torch.float16)
        if not isinstance(dtype, torch.dtype):
            dtype = torch.float16
        return int(torch.tensor([], dtype=dtype).element_size())

    def _dense_baseline_slots(self) -> int:
        slot_candidates = [
            getattr(self.config, "num_kvcache_slots", None),
            getattr(self, "num_slots", None),
            getattr(self, "full_num_slots", None),
            getattr(self, "deltakv_latent_num_slots", None),
            getattr(self, "deltakv_full_num_slots", None),
        ]
        slots = [int(value) for value in slot_candidates if isinstance(value, (int, float)) and int(value) > 0]
        if slots:
            return max(slots)
        return int(getattr(self.config, "max_num_seqs_in_gpu", 1)) * int(self.max_model_len)

    def _dense_baseline_bytes(self) -> int:
        dtype_size = self._cache_slot_dtype_size()
        slots = self._dense_baseline_slots()
        return int(slots * self.num_kv_layers * 2 * self.num_kv_heads * self.head_dim * dtype_size)

    @staticmethod
    def _tensor_storage_key(tensor: torch.Tensor) -> tuple[Any, ...]:
        storage = tensor.untyped_storage()
        return (
            str(tensor.device),
            int(storage.data_ptr()),
            int(storage.nbytes()),
        )

    @staticmethod
    def _tensor_storage_nbytes(tensor: torch.Tensor) -> int:
        return int(tensor.untyped_storage().nbytes())

    def _iter_accounting_tensors(self):
        seen_containers: set[int] = set()

        def visit(path: str, value):
            if torch.is_tensor(value):
                yield path, value
                return
            if value is None or isinstance(value, (str, bytes, int, float, bool)):
                return
            obj_id = id(value)
            if obj_id in seen_containers:
                return
            seen_containers.add(obj_id)
            if isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(key, (str, int)):
                        child_path = f"{path}.{key}"
                    else:
                        child_path = f"{path}.{type(key).__name__}"
                    yield from visit(child_path, item)
                return
            if isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    yield from visit(f"{path}.{idx}", item)
                return
            if isinstance(value, LayerBatchStates):
                for field_name, item in value.__dict__.items():
                    yield from visit(f"{path}.{field_name}", item)

        for name, value in self.__dict__.items():
            if name in {"config", "hf_config"}:
                continue
            yield from visit(name, value)

    @staticmethod
    def _memory_accounting_category(path: str) -> str:
        lower = path.lower()
        if any(token in lower for token in ("slot", "mapping", "req_to_token", "_map", "map_")):
            return "slot_map"
        if any(token in lower for token in ("scale", "scales", "min", "mins", "zero")):
            return "scale_min_metadata"
        if any(token in lower for token in ("pos", "lens", "length", "score", "indices", "idx")):
            return "metadata"
        if "cache" in lower:
            return "kv_or_latent"
        return "other"

    def _logical_live_kv_bytes(self) -> int:
        row_seq_lens = getattr(self, "row_seq_lens", None)
        if row_seq_lens is None:
            return 0
        try:
            live_tokens = int(row_seq_lens.sum())
        except Exception:
            return 0
        dtype_size = self._cache_slot_dtype_size()
        return int(live_tokens * self.num_kv_layers * 2 * self.num_kv_heads * self.head_dim * dtype_size)

    def memory_accounting(self) -> dict[str, Any]:
        """Return read-only tensor memory accounting for regression gates.

        The accounting is intentionally generic: cache-manager-specific tensors are
        grouped by stable attribute-name patterns, and unique tensor storages are
        counted once so views do not inflate the result.
        """
        seen_storages: set[tuple[Any, ...]] = set()
        categories = {
            "kv_or_latent": 0,
            "slot_map": 0,
            "scale_min_metadata": 0,
            "metadata": 0,
            "other": 0,
        }
        tensors: list[dict[str, Any]] = []
        for path, tensor in self._iter_accounting_tensors():
            key = self._tensor_storage_key(tensor)
            if key in seen_storages:
                continue
            seen_storages.add(key)
            nbytes = self._tensor_storage_nbytes(tensor)
            category = self._memory_accounting_category(path)
            categories[category] += nbytes
            tensors.append(
                {
                    "path": path,
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "device": str(tensor.device),
                    "nbytes": nbytes,
                    "category": category,
                }
            )

        dense_baseline_bytes = self._dense_baseline_bytes()
        allocated_tensor_bytes = int(sum(categories.values()))
        metadata_bytes = int(
            categories["slot_map"] + categories["scale_min_metadata"] + categories["metadata"]
        )
        observed_savings = None
        if dense_baseline_bytes > 0:
            observed_savings = 1.0 - (allocated_tensor_bytes / dense_baseline_bytes)

        theoretical = getattr(self.config, "memory_expected_savings", None)
        if theoretical is not None:
            theoretical = float(theoretical)

        return {
            "status": "success",
            "cache_manager_class": type(self).__name__,
            "dense_baseline_bytes": int(dense_baseline_bytes),
            "allocated_tensor_bytes": allocated_tensor_bytes,
            "logical_live_kv_bytes": int(self._logical_live_kv_bytes()),
            "slot_map_bytes": int(categories["slot_map"]),
            "scale_min_metadata_bytes": int(categories["scale_min_metadata"]),
            "metadata_bytes": metadata_bytes,
            "kv_or_latent_tensor_bytes": int(categories["kv_or_latent"]),
            "other_tensor_bytes": int(categories["other"]),
            "theoretical_savings": theoretical,
            "observed_savings": observed_savings,
            "tensor_count": len(tensors),
            "unique_storage_count": len(seen_storages),
            "dense_baseline": {
                "slots": int(self._dense_baseline_slots()),
                "layers": int(self.num_layers),
                "kv_layers": int(self.num_kv_layers),
                "num_kv_heads": int(self.num_kv_heads),
                "head_dim": int(self.head_dim),
                "dtype_size": int(self._cache_slot_dtype_size()),
            },
            "by_category": categories,
            "tensors": tensors,
        }

    def debug_live_seq_slots(self) -> dict[int, int]:
        """Return live seq_id -> occupied slot count for debugging."""
        return {}

    @abstractmethod
    def free_seq(self, seq_id: int):
        raise NotImplementedError

    @abstractmethod
    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def _prepare_prefill(self, seqs: list[Sequence]):
        raise NotImplementedError

    @abstractmethod
    def _prepare_decode(self, seqs: list[Sequence]):
        raise NotImplementedError

    def get_compressed_lens(self, req_indices):
        raise NotImplementedError
