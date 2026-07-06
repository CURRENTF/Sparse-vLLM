from __future__ import annotations

import os
from collections import deque
from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.constant import REDUNDANCY_BATCH_SIZE_FACTOR
from sparsevllm.method_registry import SUPPORTED_SPARSE_METHODS, normalize_sparse_method
from sparsevllm.triton_kernel.store_kvcache import store_kvcache
import sparsevllm.platforms as platforms
from sparsevllm.utils.log import logger, log_level


@dataclass
class LayerBatchStates:
    """存储当前 Batch 在特定层的前向计算状态。

    仅包含与物理存储和基本前向元数据相关的字段。
    """

    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    max_context_len: int | None = None
    req_indices: torch.Tensor | None = None


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

    def __init__(self, config: Config, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
        self.platform = platforms.current_platform
        self.device = self.platform.get_device(rank)
        self.hf_config = config.hf_config
        self.num_layers = self.hf_config.num_hidden_layers

        self.num_kv_heads = self.hf_config.num_key_value_heads // world_size
        self.head_dim = getattr(
            self.hf_config,
            "head_dim",
            self.hf_config.hidden_size // self.hf_config.num_attention_heads,
        )

        self.max_model_len = config.max_model_len
        self.max_buffer_rows = config.max_num_seqs_in_batch * REDUNDANCY_BATCH_SIZE_FACTOR

        self.kv_cache = None
        self._decode_static_max_context_len: int | None = None

    def _is_stream_capturing(self) -> bool:
        platform = getattr(self, "platform", None)
        if platform is not None:
            return platform.is_stream_capturing()
        return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())

    @staticmethod
    def create(config: Config, rank: int, world_size: int) -> "CacheManager":
        sparse_method = normalize_sparse_method(config.vllm_sparse_method)
        model_type = getattr(getattr(config, "hf_config", None), "model_type", "") or ""

        if model_type in {"deepseek_v2", "deepseek_v32"}:
            raise NotImplementedError(f"Unsupported Sparse-vLLM model_type={model_type!r}.")
        if sparse_method not in SUPPORTED_SPARSE_METHODS:
            raise ValueError(f"Unsupported vllm_sparse_method={sparse_method!r}.")
        if sparse_method == "deltakv":
            from .deltakv_runtime import DeltaKVCacheManager

            return DeltaKVCacheManager(config, rank, world_size)
        if sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            from .streamingllm import StreamingLLMCacheManager

            return StreamingLLMCacheManager(config, rank, world_size)
        if sparse_method in ("snapkv", "pyramidkv"):
            from .snapkv import SnapKVCacheManager

            return SnapKVCacheManager(config, rank, world_size)
        if sparse_method == "rkv":
            from .rkv import RKVCacheManager

            return RKVCacheManager(config, rank, world_size)
        if sparse_method == "skipkv":
            from .skipkv import SkipKVCacheManager

            return SkipKVCacheManager(config, rank, world_size)
        if sparse_method == "quest":
            from .quest import QuestCacheManager

            return QuestCacheManager(config, rank, world_size)
        if sparse_method == "omnikv":
            from .omnikv import OmniKVCacheManager

            return OmniKVCacheManager(config, rank, world_size)

        from .standard import StandardCacheManager

        return StandardCacheManager(config, rank, world_size)

    def _get_available_slots_info(self) -> tuple[int, int]:
        """返回 (可用显存字节数, 每层每 token 的字节数)"""
        config = self.config
        hf_config = config.hf_config
        free, total = self.platform.get_available_memory(self.device.index or 0)

        # 动态估计 max_num_batched_tokens
        reserved_mem = total * (1 - config.gpu_memory_utilization)
        intermediate_size = getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4)
        # MLP TP layers already assert intermediate_size is divisible by world_size.
        intermediate_size_per_rank = intermediate_size // self.world_size
        dtype_size = torch.tensor([], dtype=hf_config.torch_dtype).element_size()

        # Keep this heuristic conservative: large prefill batches can still peak on
        # MLP activations and allocator fragmentation after KV cache allocation.
        estimated_max_tokens = int(reserved_mem / (intermediate_size_per_rank * dtype_size * 10))
        allow_large_prefill_chunk = os.getenv("SPARSEVLLM_ALLOW_LARGE_PREFILL_CHUNK", "0") == "1"
        if 2 * config.chunk_prefill_size >= estimated_max_tokens:
            msg = (
                f"2*chunk_prefill_size={2 * config.chunk_prefill_size} >= "
                f"estimated_max_tokens={estimated_max_tokens}"
            )
            if not allow_large_prefill_chunk:
                raise AssertionError(msg)
            logger.warning(
                "{}; continuing because SPARSEVLLM_ALLOW_LARGE_PREFILL_CHUNK=1. "
                "This is an explicit experiment override and may OOM.",
                msg,
            )

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

        available_memory = int(total * config.gpu_memory_utilization - used - peak + current)
        slot_bytes_per_layer = 2 * self.num_kv_heads * self.head_dim * dtype_size

        if log_level == "DEBUG":
            logger.debug(
                f"[DEBUG] Available Memory: {available_memory / 1024**3:.2f} GB, "
                f"Slot Bytes Per Layer: {slot_bytes_per_layer / 1024**2:.4f} MB"
            )

        return available_memory, slot_bytes_per_layer

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
        del layer_idx, selection
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

    def uses_prefill_staging_step_capacity(self, seq: Sequence) -> bool:
        """Whether this prefill candidate uses a temporary staging/offload step capacity."""
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

    def free_slot_stats(self) -> dict[str, int]:
        """Return a small set of free-slot stats for logging/debugging."""
        return {"free_slots": int(self.num_free_slots)}

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
        return int(getattr(self.config, "max_num_seqs_in_batch", 1)) * int(self.max_model_len)

    def _dense_baseline_bytes(self) -> int:
        dtype_size = self._cache_slot_dtype_size()
        slots = self._dense_baseline_slots()
        return int(slots * self.num_layers * 2 * self.num_kv_heads * self.head_dim * dtype_size)

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
        return int(live_tokens * self.num_layers * 2 * self.num_kv_heads * self.head_dim * dtype_size)

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
