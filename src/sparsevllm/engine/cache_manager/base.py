from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from abc import ABC, abstractmethod

import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.constant import REDUNDANCY_BATCH_SIZE_FACTOR
from sparsevllm.utils.log import logger, log_level


@dataclass
class LayerBatchStates:
    """存储当前 Batch 在特定层的前向计算状态。

    仅包含与物理存储和基本前向元数据相关的字段。
    """

    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    req_indices: torch.Tensor | None = None


class CacheManager(ABC):
    """每个 Rank 只有一个 CacheManager，内部管理所有层的物理槽位和 KV Cache。"""

    def __init__(self, config: Config, rank: int, world_size: int):
        self.config = config
        self.rank = rank
        self.world_size = world_size
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

    @staticmethod
    def create(config: Config, rank: int, world_size: int) -> "CacheManager":
        sparse_method = config.vllm_sparse_method
        model_type = getattr(getattr(config, "hf_config", None), "model_type", "") or ""

        # DeepSeek MLA cache layout (required for DeepSeek-V2 / V3.2 model wiring).
        if model_type == "deepseek_v32":
            raise NotImplementedError(
                "DeepSeek-V3.2 sparsevllm support is disabled. "
                "Use DeepSeek-V2 or another backend for now."
            )
        if model_type == "deepseek_v2":
            if sparse_method not in ("", "dsa"):
                raise ValueError(
                    f"DeepSeek MLA cache does not support vllm_sparse_method={sparse_method!r}. "
                    "Use vanilla mode (empty string) or `dsa` where supported."
                )
            if sparse_method == "dsa" and model_type != "deepseek_v32":
                raise ValueError(
                    "vllm_sparse_method='dsa' is currently only supported for DeepSeek-V3.2 "
                    f"(model_type={model_type!r})."
                )
            from .deepseek_mla import DeepSeekMLACacheManager

            return DeepSeekMLACacheManager(config, rank, world_size)
        if sparse_method == "dsa":
            raise ValueError(
                "vllm_sparse_method='dsa' is currently only supported for DeepSeek-V3.2 "
                f"(model_type={model_type!r})."
            )
        if model_type == "qwen3" and isinstance(sparse_method, str) and sparse_method.startswith("deltakv"):
            raise NotImplementedError(
                "sparsevllm qwen3 + deltakv is disabled for now due to qk-norm/runtime mismatch. "
                "Use the HF backend for qwen3 DeltaKV inference."
            )

        if sparse_method == "deltakv":
            from .deltakv import DeltaKVCacheManager

            return DeltaKVCacheManager(config, rank, world_size)
        if sparse_method == "deltakv-triton":
            # Run DeltaKV logic, but use Triton for reconstruction.
            config.vllm_sparse_method = "deltakv"
            from .deltakv import DeltaKVCacheTritonManager

            return DeltaKVCacheTritonManager(config, rank, world_size)
        if sparse_method == "deltakv-triton-v2":
            # Run DeltaKV logic, but use Triton for reconstruction + eviction.
            config.vllm_sparse_method = "deltakv"
            from .deltakv import DeltaKVCacheTritonManagerV2

            return DeltaKVCacheTritonManagerV2(config, rank, world_size)
        if sparse_method == "deltakv-triton-v3":
            # Run DeltaKV logic, with Triton for reconstruction + eviction + blockwise L2-topk.
            config.vllm_sparse_method = "deltakv"
            from .deltakv import DeltaKVCacheTritonManagerV3

            return DeltaKVCacheTritonManagerV3(config, rank, world_size)
        if sparse_method == "deltakv-triton-v4":
            # Run DeltaKV logic, with Triton for reconstruction + eviction + blockwise L2-topk,
            # and extra kernel fusions (grouped-head reconstruction, fused gather+mean for clustering).
            config.vllm_sparse_method = "deltakv"
            from .deltakv import DeltaKVCacheTritonManagerV4

            return DeltaKVCacheTritonManagerV4(config, rank, world_size)
        if sparse_method == "deltakv-standalone":
            from .deltakv_standalone import DeltaKVStandaloneCacheManager

            return DeltaKVStandaloneCacheManager(config, rank, world_size)
        if sparse_method == "deltakv-snapkv":
            from .deltakv_snapkv import DeltaKVSnapKVCacheManager

            return DeltaKVSnapKVCacheManager(config, rank, world_size)
        if sparse_method in ("deltakv-triton-v3-offload", "deltakv-triton-v3-with-offload"):
            # Run DeltaKV logic, with Triton for reconstruction + eviction + blockwise L2-topk,
            # and offload latent cache to CPU.
            config.vllm_sparse_method = "deltakv"
            config.deltakv_offload_latent = True
            from .deltakv import DeltaKVCacheTritonManagerV3WithOffload

            return DeltaKVCacheTritonManagerV3WithOffload(config, rank, world_size)
        if sparse_method in ("deltakv-triton-v3-cuda-offload", "deltakv-triton-v3-with-cuda-offload"):
            # Run DeltaKV logic, and offload latents to CPU, but gather them back to GPU via
            # the custom CUDA kernel under `sparsevllm/cuda_kernel` (ShadowKV-style).
            config.vllm_sparse_method = "deltakv"
            config.deltakv_offload_latent = True
            from .deltakv import DeltaKVCacheTritonManagerV3WithCUDAOffload

            return DeltaKVCacheTritonManagerV3WithCUDAOffload(config, rank, world_size)
        if sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            from .streamingllm import StreamingLLMCacheManager

            return StreamingLLMCacheManager(config, rank, world_size)
        if sparse_method in ("snapkv", "pyramidkv"):
            from .snapkv import SnapKVCacheManager

            return SnapKVCacheManager(config, rank, world_size)
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
        free, total = torch.cuda.mem_get_info()

        # 动态估计 max_num_batched_tokens
        reserved_mem = total * (1 - config.gpu_memory_utilization)
        intermediate_size = getattr(hf_config, "intermediate_size", hf_config.hidden_size * 4)
        dtype_size = torch.tensor([], dtype=hf_config.torch_dtype).element_size()

        # Keep this heuristic conservative: large prefill batches can still peak on
        # MLP activations and allocator fragmentation after KV cache allocation.
        estimated_max_tokens = int(reserved_mem / (intermediate_size * dtype_size * 10))
        assert 2 * config.chunk_prefill_size < estimated_max_tokens, (
            f"{2 * config.chunk_prefill_size} >= {estimated_max_tokens}"
        )

        if estimated_max_tokens < config.max_num_batched_tokens:
            logger.warning(
                f"Estimated max_num_batched_tokens ({estimated_max_tokens}) is smaller than config "
                f"({config.max_num_batched_tokens}). Updating to avoid OOM."
            )
            config.max_num_batched_tokens = estimated_max_tokens

        logger.info(f"Set dynamically max_num_batched_tokens = {config.max_num_batched_tokens}")

        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

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
    def get_layer_compute_tensors(self, layer_idx: int, sparse_controller):
        raise NotImplementedError

    @abstractmethod
    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        raise NotImplementedError

    def on_kv_stored(self, layer_idx: int, k: torch.Tensor, slot_mapping: torch.Tensor):
        """Optional method-specific hook after KV has been written into cache."""
        return None

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
        return int(seq.num_prompt_tokens - seq.num_prefilled_tokens)

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], chunk_prefill_size: int) -> int:
        """Estimate slots to reserve for already-started prefills (to reduce decode thrashing)."""
        reserved = 0
        for seq in waiting_seqs:
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                reserved += int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        return reserved

    def prompt_admission_free_slots(self) -> int:
        """Slots pool used to decide whether a new prompt can be admitted."""
        return int(self.num_free_slots)

    def prompt_admission_cost(self, seq: Sequence) -> int:
        """Slots needed to admit a new prompt (at prefill start)."""
        return int(seq.num_prompt_tokens)

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        """Logical slots reserved when a new prompt is admitted (scheduler-side accounting)."""
        return int(seq.num_prompt_tokens)

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
        """Return admission costs (per budget) used by Scheduler for new prompts."""
        return {"slots": int(self.prompt_admission_cost(seq))}

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]):
        """Hook called when Scheduler admits a new prompt."""
        return

    def free_slot_stats(self) -> dict[str, int]:
        """Return a small set of free-slot stats for logging/debugging."""
        return {"free_slots": int(self.num_free_slots)}

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
