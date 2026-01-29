from collections import deque
from dataclasses import dataclass
import os
import numpy as np
import torch

from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.constant import REDUNDANCY_BATCH_SIZE_FACTOR
from sparsevllm.utils.log import logger, log_level
from sparsevllm.utils.profiler import profiler
from sparsevllm.layers.rotary_embedding import get_rope, apply_rotary_emb, reverse_rotary_emb

from abc import ABC, abstractmethod


@dataclass
class LayerBatchStates:
    """
    存储当前 Batch 在特定层的前向计算状态。
    仅包含与物理存储和基本前向元数据相关的字段。
    """
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    req_indices: torch.Tensor | None = None


class CacheManager(ABC):
    """
    每个 Rank 只有一个 CacheManager，内部管理所有层的物理槽位和 KV Cache。
    """
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
        if sparse_method == "deltakv":
            return DeltaKVCacheManager(config, rank, world_size)
        if sparse_method == "deltakv-triton":
            # Run DeltaKV logic, but use Triton for reconstruction.
            config.vllm_sparse_method = "deltakv"
            return DeltaKVCacheTritonManager(config, rank, world_size)
        if sparse_method == "deltakv-triton-v2":
            # Run DeltaKV logic, but use Triton for reconstruction + eviction.
            config.vllm_sparse_method = "deltakv"
            return DeltaKVCacheTritonManagerV2(config, rank, world_size)
        if sparse_method == "deltakv-triton-v3":
            # Run DeltaKV logic, with Triton for reconstruction + eviction + blockwise L2-topk.
            config.vllm_sparse_method = "deltakv"
            return DeltaKVCacheTritonManagerV3(config, rank, world_size)
        if sparse_method == "deltakv-triton-v4":
            # Run DeltaKV logic, with Triton for reconstruction + eviction + blockwise L2-topk,
            # and extra kernel fusions (grouped-head reconstruction, fused gather+mean for clustering).
            config.vllm_sparse_method = "deltakv"
            return DeltaKVCacheTritonManagerV4(config, rank, world_size)
        if sparse_method in ("deltakv-triton-v3-offload", "deltakv-triton-v3-with-offload"):
            # Run DeltaKV logic, with Triton for reconstruction + eviction + blockwise L2-topk,
            # and offload latent cache to CPU.
            config.vllm_sparse_method = "deltakv"
            config.deltakv_offload_latent = True
            return DeltaKVCacheTritonManagerV3WithOffload(config, rank, world_size)
        if sparse_method in ("deltakv-triton-v3-cuda-offload", "deltakv-triton-v3-with-cuda-offload"):
            # Run DeltaKV logic, and offload latents to CPU, but gather them back to GPU via
            # the custom CUDA kernel under `sparsevllm/cuda_kernel` (ShadowKV-style).
            config.vllm_sparse_method = "deltakv"
            config.deltakv_offload_latent = True
            return DeltaKVCacheTritonManagerV3WithCUDAOffload(config, rank, world_size)
        if "snapkv" == sparse_method or "pyramidkv" == sparse_method:
            return SnapKVCacheManager(config, rank, world_size)
        if "omnikv" == sparse_method:
            return OmniKVCacheManager(config, rank, world_size)
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

        estimated_max_tokens = int(reserved_mem / (intermediate_size * dtype_size * 6))
        assert 2 * config.chunk_prefill_size < estimated_max_tokens, f'{2 * config.chunk_prefill_size} >= {estimated_max_tokens}'
        # estimated_max_tokens = min(estimated_max_tokens, 2 * config.chunk_prefill_size)

        if estimated_max_tokens < config.max_num_batched_tokens:
            logger.warning(
                f"Estimated max_num_batched_tokens ({estimated_max_tokens}) is smaller than config "
                f"({config.max_num_batched_tokens}). Updating to avoid OOM."
            )
            config.max_num_batched_tokens = estimated_max_tokens

        logger.info(f'Set dynamically max_num_batched_tokens = {config.max_num_batched_tokens}')

        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]

        available_memory = int(total * config.gpu_memory_utilization - used - peak + current)
        slot_bytes_per_layer = 2 * self.num_kv_heads * self.head_dim * dtype_size

        if log_level == 'DEBUG':
            logger.debug(f"[DEBUG] Available Memory: {available_memory / 1024**3:.2f} GB, Slot Bytes Per Layer: {slot_bytes_per_layer / 1024**2:.4f} MB")

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
        return "raise"

    def prompt_admission_budgets(self) -> dict[str, int]:
        """Return admission budgets used by Scheduler for new prompts."""
        return {"slots": int(self.prompt_admission_free_slots())}

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        """Return admission costs (per budget) used by Scheduler for new prompts."""
        return {"slots": int(self.prompt_admission_cost(seq))}

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]):
        """Hook called when Scheduler admits a new prompt."""
        return

    def free_slot_stats(self) -> dict[str, int]:
        """Return a small set of free-slot stats for logging/debugging."""
        return {"free_slots": int(self.num_free_slots)}

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


class StandardCacheManager(CacheManager):

    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.allocate_kv_cache()

        num_slots = config.num_kvcache_slots
        self.free_slots_stack = torch.arange(num_slots, dtype=torch.int32, device="cuda")
        self._num_free_slots = num_slots

        self.buffer_req_to_token_slots = torch.zeros(
            (self.max_buffer_rows, self.max_model_len), dtype=torch.int32, device="cuda"
        )

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.layer_batch_state = LayerBatchStates()

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        num_layers = self.num_layers

        slot_bytes = num_layers * slot_bytes_per_layer
        self.config.num_kvcache_slots = available_memory // slot_bytes
        assert self.config.num_kvcache_slots > 0, "可用显存不足以分配 KV Cache"

        logger.info(
            f"Standard Mode: Each layer can accommodate {self.config.num_kvcache_slots} tokens."
        )
        self.kv_cache = torch.empty(
            2,
            num_layers,
            self.config.num_kvcache_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        return self.layer_batch_state

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx], self.layer_batch_state.slot_mapping

    def get_layer_compute_tensors(self, layer_idx: int, sparse_controller):
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        return self.buffer_req_to_token_slots

    @property
    def num_free_slots(self) -> int:
        return self._num_free_slots

    def _get_free_row(self, seq_id: int) -> int:
        if seq_id in self.seq_id_to_row:
            return self.seq_id_to_row[seq_id]
        if not self.free_rows:
            raise RuntimeError("No free rows in cache manager buffer!")
        row_idx = self.free_rows.popleft()
        self.seq_id_to_row[seq_id] = row_idx
        return row_idx

    @torch.no_grad()
    def _allocate(self, seq_id: int, size: int) -> torch.Tensor:
        with profiler.record("cache_allocate"):
            assert self._num_free_slots >= size, (
                f"Out of KV cache slots: need {size}, free {self._num_free_slots}"
            )

            row_idx = self._get_free_row(seq_id)
            cur_len = self.row_seq_lens[row_idx]

            ptr = self._num_free_slots
            select_index = self.free_slots_stack[ptr - size: ptr]
            self._num_free_slots -= size

            self.buffer_req_to_token_slots[row_idx, cur_len: cur_len + size] = select_index
            self.row_seq_lens[row_idx] += size

            return select_index

    @torch.no_grad()
    def _allocate_batch(self, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        assert self._num_free_slots >= batch_size, (
            f"Out of KV cache slots: need {batch_size}, free {self._num_free_slots}"
        )

        row_indices = [self._get_free_row(sid) for sid in seq_ids]
        cur_lens = self.row_seq_lens[row_indices]

        ptr = self._num_free_slots
        select_indices = self.free_slots_stack[ptr - batch_size: ptr]
        self._num_free_slots -= batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device="cuda")
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device="cuda")
        self.buffer_req_to_token_slots[rows_gpu, cols_gpu] = select_indices
        self.row_seq_lens[row_indices] += 1

        return select_indices

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                raise ValueError

            cur_len = self.row_seq_lens[row_idx]
            slots = self.buffer_req_to_token_slots[row_idx, :cur_len]

            assert cur_len > 0
            ptr = self._num_free_slots
            self.free_slots_stack[ptr: ptr + cur_len] = slots
            self._num_free_slots += cur_len

            self.buffer_req_to_token_slots[row_idx, :] = 0
            self.row_seq_lens[row_idx] = 0
            self.free_rows.append(row_idx)

            if log_level == 'DEBUG': logger.debug(f'free seq {row_idx} with {cur_len} tokens')

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise ValueError('不需要实现该方法')

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device="cuda")
            context_lens_list = []
            req_indices = []

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size

                if seq.seq_id in self.seq_id_to_row:
                    row_idx = self.seq_id_to_row[seq.seq_id]
                    if self.row_seq_lens[row_idx] != start_idx:
                        raise ValueError(
                            "KV cache row length mismatch in prefill: "
                            f"seq_id={seq.seq_id} row_seq_len={self.row_seq_lens[row_idx]} "
                            f"start_idx={start_idx}"
                        )

                self._allocate(seq.seq_id, chunk_size)
                row_idx = self.seq_id_to_row[seq.seq_id]
                slot_mapping[token_offset: token_offset + chunk_size] = self.buffer_req_to_token_slots[row_idx, start_idx:end_idx]
                context_lens_list.append(end_idx)
                req_indices.append(row_idx)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]

                input_ids_np[token_offset: token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset: token_offset + chunk_size] = np.arange(start_idx, end_idx)

                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device="cuda")
            req_indices_tensor = torch.tensor(req_indices, dtype=torch.int32, device="cuda")

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.req_indices = req_indices_tensor

            if log_level == 'DEBUG':
                logger.debug(f'{context_lens_list=}   {req_indices=}  {slot_mapping[:10].tolist()=}  {slot_mapping[-10:].tolist()=}')

            input_ids = torch.from_numpy(input_ids_np).to("cuda")
            positions = torch.from_numpy(positions_np).to("cuda")
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            batch_size = len(seqs)
            input_ids_list = [seq.last_token for seq in seqs]
            positions_list = [seq.num_tokens - 1 for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            new_slots_batch = self._allocate_batch(seq_ids, 1)
            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            context_lens = torch.tensor(
                self.row_seq_lens[row_indices],
                dtype=torch.int32,
                device="cuda",
            )
            req_indices = torch.tensor(row_indices, dtype=torch.int32, device="cuda")

            slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
            slot_mapping[:] = new_slots_batch

            self.layer_batch_state.slot_mapping = slot_mapping
            self.layer_batch_state.context_lens = context_lens
            self.layer_batch_state.req_indices = req_indices

            logger.debug(f'{slot_mapping=}   {context_lens.tolist()=}  {slot_mapping[:10]=}  {slot_mapping[-10:]=}')

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None


class SnapKVCacheManager(CacheManager):
    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        self.allocate_kv_cache()

        self.layer_num_slots = []
        self.free_slots_stack = []
        self._num_free_slots = []
        self.buffer_req_to_token_slots = []
        self.seq_id_to_row = []
        self.free_rows = []
        self.row_seq_lens = []
        self.layer_batch_states = [LayerBatchStates() for _ in range(self.num_layers)]

        for layer_id in range(self.num_layers):
            num_slots = (
                config.num_kvcache_slots[layer_id]
                if isinstance(config.num_kvcache_slots, list)
                else config.num_kvcache_slots
            )
            self.layer_num_slots.append(num_slots)
            self.free_slots_stack.append(
                torch.arange(num_slots, dtype=torch.int32, device="cuda")
            )
            self._num_free_slots.append(num_slots)
            self.buffer_req_to_token_slots.append(
                torch.zeros(
                    (self.max_buffer_rows, self.max_model_len),
                    dtype=torch.int32,
                    device="cuda",
                )
            )
            self.seq_id_to_row.append({})
            self.free_rows.append(deque(range(self.max_buffer_rows)))
            self.row_seq_lens.append(np.zeros((self.max_buffer_rows,), dtype=np.int32))

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        num_layers = self.num_layers

        if config.pyramid_layer_ratios is not None:
            # PyramidKV: 根据比例分配每层不同大小的 cache
            total_ratio = sum(config.pyramid_layer_ratios)
            base_slots = available_memory // (slot_bytes_per_layer * total_ratio)
            assert base_slots > 0, "可用显存不足以分配 KV Cache"

            layer_slots = [int(base_slots * ratio) for ratio in config.pyramid_layer_ratios]
            assert layer_slots[0] == max(layer_slots), (
                f"Layer 0 必须是最胖层，但 layer_slots[0]={layer_slots[0]}, max={max(layer_slots)}"
            )

            self.kv_cache = []
            for layer_idx in range(num_layers):
                num_slots = layer_slots[layer_idx]
                k_cache = torch.empty(
                    num_slots, self.num_kv_heads, self.head_dim,
                    dtype=self.hf_config.torch_dtype, device="cuda"
                )
                v_cache = torch.empty(
                    num_slots, self.num_kv_heads, self.head_dim,
                    dtype=self.hf_config.torch_dtype, device="cuda"
                )
                self.kv_cache.append((k_cache, v_cache))

            config.num_kvcache_slots = layer_slots
            logger.info(f"PyramidKV: Layer slots = {layer_slots}, base_slots = {base_slots}")
        else:
            # 标准模式：所有层使用相同大小
            slot_bytes = num_layers * slot_bytes_per_layer
            config.num_kvcache_slots = available_memory // slot_bytes
            assert config.num_kvcache_slots > 0, "可用显存不足以分配 KV Cache"

            logger.info(
                f"Standard Mode (SnapKV): Each layer can accommodate {config.num_kvcache_slots} tokens."
            )
            self.kv_cache = torch.empty(
                2,
                num_layers,
                config.num_kvcache_slots,
                self.num_kv_heads,
                self.head_dim,
                dtype=self.hf_config.torch_dtype,
                device="cuda",
            )

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        return self.layer_batch_states[layer_idx]

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(self.kv_cache, list):
            return self.kv_cache[layer_idx]
        elif isinstance(self.kv_cache, torch.Tensor):
            return self.kv_cache[0, layer_idx], self.kv_cache[1, layer_idx]
        else:
            raise ValueError

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        return k_cache, v_cache, self.layer_batch_states[layer_idx].slot_mapping

    def get_layer_compute_tensors(self, layer_idx: int, sparse_controller):
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        return self.buffer_req_to_token_slots[layer_idx]

    @property
    def num_free_slots(self) -> int:
        return min(self._num_free_slots)

    def prefill_batched_tokens_margin(self) -> int:
        # Keep headroom for the "window" tokens used by SnapKV/PyramidKV logic.
        return int(getattr(self.config, "snapkv_window_size", 0) or 0)

    def remaining_prefill_tokens(self, seq: Sequence) -> int:
        remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
        window = int(getattr(self.config, "snapkv_window_size", 0) or 0)
        if window > 0 and remaining > window:
            return remaining - window
        return remaining

    def _get_free_row(self, layer_idx: int, seq_id: int) -> int:
        if seq_id in self.seq_id_to_row[layer_idx]:
            return self.seq_id_to_row[layer_idx][seq_id]
        if not self.free_rows[layer_idx]:
            raise RuntimeError("No free rows in cache manager buffer!")
        row_idx = self.free_rows[layer_idx].popleft()
        self.seq_id_to_row[layer_idx][seq_id] = row_idx
        return row_idx

    @torch.no_grad()
    def _allocate(self, layer_idx: int, seq_id: int, size: int) -> torch.Tensor:
        with profiler.record("cache_allocate"):
            assert self._num_free_slots[layer_idx] >= size, (
                f"Out of KV cache slots: need {size}, free {self._num_free_slots[layer_idx]}"
            )

            row_idx = self._get_free_row(layer_idx, seq_id)
            cur_len = self.row_seq_lens[layer_idx][row_idx]

            ptr = self._num_free_slots[layer_idx]
            select_index = self.free_slots_stack[layer_idx][ptr - size: ptr]
            self._num_free_slots[layer_idx] -= size

            self.buffer_req_to_token_slots[layer_idx][row_idx, cur_len: cur_len + size] = select_index
            self.row_seq_lens[layer_idx][row_idx] += size

            return select_index

    @torch.no_grad()
    def _allocate_batch(self, layer_idx: int, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        assert self._num_free_slots[layer_idx] >= batch_size, (
            f"Out of KV cache slots: need {batch_size}, free {self._num_free_slots[layer_idx]}"
        )

        row_indices = [self._get_free_row(layer_idx, sid) for sid in seq_ids]
        cur_lens = self.row_seq_lens[layer_idx][row_indices]

        ptr = self._num_free_slots[layer_idx]
        select_indices = self.free_slots_stack[layer_idx][ptr - batch_size: ptr]
        self._num_free_slots[layer_idx] -= batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device="cuda")
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device="cuda")
        self.buffer_req_to_token_slots[layer_idx][rows_gpu, cols_gpu] = select_indices.to(torch.int32)
        self.row_seq_lens[layer_idx][row_indices] += 1

        return select_indices

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            for layer_idx in range(self.num_layers):
                row_idx = self.seq_id_to_row[layer_idx].pop(seq_id, None)
                if row_idx is None:
                    raise ValueError

                cur_len = self.row_seq_lens[layer_idx][row_idx]
                slots = self.buffer_req_to_token_slots[layer_idx][row_idx, :cur_len]

                if cur_len > 0:
                    ptr = self._num_free_slots[layer_idx]
                    self.free_slots_stack[layer_idx][ptr: ptr + cur_len] = slots
                    self._num_free_slots[layer_idx] += cur_len

                self.buffer_req_to_token_slots[layer_idx][row_idx, :] = 0
                self.row_seq_lens[layer_idx][row_idx] = 0
                self.free_rows[layer_idx].append(row_idx)

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        if keep_indices is None:
            return

        row_idx = self.seq_id_to_row[layer_idx].get(seq.seq_id)
        if row_idx is None:
            raise ValueError

        cur_len = self.row_seq_lens[layer_idx][row_idx]
        if log_level == 'DEBUG':
            keep_cnt = int(keep_indices.numel())
            logger.debug(
                "[SnapKV] free_part_slots(before): "
                f"layer={layer_idx} seq_id={seq.seq_id} row={row_idx} "
                f"context_len={int(cur_len)} keep={keep_cnt} drop={max(0, int(cur_len) - keep_cnt)}"
            )
        old_slots = self.buffer_req_to_token_slots[layer_idx][row_idx, :cur_len].clone()

        new_slots = old_slots[keep_indices]

        mask = torch.ones_like(old_slots, dtype=torch.bool)
        mask[keep_indices] = False
        dropped_slots = old_slots[mask]

        if dropped_slots.numel() > 0:
            count = dropped_slots.numel()
            ptr = self._num_free_slots[layer_idx]
            self.free_slots_stack[layer_idx][ptr: ptr + count] = dropped_slots
            self._num_free_slots[layer_idx] += count
        else:
            logger.warning(f"[SnapKV] dropped 0 tokens? layer={layer_idx} seq_id={seq.seq_id} row={row_idx} cur_len={int(cur_len)}")

        self.buffer_req_to_token_slots[layer_idx][row_idx, :] = 0
        self.buffer_req_to_token_slots[layer_idx][row_idx, :new_slots.numel()] = new_slots
        self.row_seq_lens[layer_idx][row_idx] = new_slots.numel()
        if log_level == 'DEBUG':
            logger.debug(
                "[SnapKV] free_part_slots(after): "
                f"layer={layer_idx} seq_id={seq.seq_id} row={row_idx} "
                f"context_len={int(cur_len)} -> {int(new_slots.numel())}"
            )

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            layers_slot_mapping_cuda = torch.empty(
                (self.num_layers, total_chunk_tokens), dtype=torch.int32, device="cuda"
            )
            context_lens_list = [[] for _ in range(self.num_layers)]

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size

                for layer_id in range(self.num_layers):
                    if seq.seq_id in self.seq_id_to_row[layer_id]:
                        row_idx = self.seq_id_to_row[layer_id][seq.seq_id]
                        if self.row_seq_lens[layer_id][row_idx] != start_idx:
                            raise ValueError(
                                "KV cache row length mismatch in prefill: "
                                f"layer={layer_id} seq_id={seq.seq_id} "
                                f"row_seq_len={self.row_seq_lens[layer_id][row_idx]} "
                                f"start_idx={start_idx}"
                            )
                    self._allocate(layer_id, seq.seq_id, chunk_size)
                    row_idx = self.seq_id_to_row[layer_id][seq.seq_id]
                    layers_slot_mapping_cuda[layer_id, token_offset: token_offset + chunk_size] = \
                        self.buffer_req_to_token_slots[layer_id][row_idx, start_idx:end_idx]
                    context_lens_list[layer_id].append(end_idx)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]

                input_ids_np[token_offset: token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset: token_offset + chunk_size] = np.arange(start_idx, end_idx)

                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            layers_context_lens_cuda = torch.tensor(context_lens_list, dtype=torch.int32, device="cuda")

            for layer_id in range(self.num_layers):
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = layers_context_lens_cuda[layer_id]
                req_ids = [self.seq_id_to_row[layer_id][seq.seq_id] for seq in seqs]
                state.req_indices = torch.tensor(req_ids, dtype=torch.int32, device="cuda")

            input_ids = torch.from_numpy(input_ids_np).to("cuda")
            positions = torch.from_numpy(positions_np).to("cuda")
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            batch_size = len(seqs)
            input_ids_list = [seq.last_token for seq in seqs]
            positions_list = [seq.num_tokens - 1 for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            layers_slot_mapping_cuda = torch.empty(
                (self.num_layers, batch_size), dtype=torch.int32, device="cuda"
            )
            layers_context_lens = []

            for layer_id in range(self.num_layers):
                new_slots_batch = self._allocate_batch(layer_id, seq_ids, 1)
                layers_slot_mapping_cuda[layer_id] = new_slots_batch

                row_indices = [self.seq_id_to_row[layer_id][sid] for sid in seq_ids]
                layers_context_lens.append(self.row_seq_lens[layer_id][row_indices])

            layers_context_lens_cuda = torch.from_numpy(np.array(layers_context_lens)).to(
                device="cuda",
                dtype=torch.int32,
            )

            for layer_id in range(self.num_layers):
                state = self.layer_batch_states[layer_id]
                state.slot_mapping = layers_slot_mapping_cuda[layer_id]
                state.context_lens = layers_context_lens_cuda[layer_id]
                req_ids = [self.seq_id_to_row[layer_id][seq.seq_id] for seq in seqs]
                state.req_indices = torch.tensor(req_ids, dtype=torch.int32, device="cuda")

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None


class OmniKVCacheManager(StandardCacheManager):
    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)


class DeltaKVCacheManager(CacheManager):
    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)
        assert world_size == 1, "DeltaKVCacheManager currently only supports world_size=1 (No TP support for compressors)"

        self.full_attn_layers = config.full_attn_layers
        assert isinstance(self.full_attn_layers, list) and isinstance(self.full_attn_layers[0], int)
        self.deltakv_layer_ids = [i for i in range(self.num_layers) if i not in self.full_attn_layers]
        self.full_layer_ids = [i for i in range(self.num_layers) if i in self.full_attn_layers]
        self.deltakv_layer_to_idx = {l: i for i, l in enumerate(self.deltakv_layer_ids)}
        self.full_layer_to_idx = {l: i for i, l in enumerate(self.full_layer_ids)}

        # NOTE: 这些变量在 allocate_kv_cache() 中被赋值，必须在调用前初始化为 None
        self.full_num_slots = 0
        self.deltakv_latent_num_slots = 0
        self.deltakv_full_num_slots = 0
        self.full_kv_cache = None
        self.deltakv_full_kv_cache = None
        self.deltakv_latent_cache = None
        self.deltakv_latent_to_full_slots = None
        self.deltakv_slot_to_pos = None
        # Reserved scratch slots in the sparse full-KV pool for on-the-fly reconstruction.
        # This is set in allocate_kv_cache() and used to provide backpressure to Scheduler
        # (so requests wait instead of crashing inside attention).
        self._deltakv_temp_full_reserve = 0
        # Budgeting for centers: we reserve "future center slots" at admission time to avoid
        # admitting more long prompts than the sparse full-KV pool can sustain.
        self._deltakv_centers_capacity = 0
        self._deltakv_centers_reserved_total = 0
        self._deltakv_centers_reserved_by_seq: dict[int, int] = {}

        self.allocate_kv_cache()

        self.free_slots_stack_full = torch.arange(self.full_num_slots, dtype=torch.int32, device="cuda")
        self._num_free_slots_full = self.full_num_slots

        self.free_slots_stack_deltakv_full = torch.arange(self.deltakv_full_num_slots, dtype=torch.int32, device="cuda")
        self._num_free_slots_deltakv_full = self.deltakv_full_num_slots

        self.free_slots_stack_deltakv_latent = torch.arange(self.deltakv_latent_num_slots, dtype=torch.int32, device="cuda")
        self._num_free_slots_deltakv_latent = self.deltakv_latent_num_slots

        self.full_layer_slots_map = torch.zeros(
            (self.max_buffer_rows, self.max_model_len), dtype=torch.int32, device="cuda"
        )
        self.sparse_layer_raw_slots_map = torch.full(
            (self.max_buffer_rows, self.max_model_len), -1, dtype=torch.int32, device="cuda"
        )
        self.sparse_layer_latent_slots_map = torch.full(
            (self.max_buffer_rows, self.max_model_len), -1, dtype=torch.int32, device="cuda"
        )

        self.seq_id_to_row: dict[int, int] = {}
        self.free_rows = deque(range(self.max_buffer_rows))
        self.row_seq_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.row_deltakv_compressed_lens = np.zeros((self.max_buffer_rows,), dtype=np.int32)
        self.row_deltakv_center_slots = [[None for _ in range(self.num_layers)] for _ in range(self.max_buffer_rows)]

        self.full_layer_batch_states = LayerBatchStates()
        self.deltakv_layer_batch_states = LayerBatchStates()

        from sparsevllm.utils.compressor import create_compressor
        self.compress_down = []
        self.compress_up = []
        num_deltakv_layers = len(self.deltakv_layer_ids)
        for _ in range(num_deltakv_layers):
            self.compress_down.append(create_compressor(is_down=True, config=config).cuda())
            self.compress_up.append(create_compressor(is_down=False, config=config).cuda())

        # 初始化 RoPE 模块，用于 De-RoPE/Re-RoPE 操作
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_model_len,
            base=self.hf_config.rope_theta,
            rope_scaling=None,
        ).cuda()
        # cos_sin_cache shape: (max_pos, 1, head_dim) - 包含 (cos, sin)
        self.cos_sin_cache = self.rotary_emb.cos_sin_cache

        # Per-step/per-segment cache for DeltaKV view planning (shared across layers).
        self._deltakv_view_cache_key: tuple[int, int, int, int, int] | None = None
        self._deltakv_view_cache_value = None

    def _deltakv_reset_view_cache(self):
        self._deltakv_view_cache_key = None
        self._deltakv_view_cache_value = None

    def prepare_step(self, seqs: list[Sequence], is_prefill: bool):
        # Reset per-step cache to avoid stale views across steps.
        self._deltakv_reset_view_cache()
        return super().prepare_step(seqs, is_prefill)

    def allocate_kv_cache(self):
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        dtype_size = torch.tensor([], dtype=self.hf_config.torch_dtype).element_size()

        num_full_layers = len(self.full_layer_ids)
        num_deltakv_layers = len(self.deltakv_layer_ids)
        assert num_full_layers > 0, "DeltaKV requires at least one full-attention layer."
        assert num_deltakv_layers > 0, "DeltaKV requires at least one sparse layer."

        # Full layers store all tokens. Sparse layers store:
        # - latent for all tokens (for reconstruction)
        # - a bounded full-KV pool: centers + uncompressed buffer (+ current chunk) + reconstructed top tokens.
        latent_bytes = config.kv_compressed_size * dtype_size
        cluster_ratio = max(0.0, float(config.cluster_ratio))

        per_token_bytes = (
            num_full_layers * slot_bytes_per_layer
            + num_deltakv_layers * (cluster_ratio * slot_bytes_per_layer + latent_bytes)
        )
        if per_token_bytes <= 0:
            raise ValueError("Invalid KV cache allocation configuration.")

        max_tokens = max(1, int(available_memory / per_token_bytes))

        # Reserve some headroom for the sparse full-KV pool (centers/buffer/temp).
        # This is important for large batch sizes, where the required temp slots can spike.
        reserve_ratio = float(getattr(config, "deltakv_full_pool_reserve_ratio", 0.0))
        if reserve_ratio > 0:
            reserve_ratio = max(0.0, min(0.5, reserve_ratio))
            max_tokens = max(1, int(max_tokens * (1.0 - reserve_ratio)))
        self.full_num_slots = max_tokens
        self.deltakv_latent_num_slots = max_tokens

        # Now decide the sparse full-KV pool size from remaining bytes.
        bytes_full_layers = self.full_num_slots * num_full_layers * slot_bytes_per_layer
        bytes_latent = self.deltakv_latent_num_slots * num_deltakv_layers * latent_bytes
        bytes_misc = 0  # small tensors (slot maps) are negligible vs KV
        bytes_left = available_memory - bytes_full_layers - bytes_latent - bytes_misc
        if bytes_left <= 0:
            raise RuntimeError(
                "Not enough GPU memory left for DeltaKV full-KV pool after allocating full layers + latent cache. "
                "Try reducing max_model_len / gpu_memory_utilization / kv_compressed_size."
            )
        max_deltakv_full_slots = max(1, int(bytes_left // (num_deltakv_layers * slot_bytes_per_layer)))

        sink = int(config.num_sink_tokens)
        recent = int(config.num_recent_tokens)
        # Sparse full-KV pool must cover:
        # - per-seq resident tokens: sink + (<=2*recent) uncompressed buffer
        # - current prefill step's chunk tokens across the whole batch
        # - temp reconstructed top tokens (per seq, per sparse layer attention)
        #
        # If we under-estimate this, the system should backpressure at scheduling time
        # (queue) rather than crashing in _allocate_temp_deltakv_full().
        max_seqs = int(config.max_num_seqs_in_batch)
        top_decode = int(config.num_top_tokens)
        top_prefill = int(config.num_top_tokens_in_prefill)
        # Worst-case total reconstructed top tokens within a single attention call:
        #   num_seqs_in_batch * top_k_per_seq
        # For prefill, num_seqs_in_batch is also bounded by (max_num_batched_tokens / chunk_size)
        # when chunks are full; cap by max_seqs to avoid over-estimation.
        max_prefill_seqs_by_tokens = (int(config.max_num_batched_tokens) + int(config.chunk_prefill_size) - 1) // int(
            config.chunk_prefill_size
        )
        max_prefill_seqs = min(max_seqs, max_prefill_seqs_by_tokens)
        total_top_slots = max(max_seqs * top_decode, max_prefill_seqs * top_prefill)
        max_step_chunk = int(min(int(config.max_num_batched_tokens), max_seqs * int(config.chunk_prefill_size)))
        overhead_slots = max_seqs * (sink + 2 * recent) + total_top_slots + max_step_chunk
        if max_deltakv_full_slots <= overhead_slots:
            raise RuntimeError(
                f"DeltaKV full-KV pool too small: max={max_deltakv_full_slots}, required>={overhead_slots + 1}. "
                "Reduce chunk_prefill_size/num_top_tokens/num_recent_tokens or increase gpu_memory_utilization."
            )

        desired_centers = max(1, int(cluster_ratio * self.full_num_slots * 1.5))
        centers_capacity = min(desired_centers, max_deltakv_full_slots - overhead_slots)
        self.deltakv_full_num_slots = overhead_slots + centers_capacity
        self._deltakv_centers_capacity = int(centers_capacity)
        # Reserve scratch capacity for reconstruction. Scheduler-visible free slots will exclude this
        # so requests wait instead of triggering temp-slot OOM mid-forward.
        self._deltakv_temp_full_reserve = min(self.deltakv_full_num_slots, int(total_top_slots))

        logger.info(
            f"DeltaKV allocation: full_layers_slots={self.full_num_slots}; "
            f"deltakv_full_slots={self.deltakv_full_num_slots} (overhead={overhead_slots}, centers={centers_capacity}); "
            f"deltakv_latent_slots={self.deltakv_latent_num_slots} "
            f"(full_layers={num_full_layers}, deltakv_layers={num_deltakv_layers}, "
            f"deltakv_full_pool_reserve_ratio={reserve_ratio:.3f}, "
            f"deltakv_temp_full_reserve={self._deltakv_temp_full_reserve})."
        )

        self.full_kv_cache = torch.empty(
            2,
            num_full_layers,
            self.full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )

        self.deltakv_full_kv_cache = torch.empty(
            2,
            num_deltakv_layers,
            self.deltakv_full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )
        self.deltakv_latent_cache = torch.empty(
            num_deltakv_layers,
            self.deltakv_latent_num_slots,
            config.kv_compressed_size,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )
        self.deltakv_latent_to_full_slots = torch.full(
            (num_deltakv_layers, self.deltakv_latent_num_slots, config.deltakv_k_neighbors),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        self.deltakv_slot_to_pos = torch.full(
            (self.deltakv_full_num_slots,),
            -1,
            dtype=torch.int32,
            device="cuda",
        )

    def get_layer_batch_states(self, layer_idx: int) -> LayerBatchStates:
        if layer_idx in self.full_attn_layers:
            return self.full_layer_batch_states
        else:
            return self.deltakv_layer_batch_states

    def get_layer_kv_cache(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx in self.full_layer_to_idx:
            idx = self.full_layer_to_idx[layer_idx]
            return self.full_kv_cache[0, idx], self.full_kv_cache[1, idx]
        else:
            idx = self.deltakv_layer_to_idx[layer_idx]
            return self.deltakv_full_kv_cache[0, idx], self.deltakv_full_kv_cache[1, idx]

    def get_layer_store_view(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        k_cache, v_cache = self.get_layer_kv_cache(layer_idx)
        state = self.get_layer_batch_states(layer_idx)
        return k_cache, v_cache, state.slot_mapping

    def get_layer_compute_tensors(self, layer_idx: int, sparse_controller):
        raise NotImplementedError

    def get_layer_buffer_req_to_token_slots(self, layer_idx: int) -> torch.Tensor:
        if layer_idx in self.full_layer_to_idx:
            return self.full_layer_slots_map
        else:
            # DeltaKV sparse layers never directly expose a dense Req->slots table because
            # most historical tokens are either compressed or reconstructed on-the-fly.
            raise NotImplementedError("DeltaKV sparse layers should use SparseController.get_read_view().")

    @property
    def num_free_slots(self) -> int:
        # Scheduling should be conservative: we must be able to allocate both
        # full-layer KV slots and DeltaKV full-KV slots for new tokens.
        deltakv_usable = self._num_free_slots_deltakv_full - int(getattr(self, "_deltakv_temp_full_reserve", 0) or 0)
        return min(self._num_free_slots_full, max(0, deltakv_usable))

    def num_free_slots_full_layers(self) -> int:
        return int(self._num_free_slots_full)

    def reserved_prefill_slots(self, waiting_seqs: deque[Sequence], chunk_prefill_size: int) -> int:
        # DeltaKV can evict sparse-layer KV during long prefill; reserving the entire remaining
        # prompt is overly conservative and causes decode thrashing. Reserve at most one chunk
        # per in-progress prefill sequence.
        reserved = 0
        for seq in waiting_seqs:
            if 0 < seq.num_prefilled_tokens < seq.num_prompt_tokens:
                remaining = int(seq.num_prompt_tokens - seq.num_prefilled_tokens)
                reserved += min(remaining, int(chunk_prefill_size))
        return reserved

    def prompt_admission_free_slots(self) -> int:
        # Full-attention layers store every token and cannot be evicted, so gate admission by that pool.
        return self.num_free_slots_full_layers()

    def prompt_admission_cost(self, seq: Sequence) -> int:
        # Full-attn layers must hold prompt + maximum decode length for this sequence.
        return int(seq.num_prompt_tokens + (getattr(seq, "max_tokens", 0) or 0))

    def prompt_logical_reservation_cost(self, seq: Sequence) -> int:
        # DeltaKV does not need to reserve the full prompt in sparse layers.
        return 0

    def prompt_admission_failure_action(self) -> str:
        # Defer admission until other sequences finish and free full-layer slots.
        return "defer"

    def prompt_admission_budgets(self) -> dict[str, int]:
        # Gate on both full-attention pool and (future) centers budget.
        centers_free = max(0, int(self._deltakv_centers_capacity) - int(self._deltakv_centers_reserved_total))
        return {
            "full_layers": self.num_free_slots_full_layers(),
            "deltakv_centers": centers_free,
        }

    def _estimate_centers_for_total_len(self, total_len: int) -> int:
        total_len = int(total_len)
        cluster_ratio = float(getattr(self.config, "cluster_ratio", 0.0) or 0.0)
        if cluster_ratio <= 0:
            return 0
        sink = int(getattr(self.config, "num_sink_tokens", 0) or 0)
        recent = int(getattr(self.config, "num_recent_tokens", 0) or 0)
        cluster_step = max(1, int(1.0 / max(1e-6, cluster_ratio)))
        # Tokens in sink + recent buffer never become centers; be conservative with ceil.
        effective = total_len - sink - recent
        if effective <= 0:
            return 0
        return int((effective + cluster_step - 1) // cluster_step)

    def prompt_admission_costs(self, seq: Sequence) -> dict[str, int]:
        total_len = int(seq.num_prompt_tokens + (getattr(seq, "max_tokens", 0) or 0))
        return {
            "full_layers": int(seq.num_prompt_tokens + (getattr(seq, "max_tokens", 0) or 0)),
            "deltakv_centers": self._estimate_centers_for_total_len(total_len),
        }

    def on_prompt_admitted(self, seq: Sequence, costs: dict[str, int]):
        # Reserve future centers budget to prevent admitting too many long prompts.
        seq_id = int(seq.seq_id)
        if seq_id in self._deltakv_centers_reserved_by_seq:
            return
        centers = int(costs.get("deltakv_centers", 0) or 0)
        self._deltakv_centers_reserved_by_seq[seq_id] = centers
        self._deltakv_centers_reserved_total += centers

    @torch.no_grad()
    def _allocate_temp_deltakv_full(self, size: int) -> torch.Tensor:
        """Allocate DeltaKV full-KV slots without touching per-seq slot maps (scratch for reconstruction)."""
        if self._num_free_slots_deltakv_full < size:
            raise RuntimeError(
                "Out of DeltaKV full cache slots (temp). "
                f"need={size} free={self._num_free_slots_deltakv_full} "
                f"(reserved_for_temp={int(getattr(self, '_deltakv_temp_full_reserve', 0) or 0)}). "
                "Try reducing batch_size/num_top_tokens, or increase deltakv_full_pool_reserve_ratio."
            )
        ptr = self._num_free_slots_deltakv_full
        select_index = self.free_slots_stack_deltakv_full[ptr - size: ptr]
        self._num_free_slots_deltakv_full -= size
        return select_index

    @torch.no_grad()
    def free_temp_deltakv_full(self, slots: torch.Tensor | None):
        """Return scratch slots allocated by _allocate_temp_deltakv_full()."""
        if slots is None or slots.numel() == 0:
            return
        slots = slots.to(torch.int32)
        ptr = self._num_free_slots_deltakv_full
        self.free_slots_stack_deltakv_full[ptr: ptr + slots.numel()] = slots
        self._num_free_slots_deltakv_full += slots.numel()
        # Scratch slots have no stable position.
        self.deltakv_slot_to_pos[slots] = -1
        # Any cached view that references these temp slots is now invalid.
        self._deltakv_reset_view_cache()

    def _get_free_row(self, seq_id: int) -> int:
        if seq_id in self.seq_id_to_row:
            return self.seq_id_to_row[seq_id]
        if not self.free_rows:
            raise RuntimeError("No free rows in cache manager buffer!")
        row_idx = self.free_rows.popleft()
        self.seq_id_to_row[seq_id] = row_idx
        return row_idx

    @torch.no_grad()
    def _allocate_full(self, seq_id: int, size: int) -> torch.Tensor:
        assert self._num_free_slots_full >= size, (
            f"Out of full KV cache slots: need {size}, free {self._num_free_slots_full}"
        )
        row_idx = self._get_free_row(seq_id)
        cur_len = self.row_seq_lens[row_idx]

        ptr = self._num_free_slots_full
        select_index = self.free_slots_stack_full[ptr - size: ptr]
        self._num_free_slots_full -= size

        self.full_layer_slots_map[row_idx, cur_len: cur_len + size] = select_index
        return select_index

    @torch.no_grad()
    def _allocate_deltakv_full(self, seq_id: int, size: int) -> torch.Tensor:
        usable = self._num_free_slots_deltakv_full - int(getattr(self, "_deltakv_temp_full_reserve", 0) or 0)
        if usable < size:
            raise RuntimeError(
                "Out of DeltaKV full cache slots (persistent). "
                f"need={size} free_total={self._num_free_slots_deltakv_full} free_usable={usable} "
                f"(reserved_for_temp={int(getattr(self, '_deltakv_temp_full_reserve', 0) or 0)}). "
                "Reduce concurrency/chunk size, or increase deltakv_full_pool_reserve_ratio."
            )
        row_idx = self._get_free_row(seq_id)
        cur_len = self.row_seq_lens[row_idx]

        ptr = self._num_free_slots_deltakv_full
        select_index = self.free_slots_stack_deltakv_full[ptr - size: ptr]
        self._num_free_slots_deltakv_full -= size

        self.sparse_layer_raw_slots_map[row_idx, cur_len: cur_len + size] = select_index
        self.deltakv_slot_to_pos[select_index] = torch.arange(cur_len, cur_len + size, device="cuda", dtype=torch.int32)
        return select_index

    @torch.no_grad()
    def _allocate_deltakv_latent(self, size: int) -> torch.Tensor:
        assert self._num_free_slots_deltakv_latent >= size, (
            f"Out of DeltaKV latent cache slots: need {size}, free {self._num_free_slots_deltakv_latent}"
        )
        ptr = self._num_free_slots_deltakv_latent
        select_index = self.free_slots_stack_deltakv_latent[ptr - size: ptr]
        self._num_free_slots_deltakv_latent -= size
        return select_index

    @torch.no_grad()
    def _allocate_batch_full(self, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        assert self._num_free_slots_full >= batch_size, (
            f"Out of full KV cache slots: need {batch_size}, free {self._num_free_slots_full}"
        )
        row_indices = [self._get_free_row(sid) for sid in seq_ids]
        cur_lens = self.row_seq_lens[row_indices]

        ptr = self._num_free_slots_full
        select_indices = self.free_slots_stack_full[ptr - batch_size: ptr]
        self._num_free_slots_full -= batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device="cuda")
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device="cuda")
        self.full_layer_slots_map[rows_gpu, cols_gpu] = select_indices
        return select_indices

    @torch.no_grad()
    def _allocate_batch_deltakv_full(self, seq_ids: list[int], size: int) -> torch.Tensor:
        assert size == 1, "Batch allocation currently only supports size=1 (Decode)"
        batch_size = len(seq_ids)
        usable = self._num_free_slots_deltakv_full - int(getattr(self, "_deltakv_temp_full_reserve", 0) or 0)
        if usable < batch_size:
            raise RuntimeError(
                "Out of DeltaKV full cache slots (persistent batch). "
                f"need={batch_size} free_total={self._num_free_slots_deltakv_full} free_usable={usable} "
                f"(reserved_for_temp={int(getattr(self, '_deltakv_temp_full_reserve', 0) or 0)}). "
                "Reduce concurrency, or increase deltakv_full_pool_reserve_ratio."
            )
        row_indices = [self._get_free_row(sid) for sid in seq_ids]
        cur_lens = self.row_seq_lens[row_indices]

        ptr = self._num_free_slots_deltakv_full
        select_indices = self.free_slots_stack_deltakv_full[ptr - batch_size: ptr]
        self._num_free_slots_deltakv_full -= batch_size

        rows_gpu = torch.tensor(row_indices, dtype=torch.long, device="cuda")
        cols_gpu = torch.tensor(cur_lens, dtype=torch.long, device="cuda")
        self.sparse_layer_raw_slots_map[rows_gpu, cols_gpu] = select_indices
        self.deltakv_slot_to_pos[select_indices] = cols_gpu.to(torch.int32)
        return select_indices

    def free_seq(self, seq_id: int):
        with profiler.record("cache_free_seq"):
            reserved = self._deltakv_centers_reserved_by_seq.pop(seq_id, 0)
            if reserved:
                self._deltakv_centers_reserved_total -= int(reserved)
            row_idx = self.seq_id_to_row.pop(seq_id, None)
            if row_idx is None:
                raise ValueError

            cur_len = self.row_seq_lens[row_idx]
            assert cur_len > 0

            # 清空 full layers
            full_slots = self.full_layer_slots_map[row_idx, :cur_len]
            ptr = self._num_free_slots_full
            self.free_slots_stack_full[ptr: ptr + cur_len] = full_slots
            self._num_free_slots_full += cur_len

            # 清空 deltakv layers
            deltakv_slots = self.sparse_layer_raw_slots_map[row_idx, :cur_len]
            mask = deltakv_slots >= 0
            # assert mask.any()  # 理论上必然存在 raw cache
            slots = deltakv_slots[mask]
            ptr = self._num_free_slots_deltakv_full
            # 未压缩释放
            self.free_slots_stack_deltakv_full[ptr: ptr + slots.numel()] = slots
            self._num_free_slots_deltakv_full += slots.numel()
            self.deltakv_slot_to_pos[slots] = -1

            latent_slots = self.sparse_layer_latent_slots_map[row_idx, :cur_len]
            mask_latent = latent_slots >= 0
            if mask_latent.any():
                slots = latent_slots[mask_latent]
                ptr = self._num_free_slots_deltakv_latent
                self.free_slots_stack_deltakv_latent[ptr: ptr + slots.numel()] = slots
                self._num_free_slots_deltakv_latent += slots.numel()

            self.full_layer_slots_map[row_idx, :] = 0
            self.sparse_layer_raw_slots_map[row_idx, :] = -1
            self.sparse_layer_latent_slots_map[row_idx, :] = -1
            self.row_seq_lens[row_idx] = 0
            self.row_deltakv_compressed_lens[row_idx] = 0
            self.row_deltakv_center_slots[row_idx] = [None for _ in range(self.num_layers)]
            self.free_rows.append(row_idx)

    def free_slot_stats(self) -> dict[str, int]:
        full_free = int(getattr(self, "_num_free_slots_full", 0) or 0)
        deltakv_full_free_total = int(getattr(self, "_num_free_slots_deltakv_full", 0) or 0)
        deltakv_latent_free = int(getattr(self, "_num_free_slots_deltakv_latent", 0) or 0)
        temp_reserve = int(getattr(self, "_deltakv_temp_full_reserve", 0) or 0)
        deltakv_full_free_usable = max(0, deltakv_full_free_total - temp_reserve)
        centers_cap = int(getattr(self, "_deltakv_centers_capacity", 0) or 0)
        centers_reserved = int(getattr(self, "_deltakv_centers_reserved_total", 0) or 0)
        centers_free = max(0, centers_cap - centers_reserved)
        active = int(len(getattr(self, "seq_id_to_row", {}) or {}))
        return {
            "free_slots": int(self.num_free_slots),
            "full_free": full_free,
            "deltakv_full_free_total": deltakv_full_free_total,
            "deltakv_full_free_usable": deltakv_full_free_usable,
            "deltakv_temp_reserve": temp_reserve,
            "deltakv_latent_free": deltakv_latent_free,
            "centers_capacity": centers_cap,
            "centers_reserved": centers_reserved,
            "centers_free": centers_free,
            "active_seqs": active,
        }

    def free_part_slots(self, layer_idx: int, seq: Sequence, keep_indices: torch.Tensor):
        raise ValueError("DeltaKV does not support partial slot freeing via this method.")

    def _prepare_prefill(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_prefill"):
            total_chunk_tokens = sum(seq.current_chunk_size for seq in seqs)

            input_ids_np = np.empty(total_chunk_tokens, dtype=np.int64)
            positions_np = np.empty(total_chunk_tokens, dtype=np.int64)
            cu_seqlens_q = [0]

            full_slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device="cuda")
            deltakv_slot_mapping = torch.empty(total_chunk_tokens, dtype=torch.int32, device="cuda")
            context_lens_list = []
            req_indices = []

            token_offset = 0
            for seq in seqs:
                chunk_size = seq.current_chunk_size
                start_idx = seq.num_prefilled_tokens
                end_idx = start_idx + chunk_size

                if seq.seq_id in self.seq_id_to_row:
                    row_idx = self.seq_id_to_row[seq.seq_id]
                    if self.row_seq_lens[row_idx] != start_idx:
                        raise ValueError(
                            "KV cache row length mismatch in prefill: "
                            f"seq_id={seq.seq_id} row_seq_len={self.row_seq_lens[row_idx]} "
                            f"start_idx={start_idx}"
                        )

                self._allocate_full(seq.seq_id, chunk_size)
                self._allocate_deltakv_full(seq.seq_id, chunk_size)
                row_idx = self.seq_id_to_row[seq.seq_id]
                full_slot_mapping[token_offset: token_offset + chunk_size] = \
                    self.full_layer_slots_map[row_idx, start_idx:end_idx]
                deltakv_slot_mapping[token_offset: token_offset + chunk_size] = \
                    self.sparse_layer_raw_slots_map[row_idx, start_idx:end_idx]

                self.row_seq_lens[row_idx] += chunk_size
                context_lens_list.append(end_idx)
                req_indices.append(row_idx)

                chunk_tokens = seq.token_ids
                if len(chunk_tokens) > chunk_size:
                    chunk_tokens = chunk_tokens[start_idx:end_idx]

                input_ids_np[token_offset: token_offset + chunk_size] = chunk_tokens
                positions_np[token_offset: token_offset + chunk_size] = np.arange(start_idx, end_idx)

                cu_seqlens_q.append(cu_seqlens_q[-1] + chunk_size)
                token_offset += chunk_size

            context_lens = torch.tensor(context_lens_list, dtype=torch.int32, device="cuda")
            req_indices_tensor = torch.tensor(req_indices, dtype=torch.int32, device="cuda")

            full_state = self.full_layer_batch_states
            full_state.slot_mapping = full_slot_mapping
            full_state.context_lens = context_lens
            full_state.req_indices = req_indices_tensor

            deltakv_state = self.deltakv_layer_batch_states
            deltakv_state.slot_mapping = deltakv_slot_mapping
            deltakv_state.context_lens = context_lens
            deltakv_state.req_indices = req_indices_tensor

            input_ids = torch.from_numpy(input_ids_np).to("cuda")
            positions = torch.from_numpy(positions_np).to("cuda")
            cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, device="cuda")
            return input_ids, positions, cu_seqlens_q

    def _prepare_decode(self, seqs: list[Sequence]):
        with profiler.record("cache_prepare_decode"):
            batch_size = len(seqs)
            input_ids_list = [seq.last_token for seq in seqs]
            positions_list = [seq.num_tokens - 1 for seq in seqs]
            seq_ids = [seq.seq_id for seq in seqs]

            full_slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device="cuda")
            deltakv_slot_mapping = torch.empty((batch_size,), dtype=torch.int32, device="cuda")

            full_slots = self._allocate_batch_full(seq_ids, 1)
            deltakv_slots = self._allocate_batch_deltakv_full(seq_ids, 1)
            full_slot_mapping[:] = full_slots
            deltakv_slot_mapping[:] = deltakv_slots

            row_indices = [self.seq_id_to_row[sid] for sid in seq_ids]
            self.row_seq_lens[row_indices] += 1
            context_lens = torch.tensor(
                self.row_seq_lens[row_indices],
                dtype=torch.int32,
                device="cuda",
            )
            req_indices = torch.tensor(row_indices, dtype=torch.int32, device="cuda")

            full_state = self.full_layer_batch_states
            full_state.slot_mapping = full_slot_mapping
            full_state.context_lens = context_lens
            full_state.req_indices = req_indices

            deltakv_state = self.deltakv_layer_batch_states
            deltakv_state.slot_mapping = deltakv_slot_mapping
            deltakv_state.context_lens = context_lens
            deltakv_state.req_indices = req_indices

            input_ids = torch.tensor(input_ids_list, dtype=torch.int64, device="cuda")
            positions = torch.tensor(positions_list, dtype=torch.int64, device="cuda")
            return input_ids, positions, None

    def get_compressed_lens(self, req_indices: torch.Tensor) -> torch.Tensor:
        compressed = self.row_deltakv_compressed_lens[req_indices.cpu().numpy()]
        return torch.tensor(compressed, dtype=torch.int32, device="cuda")

    @staticmethod
    def _metric_l2(kv_states, all_centers):
        """Compute an L2-equivalent *ranking* score for top-k selection.

        For squared L2 distance: ||a-b||^2 = ||a||^2 + ||b||^2 - 2*dot(a,b).
        For a fixed `a`, ||a||^2 is constant across all `b`, so argmin(||a-b||^2)
        is equivalent to argmax(2*dot(a,b) - ||b||^2).

        We return `scores = 2*dot(a,b) - ||b||^2` (higher is better), keeping the
        large (N, M) score matrix in bf16/fp16 to avoid fp32 bandwidth overhead.
        """
        # kv_states: (1, N, D), all_centers: (1, M, D) for eviction.
        a = kv_states[0]
        b = all_centers[0]
        if a.numel() == 0 or b.numel() == 0:
            return kv_states.new_empty((kv_states.shape[0], kv_states.shape[1], all_centers.shape[1]))

        # GEMM via cuBLAS / tensorcores; output stays in low precision.
        dot = torch.matmul(a, b.transpose(0, 1))  # (N, M)

        # Keep norm computation small; cast down for the broadcast combine.
        b_norm = (b * b).sum(dim=1, dtype=torch.float32).to(dot.dtype)  # (M,)
        scores = dot.mul(2.0).sub_(b_norm.unsqueeze(0))
        return scores.unsqueeze(0)

    @staticmethod
    def _metric_dot(kv_states, all_centers):
        # Used only for top-k ranking; keep low precision for speed.
        return torch.matmul(kv_states, all_centers.transpose(-1, -2))

    @staticmethod
    def _metric_cosine(kv_states, all_centers, eps: float = 1e-6):
        # Keep normalization in fp32; matmul in fp32 since inputs are normalized anyway.
        kv_states_f = kv_states.float()
        all_centers_f = all_centers.float()
        kv_norm = kv_states_f / (kv_states_f.norm(p=2, dim=-1, keepdim=True) + eps)
        c_norm = all_centers_f / (all_centers_f.norm(p=2, dim=-1, keepdim=True) + eps)
        return torch.matmul(kv_norm, c_norm.transpose(-1, -2))

    def _gather_kv_unrope_by_slots(
        self,
        layer_idx: int,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        """Gather KV (concat) in a position-independent space (de-RoPE on K; V unchanged).

        Returns: (N, kv_dim) on CUDA.
        """
        assert layer_idx in self.deltakv_layer_to_idx
        l_idx = self.deltakv_layer_to_idx[layer_idx]
        k_cache = self.deltakv_full_kv_cache[0, l_idx]
        v_cache = self.deltakv_full_kv_cache[1, l_idx]

        slots_i64 = slots.to(torch.long)
        k_rope = k_cache[slots_i64]  # (N, kv_heads, head_dim)
        v = v_cache[slots_i64]

        pos = self.deltakv_slot_to_pos[slots.to(torch.int32)].to(torch.long)
        if (pos < 0).any():
            raise RuntimeError("DeltaKV: center slot has unknown position (deltakv_slot_to_pos == -1).")
        cos_sin = self.cos_sin_cache[pos]  # (N, 1, head_dim)
        cos, sin = cos_sin.chunk(2, dim=-1)
        k_unrope = reverse_rotary_emb(k_rope, cos, sin)

        kv_dim_half = self.num_kv_heads * self.head_dim
        k_flat = k_unrope.reshape(-1, kv_dim_half)
        v_flat = v.reshape(-1, kv_dim_half)
        return torch.cat([k_flat, v_flat], dim=-1)

    def _cluster_compress(
        self,
        layer_idx: int,
        kv_states: torch.Tensor,  # (1, N, kv_dim), de-RoPE already applied on K
        existing_center_slots: torch.Tensor,  # (M0,)
        cluster_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute top-k father *slots* and per-token base KV mean for a contiguous block.

        Returns:
          father_slots: (N, K) int32 physical slots
          base_kv: (1, N, kv_dim) float/bf16 mean of father KVs (de-RoPE space)
        """
        assert kv_states.dim() == 3 and kv_states.shape[0] == 1
        _, n, kv_dim = kv_states.shape
        k_neighbors = int(self.config.deltakv_k_neighbors)

        # Existing centers are always visible (from earlier blocks). New centers come from this block.
        with profiler.record("deltakv_cluster_existing_centers"):
            existing_centers = (
                self._gather_kv_unrope_by_slots(layer_idx, existing_center_slots).unsqueeze(0)
                if existing_center_slots.numel() > 0
                else kv_states.new_zeros((1, 0, kv_dim))
            )
        new_centers = kv_states[:, ::cluster_step, :]
        all_centers = torch.cat([existing_centers, new_centers], dim=1)  # (1, M, kv_dim)
        m0 = existing_centers.shape[1]
        m_new = new_centers.shape[1]

        metric_type = self.config.cluster_metric
        with profiler.record("deltakv_cluster_metric"):
            if metric_type == "l2":
                scores = self._metric_l2(kv_states, all_centers)
            elif metric_type == "dot":
                scores = self._metric_dot(kv_states, all_centers)
            elif metric_type == "cosine":
                scores = self._metric_cosine(kv_states, all_centers)
            elif metric_type == "fastdot":
                # Fast approximate metric: pure dot-product in low precision, no fp32 casts/norms.
                # Only used for top-k selection; accuracy is intentionally relaxed for speed.
                scores = torch.bmm(kv_states, all_centers.transpose(1, 2))
            else:
                raise ValueError(f"Unknown cluster_metric: {metric_type}")

        # Causal mask: within the current block, a token can only use new centers sampled at <= its index.
        if m_new > 0:
            with profiler.record("deltakv_cluster_causal_mask"):
                rows = torch.arange(n, device=kv_states.device).view(n, 1)
                cols = (torch.arange(m_new, device=kv_states.device).view(1, m_new) * cluster_step)
                mask_new = cols <= rows  # (N, m_new)
                mask_existing = torch.ones((n, m0), device=kv_states.device, dtype=torch.bool)
                full_mask = torch.cat([mask_existing, mask_new], dim=1)  # (N, M)
                scores = scores.masked_fill(~full_mask.unsqueeze(0), float("-inf"))

        k_eff = min(k_neighbors, all_centers.shape[1])
        if k_eff <= 0:
            raise RuntimeError("DeltaKV: no available centers to assign.")
        with profiler.record("deltakv_cluster_topk"):
            topk_indices = scores.topk(k=k_eff, dim=-1).indices  # (1, N, K)

        # Base KV mean in de-RoPE space.
        with profiler.record("deltakv_cluster_gather_mean"):
            gather_idx = topk_indices.view(1, -1)[:, :, None].expand(-1, -1, kv_dim)
            gathered = all_centers.gather(1, gather_idx).view(1, n, k_eff, kv_dim).mean(dim=2)
        return topk_indices.squeeze(0).to(torch.int32), gathered

    @torch.no_grad()
    def deltakv_evict(self, seqs: list[Sequence]):
        # Called from SparseController.post_forward(), which runs outside model forward.
        # Must be no-grad to avoid building enormous autograd graphs.
        with profiler.record("deltakv_evict_total"):
            self._deltakv_evict_impl(seqs)

    def _deltakv_evict_impl(self, seqs: list[Sequence]):
        if not self.deltakv_layer_ids:
            return
        sink = int(self.config.num_sink_tokens)
        recent = int(self.config.num_recent_tokens)
        cluster_step = max(1, int(1.0 / max(1e-6, float(self.config.cluster_ratio))))

        # Compress per sequence (long-text batches are typically small).
        for seq in seqs:
            with profiler.record("deltakv_evict_seq"):
                self._deltakv_evict_one_seq(seq, sink=sink, recent=recent, cluster_step=cluster_step)

    def _deltakv_evict_one_seq(self, seq: Sequence, *, sink: int, recent: int, cluster_step: int):
        row_idx = self.seq_id_to_row.get(seq.seq_id, None)
        if row_idx is None:
            return

        total_len = int(self.row_seq_lens[row_idx])
        compressed_len = int(self.row_deltakv_compressed_lens[row_idx])  # length of finalized history (excluding sink)
        buffer_start = sink + compressed_len
        buffer_len = total_len - buffer_start
        if buffer_len <= recent:
            return

        # Evict as much as possible, but keep at least `recent` tokens in the uncompressed buffer.
        # Match the reference logic: compress in multiples of `recent` (tail_token_size).
        evict_len = ((buffer_len - recent) // recent) * recent
        if evict_len <= 0:
            return

        evict_start = buffer_start
        evict_end = evict_start + evict_len

        # Raw slots exist for the evicted block before we start.
        with profiler.record("deltakv_evict_read_slots"):
            raw_slots_block = self.sparse_layer_raw_slots_map[row_idx, evict_start:evict_end].clone()
        if (raw_slots_block < 0).any():
            raise RuntimeError("DeltaKV eviction expects raw slots for the buffer block.")

        # Select new centers (prototypes) by fixed stride within the evicted block.
        with profiler.record("deltakv_evict_select_centers"):
            center_rel = torch.arange(0, evict_len, cluster_step, device="cuda", dtype=torch.long)
            new_center_slots = raw_slots_block[center_rel].to(torch.int32)

        # Initialize per-layer center slots (previous centers, without current block).
        with profiler.record("deltakv_evict_prev_centers"):
            sink_slots = self.sparse_layer_raw_slots_map[row_idx, :sink].to(torch.int32)
            prev_center_slots_by_layer: dict[int, torch.Tensor] = {}
            for layer_idx in self.deltakv_layer_ids:
                existing = self.row_deltakv_center_slots[row_idx][layer_idx]
                prev_center_slots_by_layer[layer_idx] = (sink_slots if existing is None else existing.to(torch.int32))

        # De-RoPE KV for the whole evicted block per layer, compute assignments, and store latents
        # for non-center tokens only. Center tokens stay as full KV references (and remain mapped).
        with profiler.record("deltakv_evict_build_masks"):
            is_center = torch.zeros((evict_len,), device="cuda", dtype=torch.bool)
            is_center[center_rel] = True
            to_compress_mask = ~is_center
            num_to_compress = int(to_compress_mask.sum().item())
        if num_to_compress <= 0:
            # All tokens become centers (e.g., cluster_ratio==1). Still advance history boundary.
            with profiler.record("deltakv_evict_append_centers_only"):
                for layer_idx in self.deltakv_layer_ids:
                    self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                        [prev_center_slots_by_layer[layer_idx], new_center_slots], dim=0
                    )
                self.row_deltakv_compressed_lens[row_idx] += evict_len
            return

        # Allocate shared latent slots for this block (shared index across layers).
        with profiler.record("deltakv_evict_alloc_latent"):
            latent_slots = self._allocate_deltakv_latent(num_to_compress).to(torch.int32)
            pos_all = torch.arange(evict_start, evict_end, device="cuda", dtype=torch.long)
            pos_to_compress = pos_all[to_compress_mask]
            # Map position -> latent slot for compressed (non-center) tokens.
            self.sparse_layer_latent_slots_map[row_idx, pos_to_compress] = latent_slots

        for layer_idx in self.deltakv_layer_ids:
            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_cache = self.deltakv_full_kv_cache[0, l_idx]
            v_cache = self.deltakv_full_kv_cache[1, l_idx]

            # Gather KV for block tokens from raw slots.
            with profiler.record("deltakv_evict_gather_kv"):
                slots_i64 = raw_slots_block.to(torch.long)
                k_rope = k_cache[slots_i64]
                v = v_cache[slots_i64]
            with profiler.record("deltakv_evict_unrope_k"):
                cos_sin = self.cos_sin_cache[pos_all]  # (N, 1, head_dim)
                cos, sin = cos_sin.chunk(2, dim=-1)
                k_unrope = reverse_rotary_emb(k_rope, cos, sin)

            kv_dim_half = self.num_kv_heads * self.head_dim
            with profiler.record("deltakv_evict_build_kv_block"):
                kv_block = torch.cat(
                    [k_unrope.reshape(evict_len, kv_dim_half), v.reshape(evict_len, kv_dim_half)],
                    dim=-1,
                ).unsqueeze(0)  # (1, N, kv_dim)

            existing_center_slots = prev_center_slots_by_layer[layer_idx]
            # Compute top-k father indices (into all_centers) + base KV mean.
            with profiler.record("deltakv_evict_cluster"):
                topk_center_indices, base_kv = self._cluster_compress(
                    layer_idx=layer_idx,
                    kv_states=kv_block,
                    existing_center_slots=existing_center_slots,
                    cluster_step=cluster_step,
                )

            # Remap center indices -> physical slots.
            # all_center_slots = [existing centers..., new centers...]
            with profiler.record("deltakv_evict_remap_fathers"):
                all_center_slots = torch.cat([existing_center_slots, new_center_slots], dim=0)  # (M,)
                father_slots_full = all_center_slots[topk_center_indices.to(torch.long)]  # (N, K)
                father_slots = father_slots_full[to_compress_mask]  # (Nc, K)

            # Store father slots for reconstruction.
            with profiler.record("deltakv_evict_store_fathers"):
                K = self.deltakv_latent_to_full_slots.shape[-1]
                k_eff = father_slots.shape[1]
                if k_eff < K:
                    pad = father_slots[:, :1].expand(-1, K - k_eff)
                    father_slots = torch.cat([father_slots, pad], dim=1)
                elif k_eff > K:
                    father_slots = father_slots[:, :K]
                self.deltakv_latent_to_full_slots[l_idx, latent_slots] = father_slots.to(torch.int32)

            # Latent residual in compressed space.
            down = self.compress_down[l_idx]
            with profiler.record("deltakv_evict_compress_down"):
                kv_down = down(kv_block).squeeze(0)  # (N, latent_dim)
                base_down = down(base_kv).squeeze(0)
                latent_all = (kv_down - base_down)[to_compress_mask]  # (Nc, latent_dim)
            with profiler.record("deltakv_evict_store_latent"):
                self.deltakv_latent_cache[l_idx, latent_slots] = latent_all.to(self.deltakv_latent_cache.dtype)

        # Append new centers after this block is processed (so "existing" for next blocks).
        with profiler.record("deltakv_evict_append_centers"):
            for layer_idx in self.deltakv_layer_ids:
                self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                    [prev_center_slots_by_layer[layer_idx], new_center_slots], dim=0
                )

        # Free full-KV slots for non-center tokens in the evicted block.
        with profiler.record("deltakv_evict_free_full_slots"):
            free_slots = raw_slots_block[to_compress_mask].to(torch.int32)
            ptr = self._num_free_slots_deltakv_full
            self.free_slots_stack_deltakv_full[ptr: ptr + free_slots.numel()] = free_slots
            self._num_free_slots_deltakv_full += free_slots.numel()
            self.deltakv_slot_to_pos[free_slots] = -1

        # Mark compressed tokens as not having full KV anymore.
        with profiler.record("deltakv_evict_update_maps"):
            self.sparse_layer_raw_slots_map[row_idx, pos_to_compress] = -1
            # Finalized history grows by the whole evicted block (centers are also part of history).
            self.row_deltakv_compressed_lens[row_idx] += evict_len

    def deltakv_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
        chunk_lens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build DeltaKV reading view for a sparse layer and reconstruct missing KV into scratch slots.

        Args:
          layer_idx: actual model layer id
          active_compressed_indices: (B, Kmax) absolute positions, padded with -1 if needed; may be None
          context_lens: (B,) desired view lengths (sink + topk + buffer_len_total)
          req_indices: (B,) global row indices (into slot maps)
          chunk_lens: (B,) length of current chunk (prefill) or None for decode

        Returns:
          active_slots: (B, max_s) int32, Req->slots for kernels (local indexing by batch row)
          local_req_indices: (B,) int32 = arange(B)
          new_context_lens: (B,) int32 actual view lengths (sink + topk + buffer_len_total)
          temp_slots: (Nt,) int32 scratch slots to be freed after attention
        """
        with profiler.record("deltakv_reconstruct_total"):
            active_slots, local_req, new_context_lens, temp_slots, recon_pos, recon_latent, recon_out_slot = (
                self._deltakv_build_view_and_plan_reconstruct(layer_idx, active_compressed_indices, req_indices)
            )
        if temp_slots.numel() == 0:
            return active_slots, local_req, new_context_lens, temp_slots

        l_idx = self.deltakv_layer_to_idx[layer_idx]
        k_cache = self.deltakv_full_kv_cache[0, l_idx]
        v_cache = self.deltakv_full_kv_cache[1, l_idx]

        kv_dim_half = self.num_kv_heads * self.head_dim
        with profiler.record("deltakv_reconstruct_compress_up"):
            latent = self.deltakv_latent_cache[l_idx, recon_latent]  # (Nt, latent_dim)
            kv_delta = self.compress_up[l_idx](latent)  # (Nt, kv_dim)

        with profiler.record("deltakv_reconstruct_read_fathers"):
            father_slots = self.deltakv_latent_to_full_slots[l_idx, recon_latent].to(torch.int32)  # (Nt, K)
        if (father_slots < 0).any():
            raise RuntimeError("DeltaKV: missing father slots for reconstruction.")

        # Torch reconstruction (baseline).
        with profiler.record("deltakv_reconstruct_gather_fathers"):
            k_father_rope = k_cache[father_slots.to(torch.long)]  # (Nt, K, kv_heads, head_dim)
            v_father = v_cache[father_slots.to(torch.long)]
            father_pos = self.deltakv_slot_to_pos[father_slots].to(torch.long)  # (Nt, K)
        if (father_pos < 0).any():
            raise RuntimeError("DeltaKV: father center slot has unknown position.")
        with profiler.record("deltakv_reconstruct_unrope_and_mean"):
            cos_sin_f = self.cos_sin_cache[father_pos]  # (Nt, K, 1, head_dim)
            cos_f, sin_f = cos_sin_f.chunk(2, dim=-1)
            k_father_unrope = reverse_rotary_emb(k_father_rope, cos_f, sin_f)
            kv_father = torch.cat(
                [
                    k_father_unrope.reshape(k_father_unrope.shape[0], k_father_unrope.shape[1], kv_dim_half),
                    v_father.reshape(v_father.shape[0], v_father.shape[1], kv_dim_half),
                ],
                dim=-1,
            ).mean(dim=1)  # (Nt, kv_dim)

        with profiler.record("deltakv_reconstruct_apply_delta_and_rope"):
            kv_unrope = kv_delta + kv_father  # (Nt, kv_dim)
            k_unrope = kv_unrope[:, :kv_dim_half].reshape(-1, self.num_kv_heads, self.head_dim)
            v_out = kv_unrope[:, kv_dim_half:].reshape(-1, self.num_kv_heads, self.head_dim)

            cos_sin_t = self.cos_sin_cache[recon_pos]  # (Nt, 1, head_dim)
            cos_t, sin_t = cos_sin_t.chunk(2, dim=-1)
            k_out = apply_rotary_emb(k_unrope, cos_t, sin_t)

        with profiler.record("deltakv_reconstruct_writeback"):
            out_i64 = recon_out_slot.to(torch.long)
            k_cache[out_i64] = k_out.to(k_cache.dtype)
            v_cache[out_i64] = v_out.to(v_cache.dtype)

        return active_slots, local_req, new_context_lens, temp_slots

    def _deltakv_build_view_and_plan_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        req_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,  # active_slots
        torch.Tensor,  # local_req_indices
        torch.Tensor,  # new_context_lens
        torch.Tensor,  # temp_slots
        torch.Tensor,  # recon_pos (int32)
        torch.Tensor,  # recon_latent (int32)
        torch.Tensor,  # recon_out_slot (int32)
    ]:
        req_ptr = int(req_indices.data_ptr()) if req_indices is not None and req_indices.numel() > 0 else 0
        req_n = int(req_indices.numel()) if req_indices is not None else 0
        if active_compressed_indices is None:
            act_ptr = 0
            act_b = req_n
            act_k = 0
        else:
            act_ptr = int(active_compressed_indices.data_ptr()) if active_compressed_indices.numel() > 0 else int(active_compressed_indices.data_ptr())
            act_b = int(active_compressed_indices.shape[0])
            act_k = int(active_compressed_indices.shape[1])

        key = (req_ptr, req_n, act_ptr, act_b, act_k)
        if self._deltakv_view_cache_key == key and self._deltakv_view_cache_value is not None:
            with profiler.record("deltakv_build_view_cache_hit"):
                return self._deltakv_view_cache_value

        with profiler.record("deltakv_build_view_total"):
            out = self._deltakv_build_view_and_plan_reconstruct_impl(layer_idx, active_compressed_indices, req_indices)
        self._deltakv_view_cache_key = key
        self._deltakv_view_cache_value = out
        return out

    def _deltakv_build_view_and_plan_reconstruct_impl(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        req_indices: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if layer_idx in self.full_layer_to_idx:
            raise ValueError("deltakv_reconstruct should only be called for sparse layers.")
        if active_compressed_indices is None:
            active_compressed_indices = torch.empty((req_indices.numel(), 0), device="cuda", dtype=torch.int32)

        bsz = int(req_indices.numel())
        if bsz == 0:
            empty0 = torch.empty((0,), device="cuda", dtype=torch.int32)
            return torch.empty((0, 0), device="cuda", dtype=torch.int32), empty0, empty0, empty0, empty0, empty0, empty0

        local_req = torch.arange(bsz, device="cuda", dtype=torch.int32)
        sink = int(self.config.num_sink_tokens)

        with profiler.record("deltakv_build_view_read_lens"):
            req_indices_cpu = req_indices.cpu().numpy()
            # Keep per-seq lengths on CPU to avoid repeated CUDA sync via .item().
            total_lens_cpu = self.row_seq_lens[req_indices_cpu]
            compressed_lens_cpu = self.row_deltakv_compressed_lens[req_indices_cpu]

        plans: list[tuple[int, int, int, int, int, torch.Tensor]] = []
        new_context_lens_list = [0] * bsz
        max_s = 0
        with profiler.record("deltakv_build_view_plan_cpu"):
            for b in range(bsz):
                row = int(req_indices_cpu[b])
                total_len = int(total_lens_cpu[b])
                sink_len = min(sink, total_len)

                comp_len = int(compressed_lens_cpu[b]) if total_len > sink else 0
                comp_len = min(comp_len, max(0, total_len - sink))

                buffer_start = (sink + comp_len) if total_len > sink else sink_len
                buffer_len = total_len - buffer_start
                if buffer_len < 0:
                    raise RuntimeError("DeltaKV: negative buffer length; compressed_lens is inconsistent.")

                if active_compressed_indices.numel() == 0 or total_len <= sink or comp_len <= 0:
                    top_pos = torch.empty((0,), device="cuda", dtype=torch.int32)
                else:
                    cand = active_compressed_indices[b].to(torch.int32)
                    valid = (cand >= sink) & (cand < sink + comp_len) & (cand < total_len)
                    top_pos = cand[valid]

                k_b = int(top_pos.numel())
                ctx_len_b = sink_len + k_b + buffer_len
                new_context_lens_list[b] = ctx_len_b
                max_s = max(max_s, ctx_len_b)
                plans.append((row, total_len, sink_len, buffer_start, buffer_len, top_pos))

        new_context_lens = torch.tensor(new_context_lens_list, device="cuda", dtype=torch.int32)

        with profiler.record("deltakv_build_view_alloc_active_slots"):
            active_slots = torch.zeros((bsz, max_s), device="cuda", dtype=torch.int32)

        temp_slots_all = []
        recon_pos = []
        recon_latent = []
        recon_out_slot = []
        with profiler.record("deltakv_build_view_fill_and_alloc_temp"):
            for b, (row, _total_len, sink_len, buffer_start, buffer_len, top_pos) in enumerate(plans):
                if sink_len > 0:
                    sink_slots = self.sparse_layer_raw_slots_map[row, :sink_len].to(torch.int32)
                    if (sink_slots < 0).any():
                        raise RuntimeError("DeltaKV: missing full slots in sink window.")
                    active_slots[b, :sink_len] = sink_slots

                k_b = int(top_pos.numel())
                if k_b > 0:
                    top_slots = self.sparse_layer_raw_slots_map[row, top_pos.to(torch.long)].to(torch.int32)
                    need = top_slots < 0
                    if need.any():
                        latent_slots = self.sparse_layer_latent_slots_map[row, top_pos[need].to(torch.long)].to(torch.int32)
                        if (latent_slots < 0).any():
                            raise RuntimeError("DeltaKV: selected token has neither full slot nor latent slot.")
                        out_slots = self._allocate_temp_deltakv_full(int(need.sum().item())).to(torch.int32)
                        top_slots[need] = out_slots
                        temp_slots_all.append(out_slots)

                        recon_pos.append(top_pos[need].to(torch.int32))
                        recon_latent.append(latent_slots)
                        recon_out_slot.append(out_slots)

                    active_slots[b, sink_len: sink_len + k_b] = top_slots

                if buffer_len > 0:
                    buf_slots = self.sparse_layer_raw_slots_map[row, buffer_start: buffer_start + buffer_len].to(torch.int32)
                    if (buf_slots < 0).any():
                        raise RuntimeError("DeltaKV: buffer contains missing full slots.")
                    active_slots[b, sink_len + k_b: sink_len + k_b + buffer_len] = buf_slots

        if not temp_slots_all:
            empty = torch.empty((0,), device="cuda", dtype=torch.int32)
            return active_slots, local_req, new_context_lens, empty, empty, empty, empty

        with profiler.record("deltakv_build_view_pack_recon"):
            recon_pos = torch.cat(recon_pos, dim=0).to(torch.int32)
            recon_latent = torch.cat(recon_latent, dim=0).to(torch.int32)
            recon_out_slot = torch.cat(recon_out_slot, dim=0).to(torch.int32)
            temp_slots = torch.cat(temp_slots_all, dim=0).to(torch.int32)
        return active_slots, local_req, new_context_lens, temp_slots, recon_pos, recon_latent, recon_out_slot


# For now, keep deltakv-triton as a correctness-first alias.
class DeltaKVCacheTritonManager(DeltaKVCacheManager):
    def _deltakv_gather_kv_unrope(
        self,
        *,
        slots: torch.Tensor,
        pos: torch.Tensor,
        cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        from sparsevllm.triton_kernel.deltakv_kernels import deltakv_gather_kv_unrope

        return deltakv_gather_kv_unrope(
            slots=slots,
            pos=pos,
            cos_sin=cos_sin,
            k_cache=k_cache,
            v_cache=v_cache,
        )

    def _deltakv_reconstruct_writeback(
        self,
        *,
        kv_delta: torch.Tensor,
        father_slots: torch.Tensor,
        slot_to_pos: torch.Tensor,
        out_slots: torch.Tensor,
        out_pos: torch.Tensor,
        cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        from sparsevllm.triton_kernel.deltakv_kernels import deltakv_reconstruct_writeback

        return deltakv_reconstruct_writeback(
            kv_delta=kv_delta,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_cache,
            v_cache=v_cache,
        )

    @torch.no_grad()
    def deltakv_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
        chunk_lens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with profiler.record("deltakv_reconstruct_triton_total"):
            active_slots, local_req, new_context_lens, temp_slots, recon_pos, recon_latent, recon_out_slot = (
                self._deltakv_build_view_and_plan_reconstruct(layer_idx, active_compressed_indices, req_indices)
            )
            if temp_slots.numel() == 0:
                return active_slots, local_req, new_context_lens, temp_slots

            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_cache = self.deltakv_full_kv_cache[0, l_idx]
            v_cache = self.deltakv_full_kv_cache[1, l_idx]

            with profiler.record("deltakv_reconstruct_triton_compress_up"):
                latent = self.deltakv_latent_cache[l_idx, recon_latent]  # (Nt, latent_dim)
                kv_delta = self.compress_up[l_idx](latent)  # (Nt, kv_dim) in de-RoPE space for K

            with profiler.record("deltakv_reconstruct_triton_read_fathers"):
                father_slots = self.deltakv_latent_to_full_slots[l_idx, recon_latent].to(torch.int32)  # (Nt, K)
            if (father_slots < 0).any():
                raise RuntimeError("DeltaKV: missing father slots for reconstruction.")

            # cos_sin_cache: (max_pos, 1, head_dim) -> (max_pos, head_dim)
            cos_sin = self.cos_sin_cache[:, 0, :]

            with profiler.record("deltakv_reconstruct_triton_kernel"):
                self._deltakv_reconstruct_writeback(
                    kv_delta=kv_delta,
                    father_slots=father_slots,
                    slot_to_pos=self.deltakv_slot_to_pos,
                    out_slots=recon_out_slot,
                    out_pos=recon_pos,
                    cos_sin=cos_sin,
                    k_cache=k_cache,
                    v_cache=v_cache,
                )

            return active_slots, local_req, new_context_lens, temp_slots


class DeltaKVCacheTritonManagerV2(DeltaKVCacheTritonManager):
    def _gather_kv_unrope_by_slots(
        self,
        layer_idx: int,
        slots: torch.Tensor,
    ) -> torch.Tensor:
        with profiler.record("deltakv_gather_unrope_total"):
            if slots.numel() == 0:
                return torch.empty(
                    (0, 2 * self.num_kv_heads * self.head_dim),
                    device="cuda",
                    dtype=self.hf_config.torch_dtype,
                )

            assert layer_idx in self.deltakv_layer_to_idx
            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_cache = self.deltakv_full_kv_cache[0, l_idx]
            v_cache = self.deltakv_full_kv_cache[1, l_idx]

            slots_i32 = slots.to(torch.int32)
            pos = self.deltakv_slot_to_pos[slots_i32].to(torch.int32)
            if (pos < 0).any():
                raise RuntimeError("DeltaKV: center slot has unknown position (deltakv_slot_to_pos == -1).")

            cos_sin = self.cos_sin_cache[:, 0, :]
            with profiler.record("deltakv_gather_unrope_triton_kernel"):
                return self._deltakv_gather_kv_unrope(
                    slots=slots_i32,
                    pos=pos,
                    cos_sin=cos_sin,
                    k_cache=k_cache,
                    v_cache=v_cache,
                )

    @torch.no_grad()
    def deltakv_evict(self, seqs: list[Sequence]):
        with profiler.record("deltakv_evict_triton_total"):
            if not self.deltakv_layer_ids:
                return
            sink = int(self.config.num_sink_tokens)
            recent = int(self.config.num_recent_tokens)
            cluster_step = max(1, int(1.0 / max(1e-6, float(self.config.cluster_ratio))))
            cos_sin = self.cos_sin_cache[:, 0, :]

            for seq in seqs:
                with profiler.record("deltakv_evict_triton_seq"):
                    row_idx = self.seq_id_to_row.get(seq.seq_id, None)
                    if row_idx is None:
                        continue

                    total_len = int(self.row_seq_lens[row_idx])
                    compressed_len = int(self.row_deltakv_compressed_lens[row_idx])
                    buffer_start = sink + compressed_len
                    buffer_len = total_len - buffer_start
                    if buffer_len <= recent:
                        continue

                    evict_len = ((buffer_len - recent) // recent) * recent
                    if evict_len <= 0:
                        continue

                    evict_start = buffer_start
                    evict_end = evict_start + evict_len

                    with profiler.record("deltakv_evict_triton_read_slots"):
                        raw_slots_block = self.sparse_layer_raw_slots_map[row_idx, evict_start:evict_end].clone()
                    if (raw_slots_block < 0).any():
                        raise RuntimeError("DeltaKV eviction expects raw slots for the buffer block.")

                    with profiler.record("deltakv_evict_triton_select_centers"):
                        center_rel = torch.arange(0, evict_len, cluster_step, device="cuda", dtype=torch.long)
                        new_center_slots = raw_slots_block[center_rel].to(torch.int32)

                    with profiler.record("deltakv_evict_triton_prev_centers"):
                        sink_slots = self.sparse_layer_raw_slots_map[row_idx, :sink].to(torch.int32)
                        prev_center_slots_by_layer: dict[int, torch.Tensor] = {}
                        for layer_idx in self.deltakv_layer_ids:
                            existing = self.row_deltakv_center_slots[row_idx][layer_idx]
                            prev_center_slots_by_layer[layer_idx] = (
                                sink_slots if existing is None else existing.to(torch.int32)
                            )

                    with profiler.record("deltakv_evict_triton_build_masks"):
                        is_center = torch.zeros((evict_len,), device="cuda", dtype=torch.bool)
                        is_center[center_rel] = True
                        to_compress_mask = ~is_center
                        num_to_compress = int(to_compress_mask.sum().item())
                    if num_to_compress <= 0:
                        with profiler.record("deltakv_evict_triton_append_centers_only"):
                            for layer_idx in self.deltakv_layer_ids:
                                self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                                    [prev_center_slots_by_layer[layer_idx], new_center_slots], dim=0
                                )
                            self.row_deltakv_compressed_lens[row_idx] += evict_len
                        continue

                    with profiler.record("deltakv_evict_triton_alloc_latent"):
                        latent_slots = self._allocate_deltakv_latent(num_to_compress).to(torch.int32)
                        pos_all = torch.arange(evict_start, evict_end, device="cuda", dtype=torch.int32)
                        pos_to_compress = pos_all[to_compress_mask]
                        self.sparse_layer_latent_slots_map[row_idx, pos_to_compress.to(torch.long)] = latent_slots

                    raw_slots_block_i32 = raw_slots_block.to(torch.int32)

                    for layer_idx in self.deltakv_layer_ids:
                        l_idx = self.deltakv_layer_to_idx[layer_idx]
                        k_cache = self.deltakv_full_kv_cache[0, l_idx]
                        v_cache = self.deltakv_full_kv_cache[1, l_idx]

                        with profiler.record("deltakv_evict_triton_gather_unrope"):
                            kv_block = self._deltakv_gather_kv_unrope(
                                slots=raw_slots_block_i32,
                                pos=pos_all,
                                cos_sin=cos_sin,
                                k_cache=k_cache,
                                v_cache=v_cache,
                            ).unsqueeze(0)  # (1, N, kv_dim)

                        existing_center_slots = prev_center_slots_by_layer[layer_idx]
                        with profiler.record("deltakv_evict_triton_cluster"):
                            topk_center_indices, base_kv = self._cluster_compress(
                                layer_idx=layer_idx,
                                kv_states=kv_block,
                                existing_center_slots=existing_center_slots,
                                cluster_step=cluster_step,
                            )

                        with profiler.record("deltakv_evict_triton_remap_fathers"):
                            all_center_slots = torch.cat([existing_center_slots, new_center_slots], dim=0)  # (M,)
                            father_slots_full = all_center_slots[topk_center_indices.to(torch.long)]  # (N, K)
                            father_slots = father_slots_full[to_compress_mask]  # (Nc, K)

                        with profiler.record("deltakv_evict_triton_store_fathers"):
                            K = self.deltakv_latent_to_full_slots.shape[-1]
                            k_eff = father_slots.shape[1]
                            if k_eff < K:
                                pad = father_slots[:, :1].expand(-1, K - k_eff)
                                father_slots = torch.cat([father_slots, pad], dim=1)
                            elif k_eff > K:
                                father_slots = father_slots[:, :K]
                            self.deltakv_latent_to_full_slots[l_idx, latent_slots] = father_slots.to(torch.int32)

                        down = self.compress_down[l_idx]
                        with profiler.record("deltakv_evict_triton_compress_down"):
                            kv_down = down(kv_block).squeeze(0)  # (N, latent_dim)
                            base_down = down(base_kv).squeeze(0)
                            latent_all = (kv_down - base_down)[to_compress_mask]  # (Nc, latent_dim)
                        with profiler.record("deltakv_evict_triton_store_latent"):
                            self.deltakv_latent_cache[l_idx, latent_slots] = latent_all.to(self.deltakv_latent_cache.dtype)

                    with profiler.record("deltakv_evict_triton_append_centers"):
                        for layer_idx in self.deltakv_layer_ids:
                            self.row_deltakv_center_slots[row_idx][layer_idx] = torch.cat(
                                [prev_center_slots_by_layer[layer_idx], new_center_slots], dim=0
                            )

                    with profiler.record("deltakv_evict_triton_free_full_slots"):
                        free_slots = raw_slots_block_i32[to_compress_mask]
                        ptr = self._num_free_slots_deltakv_full
                        self.free_slots_stack_deltakv_full[ptr: ptr + free_slots.numel()] = free_slots
                        self._num_free_slots_deltakv_full += free_slots.numel()
                        self.deltakv_slot_to_pos[free_slots] = -1

                    with profiler.record("deltakv_evict_triton_update_maps"):
                        self.sparse_layer_raw_slots_map[row_idx, pos_to_compress.to(torch.long)] = -1
                        self.row_deltakv_compressed_lens[row_idx] += evict_len


class DeltaKVCacheTritonManagerV3(DeltaKVCacheTritonManagerV2):
    def _cluster_compress(
        self,
        layer_idx: int,
        kv_states: torch.Tensor,  # (1, N, kv_dim), de-RoPE already applied on K
        existing_center_slots: torch.Tensor,  # (M0,)
        cluster_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V3: fuse L2 score + causal mask + per-block topk in Triton.

        Key idea: avoid materializing the full (N, M) score matrix. We compute top-k
        within each center block in Triton, then merge candidates with a much smaller
        `torch.topk` on (N, num_blocks*K).
        """
        metric_type = self.config.cluster_metric
        if metric_type != "l2":
            return super()._cluster_compress(layer_idx, kv_states, existing_center_slots, cluster_step)

        assert kv_states.dim() == 3 and kv_states.shape[0] == 1
        _, n, kv_dim = kv_states.shape
        k_neighbors = int(self.config.deltakv_k_neighbors)

        with profiler.record("deltakv_cluster_existing_centers"):
            existing_centers = (
                self._gather_kv_unrope_by_slots(layer_idx, existing_center_slots).unsqueeze(0)
                if existing_center_slots.numel() > 0
                else kv_states.new_zeros((1, 0, kv_dim))
            )
        new_centers = kv_states[:, ::cluster_step, :]
        all_centers = torch.cat([existing_centers, new_centers], dim=1)  # (1, M, kv_dim)
        m0 = int(existing_centers.shape[1])
        M = int(all_centers.shape[1])

        k_eff = min(k_neighbors, M)
        if k_eff <= 0:
            raise RuntimeError("DeltaKV: no available centers to assign.")
        # Small M: torch is fine.
        if M < 128 or kv_states.dtype not in (torch.float16, torch.bfloat16):
            return super()._cluster_compress(layer_idx, kv_states, existing_center_slots, cluster_step)

        from sparsevllm.triton_kernel.deltakv_kernels import deltakv_l2_topk_blockwise

        with profiler.record("deltakv_cluster_metric"):
            partial_scores, partial_idx = deltakv_l2_topk_blockwise(
                tokens=kv_states[0],
                centers=all_centers[0],
                m0=m0,
                cluster_step=int(cluster_step),
                k=k_eff,
            )

        # Merge candidates across blocks: (N, MB*K) -> topk(K).
        with profiler.record("deltakv_cluster_topk"):
            NB, MB, BN, KK = partial_scores.shape
            cand_scores = partial_scores.permute(0, 2, 1, 3).reshape(NB * BN, MB * KK)[:n]
            cand_idx = partial_idx.permute(0, 2, 1, 3).reshape(NB * BN, MB * KK)[:n]
            merge_pos = cand_scores.topk(k=k_eff, dim=1).indices
            topk_indices_i32 = cand_idx.gather(1, merge_pos)  # (N, K) int32

        with profiler.record("deltakv_cluster_gather_mean"):
            topk_i64 = topk_indices_i32.to(torch.long).unsqueeze(0)  # (1, N, K)
            gather_idx = topk_i64.view(1, -1)[:, :, None].expand(-1, -1, kv_dim)
            gathered = all_centers.gather(1, gather_idx).view(1, n, k_eff, kv_dim).mean(dim=2)

        return topk_indices_i32.to(torch.int32), gathered


class DeltaKVCacheTritonManagerV4(DeltaKVCacheTritonManagerV3):
    """DeltaKV V4: extra Triton fusions for inference hot paths.

    - Reconstruction: grouped-head writeback kernel to reduce redundant loads of
      `cos_sin` / `slot_to_pos` across KV heads.
    - Eviction clustering: use a Triton gather+mean kernel to avoid materializing
      large `torch.gather` intermediates.
    """

    def _deltakv_gather_kv_unrope(
        self,
        *,
        slots: torch.Tensor,
        pos: torch.Tensor,
        cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ) -> torch.Tensor:
        from sparsevllm.triton_kernel.deltakv_kernels import deltakv_gather_kv_unrope_grouped_heads

        hp = int(getattr(self.config, "deltakv_triton_gather_heads_per_program", 4) or 1)
        hp = max(1, min(hp, int(self.num_kv_heads)))
        return deltakv_gather_kv_unrope_grouped_heads(
            slots=slots,
            pos=pos,
            cos_sin=cos_sin,
            k_cache=k_cache,
            v_cache=v_cache,
            heads_per_program=hp,
        )

    def _deltakv_reconstruct_writeback(
        self,
        *,
        kv_delta: torch.Tensor,
        father_slots: torch.Tensor,
        slot_to_pos: torch.Tensor,
        out_slots: torch.Tensor,
        out_pos: torch.Tensor,
        cos_sin: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
    ):
        from sparsevllm.triton_kernel.deltakv_kernels import deltakv_reconstruct_writeback_grouped_heads

        hp = int(getattr(self.config, "deltakv_triton_reconstruct_heads_per_program", 4) or 1)
        hp = max(1, min(hp, int(self.num_kv_heads)))
        return deltakv_reconstruct_writeback_grouped_heads(
            kv_delta=kv_delta,
            father_slots=father_slots,
            slot_to_pos=slot_to_pos,
            out_slots=out_slots,
            out_pos=out_pos,
            cos_sin=cos_sin,
            k_cache=k_cache,
            v_cache=v_cache,
            heads_per_program=hp,
        )

    def _cluster_compress(
        self,
        layer_idx: int,
        kv_states: torch.Tensor,  # (1, N, kv_dim), de-RoPE already applied on K
        existing_center_slots: torch.Tensor,  # (M0,)
        cluster_step: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V4: reuse V3's blockwise L2-topk, but fuse gather+mean in Triton."""
        metric_type = self.config.cluster_metric
        if metric_type != "l2":
            return super()._cluster_compress(layer_idx, kv_states, existing_center_slots, cluster_step)

        assert kv_states.dim() == 3 and kv_states.shape[0] == 1
        _, n, kv_dim = kv_states.shape
        k_neighbors = int(self.config.deltakv_k_neighbors)

        with profiler.record("deltakv_cluster_existing_centers"):
            existing_centers = (
                self._gather_kv_unrope_by_slots(layer_idx, existing_center_slots).unsqueeze(0)
                if existing_center_slots.numel() > 0
                else kv_states.new_zeros((1, 0, kv_dim))
            )
        new_centers = kv_states[:, ::cluster_step, :]
        all_centers = torch.cat([existing_centers, new_centers], dim=1)  # (1, M, kv_dim)
        m0 = int(existing_centers.shape[1])
        M = int(all_centers.shape[1])

        k_eff = min(k_neighbors, M)
        if k_eff <= 0:
            raise RuntimeError("DeltaKV: no available centers to assign.")
        # Small M: torch is fine.
        if M < 128 or kv_states.dtype not in (torch.float16, torch.bfloat16):
            return super()._cluster_compress(layer_idx, kv_states, existing_center_slots, cluster_step)

        from sparsevllm.triton_kernel.deltakv_kernels import batch_gather_mean, deltakv_l2_topk_blockwise

        with profiler.record("deltakv_cluster_metric"):
            partial_scores, partial_idx = deltakv_l2_topk_blockwise(
                tokens=kv_states[0],
                centers=all_centers[0],
                m0=m0,
                cluster_step=int(cluster_step),
                k=k_eff,
            )

        # Merge candidates across blocks: (N, MB*K) -> topk(K).
        with profiler.record("deltakv_cluster_topk"):
            NB, MB, BN, KK = partial_scores.shape
            cand_scores = partial_scores.permute(0, 2, 1, 3).reshape(NB * BN, MB * KK)[:n]
            cand_idx = partial_idx.permute(0, 2, 1, 3).reshape(NB * BN, MB * KK)[:n]
            merge_pos = cand_scores.topk(k=k_eff, dim=1).indices
            topk_indices_i32 = cand_idx.gather(1, merge_pos)  # (N, K) int32

        with profiler.record("deltakv_cluster_gather_mean"):
            gathered = batch_gather_mean(all_centers[0], topk_indices_i32.unsqueeze(0))  # (1, N, kv_dim)

        return topk_indices_i32.to(torch.int32), gathered


class DeltaKVCacheTritonManagerV3WithOffload(DeltaKVCacheTritonManagerV3):
    """DeltaKV V3 with CPU offload for latent cache.

    - Keep full KV pools + centers on GPU (unchanged).
    - Store `deltakv_latent_cache` on CPU for most sparse layers to save VRAM.
    - Optionally keep the first N sparse layers after each observation layer on GPU,
      so their attention compute can overlap with prefetch for later layers.
    - Reuses existing Triton reconstruction kernel (`deltakv_reconstruct_writeback`).
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        # `DeltaKVCacheManager.__init__` will call our overridden `allocate_kv_cache()`.
        super().__init__(config, rank, world_size)
        cpu_threads = int(getattr(config, "deltakv_offload_cpu_threads", 0) or 0)
        if cpu_threads > 0:
            torch.set_num_threads(cpu_threads)
        self._offload_prefetch_stream = torch.cuda.Stream()
        self._offload_prefetch_cache: dict[int, tuple[tuple[int, int], torch.Tensor, torch.Tensor, torch.cuda.Event]] = {}
        # (layer_idx -> (latent_key, host_pinned, gpu_latent, event))

    def _deltakv_reset_view_cache(self):
        super()._deltakv_reset_view_cache()
        self._offload_prefetch_cache.clear()

    def allocate_kv_cache(self):
        # Copy of DeltaKVCacheManager.allocate_kv_cache(), but exclude latent bytes from the GPU budget
        # when `deltakv_offload_latent=True`.
        available_memory, slot_bytes_per_layer = self._get_available_slots_info()
        config = self.config
        dtype_size = torch.tensor([], dtype=self.hf_config.torch_dtype).element_size()

        num_full_layers = len(self.full_layer_ids)
        num_deltakv_layers = len(self.deltakv_layer_ids)
        assert num_full_layers > 0, "DeltaKV requires at least one full-attention layer."
        assert num_deltakv_layers > 0, "DeltaKV requires at least one sparse layer."

        # Full layers store all tokens. Sparse layers store:
        # - a bounded full-KV pool: centers + uncompressed buffer (+ current chunk) + reconstructed top tokens.
        # Latent cache is offloaded to CPU (not counted in GPU budget).
        cluster_ratio = max(0.0, float(config.cluster_ratio))
        per_token_bytes = num_full_layers * slot_bytes_per_layer + num_deltakv_layers * (cluster_ratio * slot_bytes_per_layer)
        if per_token_bytes <= 0:
            raise ValueError("Invalid KV cache allocation configuration.")

        max_tokens = max(1, int(available_memory / per_token_bytes))
        reserve_ratio = float(getattr(config, "deltakv_full_pool_reserve_ratio", 0.0))
        if reserve_ratio > 0:
            reserve_ratio = max(0.0, min(0.5, reserve_ratio))
            max_tokens = max(1, int(max_tokens * (1.0 - reserve_ratio)))

        self.full_num_slots = max_tokens
        self.deltakv_latent_num_slots = max_tokens

        bytes_full_layers = self.full_num_slots * num_full_layers * slot_bytes_per_layer
        bytes_misc = 0
        bytes_left = available_memory - bytes_full_layers - bytes_misc
        if bytes_left <= 0:
            raise RuntimeError(
                "Not enough GPU memory left for DeltaKV full-KV pool after allocating full layers. "
                "Try reducing max_model_len / gpu_memory_utilization."
            )
        max_deltakv_full_slots = max(1, int(bytes_left // (num_deltakv_layers * slot_bytes_per_layer)))

        sink = int(config.num_sink_tokens)
        recent = int(config.num_recent_tokens)
        max_seqs = int(getattr(config, "max_num_seqs_in_batch", 1) or 1)
        top_decode = int(config.num_top_tokens)
        top_prefill = int(getattr(config, "num_top_tokens_in_prefill", config.num_top_tokens) or config.num_top_tokens)
        max_prefill_seqs_by_tokens = (int(config.max_num_batched_tokens) + int(config.chunk_prefill_size) - 1) // int(
            config.chunk_prefill_size
        )
        max_prefill_seqs = min(max_seqs, max_prefill_seqs_by_tokens)
        total_top_slots = max(max_seqs * top_decode, max_prefill_seqs * top_prefill)
        max_step_chunk = int(min(int(config.max_num_batched_tokens), max_seqs * int(config.chunk_prefill_size)))
        overhead_slots = max_seqs * (sink + 2 * recent) + total_top_slots + max_step_chunk
        if max_deltakv_full_slots <= overhead_slots:
            raise RuntimeError(
                f"DeltaKV full-KV pool too small: max={max_deltakv_full_slots}, required>={overhead_slots + 1}. "
                "Reduce chunk_prefill_size/num_top_tokens/num_recent_tokens or increase gpu_memory_utilization."
            )

        desired_centers = max(1, int(cluster_ratio * self.full_num_slots * 1.5))
        centers_capacity = min(desired_centers, max_deltakv_full_slots - overhead_slots)
        self.deltakv_full_num_slots = overhead_slots + centers_capacity
        self._deltakv_centers_capacity = int(centers_capacity)
        self._deltakv_temp_full_reserve = min(self.deltakv_full_num_slots, int(total_top_slots))

        logger.info(
            f"DeltaKV offload allocation: full_layers_slots={self.full_num_slots}; "
            f"deltakv_full_slots={self.deltakv_full_num_slots} (overhead={overhead_slots}, centers={centers_capacity}); "
            f"deltakv_latent_slots={self.deltakv_latent_num_slots} (CPU offload) "
            f"(full_layers={num_full_layers}, deltakv_layers={num_deltakv_layers}, "
            f"deltakv_full_pool_reserve_ratio={reserve_ratio:.3f}, "
            f"deltakv_temp_full_reserve={self._deltakv_temp_full_reserve})."
        )

        self.full_kv_cache = torch.empty(
            2,
            num_full_layers,
            self.full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )

        self.deltakv_full_kv_cache = torch.empty(
            2,
            num_deltakv_layers,
            self.deltakv_full_num_slots,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.hf_config.torch_dtype,
            device="cuda",
        )

        # Father mapping stays on GPU (small-ish, int32).
        self.deltakv_latent_to_full_slots = torch.full(
            (num_deltakv_layers, self.deltakv_latent_num_slots, config.deltakv_k_neighbors),
            -1,
            dtype=torch.int32,
            device="cuda",
        )
        self.deltakv_slot_to_pos = torch.full(
            (self.deltakv_full_num_slots,),
            -1,
            dtype=torch.int32,
            device="cuda",
        )

        # Decide which sparse layers keep latents on GPU.
        keep_after = int(getattr(config, "deltakv_offload_keep_after_obs_layers", 0) or 0)
        keep_layer_ids: set[int] = set()
        if keep_after > 0:
            obs_layers = list(getattr(config, "obs_layer_ids", []) or [])
            full_layers = set(getattr(config, "full_attn_layers", []) or [])
            for obs in obs_layers:
                cnt = 0
                for l in range(int(obs) + 1, self.num_layers):
                    if l in full_layers:
                        break
                    keep_layer_ids.add(l)
                    cnt += 1
                    if cnt >= keep_after:
                        break

        keep_lidx = sorted([self.deltakv_layer_to_idx[l] for l in keep_layer_ids if l in self.deltakv_layer_to_idx])

        latent_dim = int(config.kv_compressed_size)
        if keep_lidx:
            # Guardrail: keeping latents on GPU is extremely memory-hungry (O(num_slots)).
            # Fit the requested keep set into currently available GPU memory to avoid init-time OOM.
            free_now, _total = torch.cuda.mem_get_info()
            bytes_per_keep_layer = int(self.deltakv_latent_num_slots) * int(latent_dim) * int(dtype_size)
            reserve_bytes = 1024**3  # leave ~1GiB headroom for runtime allocations
            keep_budget = max(0, int(free_now) - int(reserve_bytes))
            max_keep_layers = 0 if bytes_per_keep_layer <= 0 else int(keep_budget // bytes_per_keep_layer)
            if max_keep_layers <= 0:
                logger.warning(
                    "DeltaKV offload: disabling GPU latent keep to avoid OOM "
                    f"(requested_layers={len(keep_lidx)}, free={free_now / 1024**3:.2f}GiB, "
                    f"per_layer={bytes_per_keep_layer / 1024**3:.2f}GiB)."
                )
                keep_lidx = []
            elif max_keep_layers < len(keep_lidx):
                logger.warning(
                    "DeltaKV offload: trimming GPU latent keep to fit memory "
                    f"(requested_layers={len(keep_lidx)}, keeping={max_keep_layers}, "
                    f"free={free_now / 1024**3:.2f}GiB, per_layer={bytes_per_keep_layer / 1024**3:.2f}GiB)."
                )
                keep_lidx = keep_lidx[:max_keep_layers]

        self._deltakv_latent_keep_gpu_lidx = set(keep_lidx)
        self._deltakv_latent_gpu_map: dict[int, int] = {lidx: i for i, lidx in enumerate(keep_lidx)}

        if keep_lidx:
            self.deltakv_latent_cache_gpu = torch.empty(
                (len(keep_lidx), self.deltakv_latent_num_slots, latent_dim),
                dtype=self.hf_config.torch_dtype,
                device="cuda",
            )
        else:
            self.deltakv_latent_cache_gpu = None

        # CPU latent cache for offloaded layers only.
        self.deltakv_latent_cache_cpu_layers: list[torch.Tensor | None] = [None for _ in range(num_deltakv_layers)]
        for lidx in range(num_deltakv_layers):
            if lidx in self._deltakv_latent_keep_gpu_lidx:
                continue
            self.deltakv_latent_cache_cpu_layers[lidx] = self._create_cpu_latent_layer(
                lidx=lidx,
                num_slots=self.deltakv_latent_num_slots,
                latent_dim=latent_dim,
            )

        # Keep an attribute for compatibility; do not use directly in offload mode.
        self.deltakv_latent_cache = None

    def _create_cpu_latent_layer(self, *, lidx: int, num_slots: int, latent_dim: int) -> torch.Tensor:
        # Default: pageable CPU tensor (faster to allocate, slower for transfer).
        return torch.empty(
            (num_slots, latent_dim),
            dtype=self.hf_config.torch_dtype,
            device="cpu",
        )

    def _store_latent_offload(
        self,
        *,
        l_idx: int,
        latent_slots_cpu_i64: torch.Tensor,
        latent_all: torch.Tensor,
    ) -> None:
        cpu_layer = self.deltakv_latent_cache_cpu_layers[l_idx]
        if cpu_layer is None:
            raise RuntimeError("DeltaKV offload: missing CPU latent cache for this layer.")
        host = torch.empty(
            (latent_all.shape[0], latent_all.shape[1]),
            dtype=cpu_layer.dtype,
            pin_memory=True,
        )
        host.copy_(latent_all.to(host.dtype), non_blocking=False)
        cpu_layer.index_copy_(0, latent_slots_cpu_i64, host)

    def _latent_key(self, recon_latent: torch.Tensor) -> tuple[int, int]:
        return (int(recon_latent.data_ptr()), int(recon_latent.numel()))

    def _prefetch_latents_to_gpu(
        self,
        *,
        layer_idx: int,
        recon_latent: torch.Tensor,
        idx_cpu_i64: torch.Tensor,
    ):
        l_idx = self.deltakv_layer_to_idx[layer_idx]
        if l_idx in self._deltakv_latent_keep_gpu_lidx:
            return
        cpu_layer = self.deltakv_latent_cache_cpu_layers[l_idx]
        if cpu_layer is None:
            raise RuntimeError("DeltaKV offload: missing CPU latent cache for this layer.")

        latent_key = self._latent_key(recon_latent)
        # Allocate pinned host buffer for async H2D.
        host = torch.empty((idx_cpu_i64.numel(), cpu_layer.shape[1]), dtype=cpu_layer.dtype, pin_memory=True)
        # Default to index_select (often faster for large tensors); allow opt-in to advanced indexing
        # via env var for benchmarking.
        if os.getenv("USE_ADVSEL"):
            host.copy_(cpu_layer[idx_cpu_i64], non_blocking=False)
        else:
            host.copy_(cpu_layer.index_select(0, idx_cpu_i64), non_blocking=False)

        with torch.cuda.stream(self._offload_prefetch_stream):
            gpu_latent = host.to(device="cuda", non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()

        self._offload_prefetch_cache[layer_idx] = (latent_key, host, gpu_latent, ev)

    def _get_gpu_latents_for_layer(
        self,
        *,
        layer_idx: int,
        recon_latent: torch.Tensor,
    ) -> torch.Tensor:
        """Return a CUDA tensor (Nt, latent_dim) for the given layer and latent ids."""
        l_idx = self.deltakv_layer_to_idx[layer_idx]
        idx_i64 = recon_latent.to(torch.long)

        if l_idx in self._deltakv_latent_keep_gpu_lidx:
            assert self.deltakv_latent_cache_gpu is not None
            gidx = self._deltakv_latent_gpu_map[l_idx]
            return self.deltakv_latent_cache_gpu[gidx].index_select(0, idx_i64)

        latent_key = self._latent_key(recon_latent)
        cached = self._offload_prefetch_cache.get(layer_idx, None)
        if cached is not None and cached[0] == latent_key:
            _, _host, gpu_latent, ev = cached
            torch.cuda.current_stream().wait_event(ev)
            return gpu_latent

        # Fallback: synchronous gather on CPU + async H2D, then wait.
        cpu_layer = self.deltakv_latent_cache_cpu_layers[l_idx]
        if cpu_layer is None:
            raise RuntimeError("DeltaKV offload: missing CPU latent cache for this layer.")
        idx_cpu = idx_i64.to(device="cpu", non_blocking=False)
        host = torch.empty((idx_cpu.numel(), cpu_layer.shape[1]), dtype=cpu_layer.dtype, pin_memory=True)
        if os.getenv("USE_ADVSEL"):
            host.copy_(cpu_layer[idx_cpu], non_blocking=False)
        else:
            host.copy_(cpu_layer.index_select(0, idx_cpu), non_blocking=False)

        with torch.cuda.stream(self._offload_prefetch_stream):
            gpu_latent = host.to(device="cuda", non_blocking=True)
            ev = torch.cuda.Event()
            ev.record()
        torch.cuda.current_stream().wait_event(ev)
        return gpu_latent

    @torch.no_grad()
    def deltakv_reconstruct(
        self,
        layer_idx: int,
        active_compressed_indices: torch.Tensor | None,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor,
        chunk_lens: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with profiler.record("deltakv_reconstruct_triton_total"):
            active_slots, local_req, new_context_lens, temp_slots, recon_pos, recon_latent, recon_out_slot = (
                self._deltakv_build_view_and_plan_reconstruct(layer_idx, active_compressed_indices, req_indices)
            )
            if temp_slots.numel() == 0:
                return active_slots, local_req, new_context_lens, temp_slots

            from sparsevllm.triton_kernel.deltakv_kernels import deltakv_reconstruct_writeback

            # Precompute CPU indices once (shared across layers in the segment).
            idx_cpu_i64 = recon_latent.to(torch.long).to(device="cpu", non_blocking=False)

            # Prefetch for future layers (1..N ahead) to overlap with attention compute.
            prefetch_n = int(getattr(self.config, "deltakv_offload_prefetch_distance", 1) or 0)
            if prefetch_n > 0:
                # Find next sparse layers until next full-attn layer.
                full_layers = set(self.full_attn_layers)
                scheduled = 0
                for nxt in range(layer_idx + 1, self.num_layers):
                    if nxt in full_layers:
                        break
                    if nxt == layer_idx:
                        continue
                    # Only prefetch for offloaded sparse layers.
                    if nxt in self.deltakv_layer_to_idx and self.deltakv_layer_to_idx[nxt] not in self._deltakv_latent_keep_gpu_lidx:
                        self._prefetch_latents_to_gpu(layer_idx=nxt, recon_latent=recon_latent, idx_cpu_i64=idx_cpu_i64)
                        scheduled += 1
                        if scheduled >= prefetch_n:
                            break

            # Current layer: get latents on CUDA (may use cache or on-demand).
            latent_gpu = self._get_gpu_latents_for_layer(layer_idx=layer_idx, recon_latent=recon_latent)

            l_idx = self.deltakv_layer_to_idx[layer_idx]
            k_cache = self.deltakv_full_kv_cache[0, l_idx]
            v_cache = self.deltakv_full_kv_cache[1, l_idx]

            with profiler.record("deltakv_reconstruct_triton_compress_up"):
                kv_delta = self.compress_up[l_idx](latent_gpu)  # (Nt, kv_dim) in de-RoPE space for K

            with profiler.record("deltakv_reconstruct_triton_read_fathers"):
                father_slots = self.deltakv_latent_to_full_slots[l_idx, recon_latent].to(torch.int32)  # (Nt, K)
            if (father_slots < 0).any():
                raise RuntimeError("DeltaKV: missing father slots for reconstruction.")

            cos_sin = self.cos_sin_cache[:, 0, :]

            with profiler.record("deltakv_reconstruct_triton_kernel"):
                deltakv_reconstruct_writeback(
                    kv_delta=kv_delta,
                    father_slots=father_slots,
                    slot_to_pos=self.deltakv_slot_to_pos,
                    out_slots=recon_out_slot,
                    out_pos=recon_pos,
                    cos_sin=cos_sin,
                    k_cache=k_cache,
                    v_cache=v_cache,
                )

            return active_slots, local_req, new_context_lens, temp_slots

    @torch.no_grad()
    def deltakv_evict(self, seqs: list[Sequence]):
        # Re-implement the V2 eviction store path to support CPU latent offload.
        from sparsevllm.triton_kernel.deltakv_kernels import deltakv_gather_kv_unrope

        with profiler.record("deltakv_evict_triton_total"):
            if not self.deltakv_layer_ids:
                return
            sink = int(self.config.num_sink_tokens)
            recent = int(self.config.num_recent_tokens)
            cluster_step = max(1, int(1.0 / max(1e-6, float(self.config.cluster_ratio))))
            cos_sin = self.cos_sin_cache[:, 0, :]

            for seq in seqs:
                with profiler.record("deltakv_evict_triton_seq"):
                    row_idx = self.seq_id_to_row.get(seq.seq_id, None)
                    if row_idx is None:
                        continue

                    total_len = int(self.row_seq_lens[row_idx])
                    compressed_len = int(self.row_deltakv_compressed_lens[row_idx])
                    buffer_start = sink + compressed_len
                    buffer_len = total_len - buffer_start
                    if buffer_len <= recent:
                        continue

                    evict_len = ((buffer_len - recent) // recent) * recent
                    if evict_len <= 0:
                        continue

                    evict_start = buffer_start
                    evict_end = evict_start + evict_len

                    with profiler.record("deltakv_evict_triton_read_slots"):
                        raw_slots_block = self.sparse_layer_raw_slots_map[row_idx, evict_start:evict_end].clone()
                    if (raw_slots_block < 0).any():
                        raise RuntimeError("DeltaKV eviction expects raw slots for the buffer block.")

                    with profiler.record("deltakv_evict_triton_select_centers"):
                        center_rel = torch.arange(0, evict_len, cluster_step, device="cuda", dtype=torch.long)
                        new_center_slots = raw_slots_block[center_rel].to(torch.int32)

                    with profiler.record("deltakv_evict_triton_prev_centers"):
                        sink_slots = self.sparse_layer_raw_slots_map[row_idx, :sink].to(torch.int32)
                        prev_center_slots_by_layer: dict[int, torch.Tensor] = {}
                        for lyr in self.deltakv_layer_ids:
                            existing = self.row_deltakv_center_slots[row_idx][lyr]
                            prev_center_slots_by_layer[lyr] = (
                                sink_slots if existing is None else existing.to(torch.int32)
                            )

                    with profiler.record("deltakv_evict_triton_build_masks"):
                        is_center = torch.zeros((evict_len,), device="cuda", dtype=torch.bool)
                        is_center[center_rel] = True
                        to_compress_mask = ~is_center
                        num_to_compress = int(to_compress_mask.sum().item())

                    if num_to_compress <= 0:
                        with profiler.record("deltakv_evict_triton_append_centers_only"):
                            for lyr in self.deltakv_layer_ids:
                                self.row_deltakv_center_slots[row_idx][lyr] = torch.cat(
                                    [prev_center_slots_by_layer[lyr], new_center_slots], dim=0
                                )
                            self.row_deltakv_compressed_lens[row_idx] += evict_len
                        continue

                    with profiler.record("deltakv_evict_triton_alloc_latent"):
                        latent_slots = self._allocate_deltakv_latent(num_to_compress).to(torch.int32)
                        pos_all = torch.arange(evict_start, evict_end, device="cuda", dtype=torch.int32)
                        pos_to_compress = pos_all[to_compress_mask]
                        self.sparse_layer_latent_slots_map[row_idx, pos_to_compress.to(torch.long)] = latent_slots

                    raw_slots_block_i32 = raw_slots_block.to(torch.int32)

                    # Pre-copy latent slot indices to CPU once (used by offloaded layers).
                    latent_slots_cpu_i64 = latent_slots.to(torch.long).to(device="cpu", non_blocking=False)

                    for lyr in self.deltakv_layer_ids:
                        l_idx = self.deltakv_layer_to_idx[lyr]
                        k_cache = self.deltakv_full_kv_cache[0, l_idx]
                        v_cache = self.deltakv_full_kv_cache[1, l_idx]

                        with profiler.record("deltakv_evict_triton_gather_unrope"):
                            kv_block = deltakv_gather_kv_unrope(
                                slots=raw_slots_block_i32,
                                pos=pos_all,
                                cos_sin=cos_sin,
                                k_cache=k_cache,
                                v_cache=v_cache,
                            ).unsqueeze(0)

                        existing_center_slots = prev_center_slots_by_layer[lyr]
                        with profiler.record("deltakv_evict_triton_cluster"):
                            topk_center_indices, base_kv = self._cluster_compress(
                                layer_idx=lyr,
                                kv_states=kv_block,
                                existing_center_slots=existing_center_slots,
                                cluster_step=cluster_step,
                            )

                        with profiler.record("deltakv_evict_triton_remap_fathers"):
                            all_center_slots = torch.cat([existing_center_slots, new_center_slots], dim=0)
                            father_slots_full = all_center_slots[topk_center_indices.to(torch.long)]
                            father_slots = father_slots_full[to_compress_mask]

                        with profiler.record("deltakv_evict_triton_store_fathers"):
                            K = self.deltakv_latent_to_full_slots.shape[-1]
                            k_eff = father_slots.shape[1]
                            if k_eff < K:
                                pad = father_slots[:, :1].expand(-1, K - k_eff)
                                father_slots = torch.cat([father_slots, pad], dim=1)
                            elif k_eff > K:
                                father_slots = father_slots[:, :K]
                            self.deltakv_latent_to_full_slots[l_idx, latent_slots] = father_slots.to(torch.int32)

                        down = self.compress_down[l_idx]
                        with profiler.record("deltakv_evict_triton_compress_down"):
                            kv_down = down(kv_block).squeeze(0)
                            base_down = down(base_kv).squeeze(0)
                            latent_all = (kv_down - base_down)[to_compress_mask]

                        with profiler.record("deltakv_evict_triton_store_latent"):
                            if l_idx in self._deltakv_latent_keep_gpu_lidx:
                                assert self.deltakv_latent_cache_gpu is not None
                                gidx = self._deltakv_latent_gpu_map[l_idx]
                                self.deltakv_latent_cache_gpu[gidx, latent_slots.to(torch.long)] = latent_all.to(
                                    self.deltakv_latent_cache_gpu.dtype
                                )
                            else:
                                self._store_latent_offload(
                                    l_idx=l_idx,
                                    latent_slots_cpu_i64=latent_slots_cpu_i64,
                                    latent_all=latent_all,
                                )

                    with profiler.record("deltakv_evict_triton_append_centers"):
                        for lyr in self.deltakv_layer_ids:
                            self.row_deltakv_center_slots[row_idx][lyr] = torch.cat(
                                [prev_center_slots_by_layer[lyr], new_center_slots], dim=0
                            )

                    with profiler.record("deltakv_evict_triton_free_full_slots"):
                        free_slots = raw_slots_block_i32[to_compress_mask]
                        ptr = self._num_free_slots_deltakv_full
                        self.free_slots_stack_deltakv_full[ptr: ptr + free_slots.numel()] = free_slots
                        self._num_free_slots_deltakv_full += free_slots.numel()
                        self.deltakv_slot_to_pos[free_slots] = -1

                    with profiler.record("deltakv_evict_triton_update_maps"):
                        self.sparse_layer_raw_slots_map[row_idx, pos_to_compress.to(torch.long)] = -1
                        self.row_deltakv_compressed_lens[row_idx] += evict_len


# Kernel是ShadowKV的，好像根据github issues不能保证正确性
class DeltaKVCacheTritonManagerV3WithCUDAOffload(DeltaKVCacheTritonManagerV3WithOffload):
    """DeltaKV V3 with CUDA-side gather from pinned CPU latents.

    Uses the CUDA extension under `src/sparsevllm/cuda_kernel` (ShadowKV-style) to let
    GPU gather selected latent rows directly from pinned CPU memory. This avoids the
    CPU-side gather + H2D copy in `DeltaKVCacheTritonManagerV3WithOffload`.

    Notes/assumptions (for testing):
    - Requires bf16 latents and `kv_compressed_size` to be a multiple of 128 (kernel copies 128-wide chunks).
    - CPU latent caches are allocated as pinned memory in layout compatible with the kernel.
    - This manager *requires* the extension to be installed; otherwise it raises at init.
    """

    def __init__(self, config: Config, rank: int, world_size: int):
        # IMPORTANT: decide whether CUDA offload is usable *before* `super().__init__()`,
        # because the parent init will call `allocate_kv_cache()`, which uses our
        # `_create_cpu_latent_layer()` override.
        self._cuda_offload_map_size = int(getattr(config, "deltakv_cuda_offload_map_size", 256) or 256)
        self._cuda_offload_latent_dim = int(getattr(config, "kv_compressed_size", 0) or 0)
        self._cuda_offload_latent_heads = 0
        self._cpu_gpu_transfer_cuda = None

        # Import the extension (required for this manager).
        try:
            import cpu_gpu_transfer_cuda  # type: ignore

            self._cpu_gpu_transfer_cuda = cpu_gpu_transfer_cuda
        except Exception as e:
            raise RuntimeError(
                "DeltaKV CUDA offload requested but extension `cpu_gpu_transfer_cuda` is not available. "
                "Install it with: `pip install -e src/sparsevllm/cuda_kernel`."
            ) from e

        # Validate constraints using config.hf_config (available before CacheManager init).
        hf_dtype = getattr(getattr(config, "hf_config", None), "torch_dtype", None)
        if hf_dtype != torch.bfloat16:
            raise RuntimeError(f"DeltaKV CUDA offload requires bf16 latents; got {hf_dtype}.")

        if self._cuda_offload_latent_dim % 128 != 0:
            raise RuntimeError(
                f"CUDA offload requires kv_compressed_size multiple of 128; got {self._cuda_offload_latent_dim}."
            )
        self._cuda_offload_latent_heads = self._cuda_offload_latent_dim // 128

        if self._cuda_offload_map_size not in (128, 256, 512, 1024):
            raise RuntimeError(
                f"CUDA offload map_size must be 128/256/512/1024; got {self._cuda_offload_map_size}."
            )

        # Parent init performs allocation and sets up streams/caches.
        super().__init__(config, rank, world_size)

        assert self._cpu_gpu_transfer_cuda is not None
        h = self._cuda_offload_latent_heads
        m = self._cuda_offload_map_size
        logger.info(f"DeltaKV CUDA offload enabled: latent_dim={self._cuda_offload_latent_dim}, map_size={m}.")
        # Kernel-side bookkeeping buffers (like EfficientTransfer), fixed for (batch=1, heads=h, map_size=m).
        self._cuda_cached_pos_ids = torch.full((1, h, m), -1, device="cuda", dtype=torch.int64)
        self._cuda_offsets = torch.zeros((h * m,), device="cuda", dtype=torch.int32)
        self._cuda_cnts = torch.zeros((h,), device="cuda", dtype=torch.int32)
        self._cuda_signals = torch.zeros((h,), device="cuda", dtype=torch.int32)
        self._cuda_gather_buf = torch.empty((1, h, m, 128), device="cuda", dtype=torch.bfloat16)
        self._cuda_temp_buf = torch.empty_like(self._cuda_gather_buf)

    def _create_cpu_latent_layer(self, *, lidx: int, num_slots: int, latent_dim: int) -> torch.Tensor:
        # Allocate pinned CPU memory in [1, heads, slots, 128] layout so the CUDA kernel can
        # gather 128-wide chunks efficiently.
        if latent_dim % 128 != 0:
            raise RuntimeError(f"DeltaKV cuda-offload expects latent_dim multiple of 128; got {latent_dim}.")
        heads = latent_dim // 128
        return torch.empty((1, heads, num_slots, 128), dtype=self.hf_config.torch_dtype, device="cpu", pin_memory=True)

    def _store_latent_offload(
        self,
        *,
        l_idx: int,
        latent_slots_cpu_i64: torch.Tensor,
        latent_all: torch.Tensor,
    ) -> None:
        cpu_layer = self.deltakv_latent_cache_cpu_layers[l_idx]
        if cpu_layer is None:
            raise RuntimeError("DeltaKV cuda-offload: missing CPU latent cache for this layer.")

        if cpu_layer.dim() != 4:
            raise RuntimeError("DeltaKV cuda-offload: expected 4D pinned CPU latent cache layout.")

        # Stage GPU -> pinned host (2D), then scatter into pinned CPU cache.
        host2d = torch.empty((latent_all.shape[0], latent_all.shape[1]), dtype=cpu_layer.dtype, pin_memory=True)
        host2d.copy_(latent_all.to(host2d.dtype), non_blocking=False)

        heads = cpu_layer.shape[1]
        host3 = host2d.view(-1, heads, 128).permute(1, 0, 2).contiguous()  # (H, Nc, 128)
        cpu_layer0 = cpu_layer[0]  # (H, slots, 128)
        cpu_layer0.index_copy_(1, latent_slots_cpu_i64, host3)

    def _prefetch_latents_to_gpu(self, *, layer_idx: int, recon_latent: torch.Tensor, idx_cpu_i64: torch.Tensor):
        # CUDA gather reads directly from pinned CPU memory; no explicit H2D prefetch.
        assert self._cpu_gpu_transfer_cuda is not None
        return

    def _get_gpu_latents_for_layer(
        self,
        *,
        layer_idx: int,
        recon_latent: torch.Tensor,
    ) -> torch.Tensor:
        # GPU-resident layers keep the same behavior.
        l_idx = self.deltakv_layer_to_idx[layer_idx]
        if l_idx in self._deltakv_latent_keep_gpu_lidx:
            return super()._get_gpu_latents_for_layer(layer_idx=layer_idx, recon_latent=recon_latent)

        assert self._cpu_gpu_transfer_cuda is not None

        cpu_layer = self.deltakv_latent_cache_cpu_layers[l_idx]
        if cpu_layer is None:
            raise RuntimeError("DeltaKV cuda-offload: missing CPU latent cache for this layer.")
        if cpu_layer.dim() != 4:
            raise RuntimeError("DeltaKV cuda-offload: expected 4D pinned CPU latent cache layout.")

        # Gather in chunks of map_size from pinned CPU -> GPU buffer, then pack to (Nt, latent_dim).
        nt = int(recon_latent.numel())
        if nt == 0:
            return torch.empty((0, self._cuda_offload_latent_dim), device="cuda", dtype=torch.bfloat16)

        h = self._cuda_offload_latent_heads
        m = self._cuda_offload_map_size
        out = torch.empty((nt, self._cuda_offload_latent_dim), device="cuda", dtype=torch.bfloat16)

        idx_i64 = recon_latent.to(torch.int64)
        cpu_v_length = int(cpu_layer.shape[2] * 128)  # slots * 128
        gpu_v_length = int(m * 128)

        for start in range(0, nt, m):
            end = min(nt, start + m)
            chunk = idx_i64[start:end]
            if end - start < m:
                pad = chunk[-1:].expand(m - (end - start))
                chunk = torch.cat([chunk, pad], dim=0)

            cur_pos = chunk.view(1, 1, m).expand(1, h, m).contiguous()

            # Force "all miss" behavior per chunk (we don't want cross-chunk caching).
            self._cuda_cached_pos_ids.fill_(-1)
            self._cuda_offsets.zero_()
            self._cuda_cnts.zero_()
            self._cuda_signals.zero_()

            self._cpu_gpu_transfer_cuda.reorder_keys_and_compute_offsets(
                self._cuda_cached_pos_ids,
                cur_pos,
                self._cuda_offsets,
                self._cuda_cnts,
                1,
                h,
                m,
            )

            self._cpu_gpu_transfer_cuda.gather_copy_with_offsets(
                cpu_layer,
                self._cuda_gather_buf,
                self._cuda_temp_buf,
                self._cuda_offsets,
                self._cuda_cnts,
                self._cuda_signals,
                1,
                h,
                cpu_v_length,
                gpu_v_length,
                0,
                gpu_v_length,
                m,
            )

            gathered = self._cuda_gather_buf[0].permute(1, 0, 2).reshape(m, self._cuda_offload_latent_dim)
            out[start:end] = gathered[: end - start]

        return out
