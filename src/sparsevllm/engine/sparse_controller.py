from dataclasses import dataclass
import torch
from sparsevllm.config import Config
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.cache_manager import CacheManager, DeltaKVCacheManager
from sparsevllm.utils.profiler import profiler
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger, log_level
from sparsevllm.triton_kernel.omnikv_fused import build_omnikv_keep_and_slots


@dataclass
class LayerBatchSparseState:
    """每一层的逻辑稀疏状态"""
    attn_score: torch.Tensor | None = None
    active_indices: torch.Tensor | None = None # 逻辑索引 [B, K]
    active_slots: torch.Tensor | None = None   # 物理槽位 [B, K]
    req_indices: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None

    # for DeltaKV
    active_compressed_indices: torch.Tensor | None = None
    # Global row indices (into CacheManager slot maps). For some sparse views we may
    # also return local req indices to kernels.
    global_req_indices: torch.Tensor | None = None
    # DeltaKV uses scratch (temp) slots during reconstruction; only the last layer in a
    # segment should free them so other layers can reuse the same view/slots.
    deltakv_free_temp_slots: bool = False

class SparseController:
    """
    稀疏策略控制器，管理 KV Cache 的逻辑视图 (Reading View) 和 压缩策略。
    """
    def __init__(self, config: Config, cache_manager: CacheManager):
        self.sparse_method = config.vllm_sparse_method
        
        self.config = config
        self.cache_manager = cache_manager

        self.obs_layer_ids = self.config.obs_layer_ids
        self.full_attn_layers = self.config.full_attn_layers
        self.num_layers = self.config.hf_config.num_hidden_layers

        self.num_sink = self.config.num_sink_tokens
        self.num_recent = self.config.num_recent_tokens
        self.num_top = self.config.num_top_tokens
        self.num_top_in_prefill = self.config.num_top_tokens_in_prefill
        
        # 稀疏层私有状态: dict[layer_idx, LayerSparseState]
        self.layer_batch_sparse_states: dict[int, LayerBatchSparseState] = {}
        for i in range(self.num_layers):
            self.layer_batch_sparse_states[i] = LayerBatchSparseState()

        # 静态配置
        self.sparse_config = {
            "vllm_sparse_method": self.sparse_method,
            "num_sink_tokens": self.config.num_sink_tokens,
            "num_recent_tokens": self.config.num_recent_tokens,
            "num_top_tokens": self.config.num_top_tokens,
            "obs_layer_ids": self.config.obs_layer_ids,
            "full_attn_layers": self.config.full_attn_layers,
        }

        self.layers = None

    @torch.no_grad()
    def prepare_forward(self, seqs: list[Sequence], is_prefill: bool):
        """前向计算前，重置并准备各层的稀疏视图"""
        # 每步 prefill or decode 前会执行
        ctx = get_context()
        ctx.sparse_config = self.sparse_config if self.sparse_method else None

        for i in range(self.num_layers):
            state = self.layer_batch_sparse_states[i]
            batch_state = self.cache_manager.get_layer_batch_states(i)
            
            # 统一语义：context_lens 代表当前 attn 可见长度 （即使是动态稀疏方法）
            state.context_lens = batch_state.context_lens.clone()  # 虽然clone，但是感觉开销不大
            state.req_indices = batch_state.req_indices
            state.global_req_indices = batch_state.req_indices
            state.attn_score = None

            # 默认视图
            state.active_indices = None
            # 默认应该是全量的；active 开头的属性，只对 omnikv，deltakv，quest 这些不会物理删除token，但是有动态稀疏性的方法起效
            state.active_slots = None
            state.active_compressed_indices = None
            state.deltakv_free_temp_slots = False

            # 为需要收集注意力分数的层分配 attn score 的对应 tensor
            if self._needs_attn_score(i, is_prefill, seqs):
                batch_size = len(seqs)
                num_heads = self.config.hf_config.num_attention_heads // self.config.tensor_parallel_size
                max_len = int(state.context_lens.max())
                # TODO 开销比较大的 attn score 初始化？
                _val = 0.0 if is_prefill else -1e20
                with profiler.record("sparse_prepare_attn_score"):
                    # TODO 这个后面用 bf16 应该问题不大吧？
                    state.attn_score = torch.full((batch_size, num_heads, max_len), _val, dtype=torch.float32, device="cuda")

    def set_modules(self, modules):
        self.layers = modules

    @torch.no_grad()
    def post_forward(self, seqs: list[Sequence], is_prefill: bool):
        """持久化压缩 (如 SnapKV / DeltaKV)"""
        if get_context().is_long_text is False and self.sparse_method != 'deltakv':
            return

        if is_prefill:
            self.on_every_chunk_prefill_end(seqs)

        # Decode 阶段如果 Recent Buffer 溢出也需要压缩 (对于 DeltaKV)
        if not is_prefill and self.sparse_method == 'deltakv':
             self._deltakv_eviction(seqs)
        if not is_prefill and self.sparse_method in ('snapkv', 'pyramidkv'):
            self._snapkv_decode_eviction(seqs)

    @torch.no_grad()
    def on_every_chunk_prefill_end(self, seqs: list[Sequence]):
        if get_context().is_long_text is False and self.sparse_method != 'deltakv':
            return

        # DeltaKV: Always try to compress incrementally (to save memory during long prefill)
        if self.sparse_method == 'deltakv':
            assert self.config.chunk_prefill_accel_omnikv
            self._deltakv_eviction(seqs)
            return

        # SnapKV / PyramidKV: Only evict at the end of prefill
        is_last_chunk = any(seq.is_last_chunk_prefill for seq in seqs)
        if not is_last_chunk:
            return

        if self.sparse_method == 'snapkv' or self.sparse_method == 'pyramidkv':
            self._snapkv_prefill_eviction(seqs)

    def get_read_view(self, layer_idx: int):
        """
        供 Attention.forward 调用：获取当前层应该读取的逻辑视图。
        返回 (active_slots, active_indices, req_indices, context_lens, attn_score, temp_slots)
        """
        sparse_state = self.layer_batch_sparse_states[layer_idx]
        if (self.sparse_method in ("omnikv", "deltakv") and layer_idx in self.full_attn_layers) or \
            self.sparse_method in ('snapkv', 'pyramidkv', ''):

            return (
                self.cache_manager.get_layer_buffer_req_to_token_slots(layer_idx),  # 全部 token slots
                None,
                sparse_state.req_indices,
                sparse_state.context_lens,
                sparse_state.attn_score,
                None,
            )

        assert layer_idx not in self.full_attn_layers
        if self.sparse_method == 'deltakv':
            ctx = get_context()
            # active_compressed_indices: (B, Kmax), padded with -1; may be None (treated as K=0)
            active = sparse_state.active_compressed_indices
            # For DeltaKV we always use a batch-major Req->slots table, so kernels use local req indices.
            chunk_lens = None
            if ctx.is_prefill:
                if ctx.cu_seqlens_q is None or ctx.cu_seqlens_q.numel() <= 1:
                    chunk_lens = None
                else:
                    chunk_lens = (ctx.cu_seqlens_q[1:] - ctx.cu_seqlens_q[:-1]).to(torch.int32)

            active_slots, local_req_indices, new_context_lens, temp_slots = self.cache_manager.deltakv_reconstruct(
                layer_idx=layer_idx,
                active_compressed_indices=active,
                context_lens=sparse_state.context_lens,
                req_indices=sparse_state.global_req_indices,
                chunk_lens=chunk_lens,
            )
            return (
                active_slots,
                None,
                local_req_indices,
                new_context_lens,
                sparse_state.attn_score,
                temp_slots if sparse_state.deltakv_free_temp_slots else None,
            )

        if self.sparse_method == 'omnikv':
            if sparse_state.active_slots is not None:
                active_slots = sparse_state.active_slots
                logger.debug('active_slots 是被 omnikv 选到的 slots')
            else:
                active_slots = self.cache_manager.get_layer_buffer_req_to_token_slots(layer_idx)
                logger.debug('active_slots is None')

            return (
                active_slots,
                sparse_state.active_indices,
                sparse_state.req_indices,
                sparse_state.context_lens,
                sparse_state.attn_score,
                None,
            )
        else:
            raise ValueError

    def on_layer_end(self, layer_idx: int, context):
        """每一层结束后的动态策略 (如 OmniKV / DeltaKV)"""
        if get_context().is_long_text is False and self.sparse_method != 'deltakv':
            return

        if self.sparse_method not in ('omnikv', 'deltakv'):
            return

        if context.is_prefill and not self.config.chunk_prefill_accel_omnikv:
            return

        if layer_idx not in self.obs_layer_ids:
            return

        with profiler.record("sparse_on_layer_end"):
            state = self.layer_batch_sparse_states[layer_idx]
            if state.attn_score is None:
                raise ValueError("Attn Score hasn't been initialized")

            if state.attn_score.dim() == 3:
                if context.is_prefill:
                    chunk_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                    state.attn_score /= chunk_lens.view(-1, 1, 1)  # 不除其实也无所谓
                # 对head做max pooling
                state.attn_score = state.attn_score.max(dim=1).values

            target_layers = []
            for j in range(layer_idx + 1, self.num_layers):
                if j in self.full_attn_layers: break
                target_layers.append(j)
            assert len(target_layers) > 0

            self._update_dynamic_omnikv_indices(layer_idx, target_layers)

    @torch.no_grad()
    def _deltakv_eviction(self, seqs: list[Sequence]):
        assert get_context().is_long_text or self.sparse_method == 'deltakv'
        self.cache_manager.deltakv_evict(seqs)

    @torch.no_grad()
    def reconstruct_deltakv_kv_fused(
        self,
        layer_idx: int,
        module: torch.nn.Module,
        physical_slots: torch.Tensor,
        active_indices: torch.Tensor | None,
        context_lens: torch.Tensor,
        req_indices: torch.Tensor
    ):
        assert get_context().is_long_text or self.sparse_method == 'deltakv'
        return self.cache_manager.deltakv_reconstruct(
            layer_idx=layer_idx,
            active_compressed_indices=active_indices,
            context_lens=context_lens,
            req_indices=req_indices,
            chunk_lens=None,
        )

    @torch.no_grad()
    def _snapkv_prefill_eviction(self, seqs: list[Sequence]):
        for layer_idx in range(self.num_layers):
            state = self.layer_batch_sparse_states[layer_idx]
            attn_scores = state.attn_score
            if attn_scores is None:
                continue
            if attn_scores.dim() == 3:
                attn_scores = attn_scores.max(dim=1).values
            budget = self._get_layer_budget(layer_idx, is_prefill=True)
            if budget is None:
                continue
            for b_idx, seq in enumerate(seqs):
                if not seq.is_last_chunk_prefill:
                    continue
                kv_len = int(state.context_lens[b_idx])
                if kv_len <= budget:
                    continue
                if log_level == 'DEBUG':
                    logger.debug(
                        "[SnapKV] prefill eviction: "
                        f"layer={layer_idx} seq_id={seq.seq_id} kv_len={kv_len} budget={budget}"
                    )
                keep_indices = self._snapkv_select_indices(
                    attn_scores[b_idx, :kv_len], kv_len, budget
                )
                self.cache_manager.free_part_slots(layer_idx, seq, keep_indices)

    @torch.no_grad()
    def _snapkv_decode_eviction(self, seqs: list[Sequence]):
        for layer_idx in range(self.num_layers):
            state = self.layer_batch_sparse_states[layer_idx]
            attn_scores = state.attn_score
            if attn_scores is None:
                continue
            if attn_scores.dim() == 3:
                attn_scores = attn_scores.max(dim=1).values

            budget = self._get_layer_budget(layer_idx, is_prefill=False)
            if budget is None:
                continue

            top_budget = budget - self.num_sink - self.num_recent
            trigger_len = int(2.0 * top_budget)
            for b_idx, seq in enumerate(seqs):
                kv_len = int(state.context_lens[b_idx])
                if kv_len <= budget or kv_len < trigger_len:
                    continue
                if log_level == 'DEBUG':
                    logger.debug(
                        "[SnapKV] decode eviction: "
                        f"layer={layer_idx} seq_id={seq.seq_id} kv_len={kv_len} budget={budget} trigger_len={trigger_len}"
                    )
                keep_indices = self._snapkv_select_indices(
                    attn_scores[b_idx, :kv_len], kv_len, budget
                )
                self.cache_manager.free_part_slots(layer_idx, seq, keep_indices)

    def _snapkv_select_indices(self, scores: torch.Tensor, kv_len: int, budget: int) -> torch.Tensor:
        assert kv_len > budget
        device = scores.device
        
        # 1. Sink indices
        sink_indices = torch.arange(self.num_sink, device=device)
        
        # 2. Recent indices
        recent_start = kv_len - self.num_recent
        recent_indices = torch.arange(recent_start, kv_len, device=device)
        
        # 3. Top-K indices
        num_topk = budget - self.num_sink - self.num_recent
        if num_topk > 0 and recent_start > self.num_sink:
            middle_scores = scores[self.num_sink:recent_start]
            topk_indices_relative = middle_scores.topk(min(num_topk, middle_scores.shape[0]), dim=-1).indices
            topk_indices = topk_indices_relative + self.num_sink
            keep_indices = torch.cat([sink_indices, topk_indices, recent_indices])
        else:
            keep_indices = torch.cat([sink_indices, recent_indices])
            
        return keep_indices

    def _update_dynamic_omnikv_indices(self, obs_layer_idx, target_layers):
        assert get_context().is_long_text or self.sparse_method == 'deltakv'

        with profiler.record("sparse_update_dynamic_indices"):
            ctx = get_context()
            # full attn layer 的 req indices 是未处理的
            obs_sparse_state = self.layer_batch_sparse_states[obs_layer_idx]
            token_scores = obs_sparse_state.attn_score # (B, L)
            batch_size, max_len = token_scores.shape

            # 计算实际可检索的历史长度
            if ctx.is_prefill:
                chunk_lens = ctx.cu_seqlens_q[1:] - ctx.cu_seqlens_q[:-1]
                # num_recent 是在chunk之外额外再留 recent 个token
                hist_lens = obs_sparse_state.context_lens - chunk_lens - self.num_recent
            else:
                # num_recent 覆盖当前token
                hist_lens = obs_sparse_state.context_lens - self.num_recent
            if self.sparse_method == 'omnikv':
                hist_lens = hist_lens.clamp_min(self.num_sink)
            
            # 直接切除 Sink 之前的分数
            search_scores = token_scores[:, self.num_sink:]
            if self.sparse_method == 'omnikv':
                rel_hist_lens = hist_lens - self.num_sink
            elif self.sparse_method == 'deltakv':
                rel_hist_lens = self.cache_manager.get_compressed_lens(obs_sparse_state.req_indices)
            else:
                raise ValueError

            # 2. 掩码处理 (处理不等长 + 防止 topk 选到 buffer/chunk 区域)
            mask = torch.arange(search_scores.size(1), device="cuda") >= rel_hist_lens.unsqueeze(1)
            search_scores.masked_fill_(mask, -1e10)

            # 3. 提取 Top-K (per-seq, padded with -1)
            num_top = self.num_top_in_prefill if ctx.is_prefill else self.num_top
            topk_list = []
            k_list = []
            for b in range(batch_size):
                avail = int(rel_hist_lens[b].item())
                k_b = min(int(num_top), int(search_scores.size(1)), max(0, avail))
                k_list.append(k_b)
                if k_b <= 0:
                    topk_list.append(torch.empty((0,), device="cuda", dtype=torch.int32))
                else:
                    idx = search_scores[b].topk(k_b, dim=0).indices.to(torch.int32) + self.num_sink
                    topk_list.append(idx)
            k_max = max(k_list) if k_list else 0
            if k_max > 0:
                topk_indices = torch.full((batch_size, k_max), -1, device="cuda", dtype=torch.int32)
                for b in range(batch_size):
                    k_b = k_list[b]
                    if k_b > 0:
                        topk_indices[b, :k_b] = topk_list[b]
            else:
                topk_indices = torch.empty((batch_size, 0), device="cuda", dtype=torch.int32)

            # 4. 根据方法更新目标层状态
            if self.sparse_method == 'omnikv':
                local_req_indices = torch.arange(batch_size, dtype=torch.int32, device="cuda")
                topk_lens = torch.tensor(k_list, dtype=torch.int32, device="cuda")
                keep_indices, active_slots, new_context_lens = build_omnikv_keep_and_slots(
                    topk_indices,
                    topk_lens,
                    hist_lens,
                    obs_sparse_state.context_lens - hist_lens,  # lens of recent and chunk
                    self.cache_manager.get_layer_buffer_req_to_token_slots(obs_layer_idx + 1),
                    obs_sparse_state.req_indices,
                    self.num_sink,
                )

                for l_idx in target_layers:
                    target_sparse_state = self.layer_batch_sparse_states[l_idx]
                    target_sparse_state.active_indices = keep_indices
                    target_sparse_state.active_slots = active_slots
                    target_sparse_state.context_lens = new_context_lens
                    target_sparse_state.req_indices = local_req_indices
            
            elif self.sparse_method == 'deltakv':
                for l_idx in target_layers:
                    target_sparse_state = self.layer_batch_sparse_states[l_idx]
                    target_sparse_state.active_compressed_indices = topk_indices
                    # context_lens is finalized in cache_manager.deltakv_reconstruct(); keep a placeholder here.
                    target_sparse_state.context_lens = obs_sparse_state.context_lens
                    target_sparse_state.req_indices = obs_sparse_state.req_indices
                    target_sparse_state.global_req_indices = obs_sparse_state.req_indices
                    target_sparse_state.deltakv_free_temp_slots = (l_idx == target_layers[-1])
            else:
                raise ValueError
    
    def _needs_attn_score(self, layer_idx: int, is_prefill: bool, seqs: list[Sequence]) -> bool:
        if self.sparse_method in ('omnikv', 'deltakv') and layer_idx in self.obs_layer_ids:
            if is_prefill and not self.config.chunk_prefill_accel_omnikv:
                return False
            return True
        if self.sparse_method in ('snapkv', 'pyramidkv'):
            if is_prefill:
                if get_context().is_long_text is False:
                    return False
                is_last_chunk = any(seq.is_last_chunk_prefill for seq in seqs)
                return is_last_chunk

            # Decode: only collect scores when we're about to evict.
            budget = self._get_layer_budget(layer_idx, is_prefill=False)
            if budget is None:
                return False
            state = self.layer_batch_sparse_states[layer_idx]
            if state.context_lens is None:
                return False
            top_budget = budget - self.num_sink - self.num_recent
            trigger_len = int(2.0 * top_budget)
            return bool(((state.context_lens >= trigger_len) & (state.context_lens > budget)).any())
        return False
    
    def _get_layer_budget(self, layer_idx: int, is_prefill: bool) -> int | None:
        if layer_idx < self.config.snapkv_num_full_layers:
            return None
        num_top = self.num_top_in_prefill if is_prefill else self.num_top
        if self.config.pyramid_layer_ratios is not None:
            ratio = self.config.pyramid_layer_ratios[layer_idx]
            base_ratio = self.config.pyramid_layer_ratios[0]
            scaled_top_tokens = int(num_top * ratio / base_ratio)
            return self.num_sink + scaled_top_tokens + self.num_recent
        elif self.sparse_method == 'snapkv':
            return self.num_sink + num_top + self.num_recent
        return None
