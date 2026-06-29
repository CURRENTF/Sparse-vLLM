from dataclasses import dataclass
import os
import torch
import torch.nn.functional as F
from sparsevllm.config import Config
from sparsevllm.engine.activation_controller import ActivationController
from sparsevllm.engine.sequence import Sequence
from sparsevllm.engine.cache_manager import CacheManager, SparseSelection
from sparsevllm.utils.profiler import profiler
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger, log_level


def build_omnikv_keep_and_slots(*args, **kwargs):
    from sparsevllm.triton_kernel.omnikv_fused import build_omnikv_keep_and_slots as _build

    return _build(*args, **kwargs)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{name} must be one of 1/0, true/false, yes/no, or on/off; got {value!r}."
    )


@dataclass
class LayerBatchSparseState:
    """每一层的逻辑稀疏状态"""
    attn_score: torch.Tensor | None = None
    active_indices: torch.Tensor | None = None # 逻辑索引 [B, K]
    active_slots: torch.Tensor | None = None   # 物理槽位 [B, K]
    req_indices: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    max_context_len: int | None = None

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
        self.is_deltakv_family = isinstance(self.sparse_method, str) and self.sparse_method.startswith('deltakv')
        self.debug_dynamic_selection = {}
        self.debug_dynamic_selection_detail = os.environ.get(
            "SPARSEVLLM_DEBUG_DYNAMIC_SELECTION_DETAIL", ""
        ).lower() in ("1", "true", "yes", "on")
        self.dynamic_deltakv_topk_tiebreak = _env_bool(
            "SPARSEVLLM_DELTAKV_DETERMINISTIC_TOPK_TIEBREAK",
            False,
        )
        
        self.config = config
        self.cache_manager = cache_manager
        self.device = getattr(
            cache_manager,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )
        self.activation_controller = ActivationController.create(config, cache_manager)

        self.obs_layer_ids = self.config.obs_layer_ids
        self.full_attn_layers = self.config.full_attn_layers
        self.num_layers = self.config.hf_config.num_hidden_layers

        self.num_sink = self.config.num_sink_tokens
        self.num_recent = self.config.num_recent_tokens
        self.decode_keep_tokens = self.config.decode_keep_tokens
        head_dim = int(
            getattr(self.config.hf_config, "head_dim", None)
            or (self.config.hf_config.hidden_size // self.config.hf_config.num_attention_heads)
        )
        self.attn_softmax_scale = float(head_dim) ** -0.5
        score_dtype_name = str(getattr(self.config, "sparse_attn_score_dtype", "float32") or "float32").lower()
        self.attn_score_dtype = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }[score_dtype_name]
        
        # 稀疏层私有状态: dict[layer_idx, LayerSparseState]
        self.layer_batch_sparse_states: dict[int, LayerBatchSparseState] = {}
        for i in range(self.num_layers):
            self.layer_batch_sparse_states[i] = LayerBatchSparseState()
        self._decode_attn_score_buffers: dict[int, torch.Tensor] = {}

        # 静态配置
        self.sparse_config = {
            "vllm_sparse_method": self.sparse_method,
            "num_sink_tokens": self.config.num_sink_tokens,
            "num_recent_tokens": self.config.num_recent_tokens,
            "decode_keep_tokens": self.config.decode_keep_tokens,
            "obs_layer_ids": self.config.obs_layer_ids,
            "full_attn_layers": self.config.full_attn_layers,
            "dynamic_deltakv_topk_tiebreak": self.dynamic_deltakv_topk_tiebreak,
        }

        self.layers = None

    def set_tokenizer_metadata(
        self,
        *,
        delimiter_token_ids: list[int] | set[int] | tuple[int, ...] | None = None,
        non_execution_token_ids: list[int] | set[int] | tuple[int, ...] | None = None,
    ):
        self.activation_controller.set_tokenizer_metadata(
            delimiter_token_ids=delimiter_token_ids,
            non_execution_token_ids=non_execution_token_ids,
        )

    def clear_decode_attn_score_buffers(self):
        self._decode_attn_score_buffers.clear()

    def decode_cuda_graph_keepalive_tensors(self) -> list[torch.Tensor]:
        return self.activation_controller.decode_cuda_graph_keepalive_tensors()

    def _debug_record_dynamic_selection(self, bucket: str, layer_idx: int, **fields):
        entry = self.debug_dynamic_selection.setdefault(bucket, {}).setdefault(str(int(layer_idx)), {"calls": 0})
        entry["calls"] += 1
        entry.update(fields)

    def _debug_tensor_preview(self, tensor: torch.Tensor, limit: int = 16):
        t = tensor.detach().flatten()[:limit].cpu()
        if t.dtype in (torch.int8, torch.int16, torch.int32, torch.int64, torch.long, torch.bool):
            return [int(x) for x in t.tolist()]
        return [float(x) for x in t.tolist()]

    def _decode_softmax_token_scores(
        self,
        scores: torch.Tensor,
        *,
        candidate_start: int,
        candidate_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Convert decode raw QK logits [B, H, L] into masked token scores [B, L]."""
        if scores.dim() != 3:
            raise ValueError(f"Expected decode scores with shape [B, H, L], got {tuple(scores.shape)}.")
        candidate_start = int(candidate_start)
        if candidate_start < 0 or candidate_start > scores.shape[-1]:
            raise ValueError(
                f"candidate_start must be within score length; got {candidate_start} for L={scores.shape[-1]}."
            )
        candidate_scores = scores[:, :, candidate_start:]
        candidate_lens = candidate_lens.to(device=scores.device, dtype=torch.long).clamp_min(0)
        candidate_lens = candidate_lens.clamp_max(candidate_scores.shape[-1])
        candidate_pos = torch.arange(candidate_scores.shape[-1], device=scores.device)
        candidate_mask = candidate_pos.unsqueeze(0) < candidate_lens.unsqueeze(1)

        logits = candidate_scores.float() * float(self.attn_softmax_scale)
        logits = logits.masked_fill(~candidate_mask[:, None, :], torch.finfo(logits.dtype).min)
        candidate_token_scores = torch.softmax(logits, dim=-1).max(dim=1).values

        model_dtype = getattr(self.config.hf_config, "torch_dtype", None)
        if isinstance(model_dtype, str):
            model_dtype = {
                "float16": torch.float16,
                "torch.float16": torch.float16,
                "bfloat16": torch.bfloat16,
                "torch.bfloat16": torch.bfloat16,
            }.get(model_dtype.lower())
        if model_dtype in (torch.float16, torch.bfloat16):
            candidate_token_scores = candidate_token_scores.to(model_dtype)
        min_score = torch.finfo(candidate_token_scores.dtype).min
        candidate_token_scores = candidate_token_scores.masked_fill(~candidate_mask, min_score)
        token_scores = torch.full(
            (scores.shape[0], scores.shape[-1]),
            min_score,
            dtype=candidate_token_scores.dtype,
            device=candidate_token_scores.device,
        )
        token_scores[:, candidate_start:] = candidate_token_scores
        return token_scores

    @torch.no_grad()
    def prepare_forward(self, seqs: list[Sequence], is_prefill: bool):
        """前向计算前，重置并准备各层的稀疏视图"""
        # 每步 prefill or decode 前会执行
        ctx = get_context()
        ctx.sparse_config = self.sparse_config if self.sparse_method else None
        self.activation_controller.prepare_forward(seqs, is_prefill)

        for i in range(self.num_layers):
            state = self.layer_batch_sparse_states[i]
            batch_state = self.cache_manager.get_layer_batch_states(i)
            
            # 统一语义：context_lens 代表当前 attn 可见长度 （即使是动态稀疏方法）
            if not is_prefill:
                # CUDA Graph replay updates the cache-manager decode metadata in place;
                # full/observation layers must read the stable tensor address captured here.
                state.context_lens = batch_state.context_lens
            else:
                state.context_lens = batch_state.context_lens.clone()  # 虽然clone，但是感觉开销不大
            state.max_context_len = batch_state.max_context_len
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
                if not is_prefill and state.context_lens is not None:
                    batch_size = int(state.context_lens.numel())
                else:
                    batch_size = len(seqs)
                num_heads = self.config.hf_config.num_attention_heads // self.config.tensor_parallel_size
                max_len = self._state_max_context_len(state)
                _val = 0.0 if is_prefill else -1e20
                with profiler.record("sparse_prepare_attn_score"):
                    if is_prefill:
                        # Prefill score shapes follow chunking and are not replayed by decode CUDA graphs.
                        state.attn_score = torch.full(
                            (batch_size, num_heads, max_len),
                            _val,
                            dtype=self.attn_score_dtype,
                            device=self.device,
                        )
                    else:
                        state.attn_score = self._get_decode_attn_score_buffer(
                            i,
                            batch_size,
                            num_heads,
                            max_len,
                            fill_value=_val,
                        )

    def set_modules(self, modules):
        self.layers = modules

    def apply_activation_hook(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        context,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        return self.activation_controller.apply_layer_hook(
            layer_idx,
            hidden_states,
            residual,
            context,
        )

    def _get_decode_attn_score_buffer(
        self,
        layer_idx: int,
        batch_size: int,
        num_heads: int,
        max_len: int,
        *,
        fill_value: float,
    ) -> torch.Tensor:
        """Return a decode attn-score view backed by a stable per-layer buffer."""
        if batch_size <= 0 or num_heads <= 0 or max_len <= 0:
            raise RuntimeError(
                "Decode attention score buffer requires positive shape: "
                f"layer={layer_idx} batch={batch_size} heads={num_heads} max_len={max_len}."
            )
        buf = self._decode_attn_score_buffers.get(int(layer_idx))
        needs_alloc = (
            buf is None
            or buf.dtype != self.attn_score_dtype
            or buf.device != self.device
            or int(buf.shape[0]) < int(batch_size)
            or int(buf.shape[1]) < int(num_heads)
            or int(buf.shape[2]) < int(max_len)
        )
        if needs_alloc:
            buf = torch.empty(
                (int(batch_size), int(num_heads), int(max_len)),
                dtype=self.attn_score_dtype,
                device=self.device,
            )
            self._decode_attn_score_buffers[int(layer_idx)] = buf
        view = buf[:batch_size, :num_heads, :max_len]
        view.fill_(fill_value)
        return view

    def _state_max_context_len(self, state: LayerBatchSparseState) -> int:
        if state.max_context_len is not None:
            return int(state.max_context_len)
        return int(state.context_lens.max().item())

    def get_layer_max_context_len(self, layer_idx: int) -> int | None:
        return self.layer_batch_sparse_states[layer_idx].max_context_len

    @torch.no_grad()
    def on_layer_attention_end(self, layer_idx: int):
        """Layer-local sparse finalization for methods that use temporary prefill KV."""
        ctx = get_context()
        if not ctx.is_prefill or self.sparse_method != "pyramidkv":
            return
        if not self.cache_manager.has_prefill_staging_view(layer_idx):
            return

        with profiler.record("pyramidkv_staging_materialize_layer"):
            state = self.layer_batch_sparse_states[layer_idx]
            budget = self._get_layer_budget(layer_idx, is_prefill=True)
            if budget is None:
                raise RuntimeError("PyramidKV full-prefill staging requires a layer budget.")

            seqs = getattr(ctx, "seqs", None)
            if seqs is None:
                raise RuntimeError("PyramidKV full-prefill staging requires current seqs in context.")
            seq_keep_indices = []
            for b_idx, seq in enumerate(seqs):
                if not seq.is_last_chunk_prefill:
                    raise RuntimeError("PyramidKV full-prefill staging should only run on the final prefill chunk.")
                kv_len = int(state.context_lens[b_idx])
                if kv_len <= budget:
                    keep_indices = torch.arange(kv_len, device=self.device, dtype=torch.long)
                else:
                    attn_scores = self.cache_manager.pop_prefill_attention_score(layer_idx, seq)
                    if attn_scores is None:
                        raise RuntimeError(
                            "PyramidKV full-prefill staging requires prefill attention scores. "
                            f"layer={layer_idx} seq_id={seq.seq_id}"
                        )
                    if attn_scores.dim() == 2:
                        attn_scores = attn_scores.max(dim=0).values
                    keep_indices = self._snapkv_select_indices(
                        attn_scores[:kv_len],
                        kv_len,
                        budget,
                        pool_kernel_size=int(getattr(self.config, "pool_kernel_size", 1) or 1),
                    )
                seq_keep_indices.append((seq, keep_indices))
            self.cache_manager.materialize_prefill_staging_layer_batch(layer_idx, seq_keep_indices)

    @torch.no_grad()
    def post_forward(self, seqs: list[Sequence], is_prefill: bool):
        """持久化压缩 (如 SnapKV / DeltaKV)"""
        self.activation_controller.post_forward(seqs, is_prefill)

        if get_context().is_long_text is False and not self.is_deltakv_family:
            return

        if is_prefill:
            self.on_every_chunk_prefill_end(seqs)

        # Decode 阶段如果 Recent Buffer 溢出也需要压缩 (对于 DeltaKV)
        if not is_prefill and self.is_deltakv_family:
             self._deltakv_eviction(seqs)
        if not is_prefill and self.sparse_method in ('snapkv', 'pyramidkv'):
            self._snapkv_decode_eviction(seqs)
        if not is_prefill and self.sparse_method == "rkv":
            self._rkv_decode_eviction(seqs)
        if not is_prefill and self.sparse_method == "skipkv":
            self._skipkv_decode_eviction(seqs)
        if not is_prefill and self.sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            self._streamingllm_decode_eviction(seqs)

    @torch.no_grad()
    def on_every_chunk_prefill_end(self, seqs: list[Sequence]):
        if get_context().is_long_text is False and not self.is_deltakv_family:
            return

        # DeltaKV: Always try to compress incrementally (to save memory during long prefill)
        if self.is_deltakv_family:
            self._deltakv_eviction(seqs)
            return

        # SnapKV / PyramidKV: Only evict at the end of prefill
        is_last_chunk = any(seq.is_last_chunk_prefill for seq in seqs)
        if not is_last_chunk:
            return

        if self.sparse_method == "pyramidkv" and getattr(self.cache_manager, "prefill_staging_was_active", lambda: False)():
            return

        if self.sparse_method == 'snapkv' or self.sparse_method == 'pyramidkv':
            self._snapkv_prefill_eviction(seqs)
        if self.sparse_method in ("streamingllm", "attention-sink", "attention_sink"):
            self._streamingllm_prefill_eviction(seqs)

    def _build_selection(self, layer_idx: int, *, is_prefill: bool, q: torch.Tensor | None = None) -> SparseSelection:
        """Return logical sparse selection only; cache managers build physical views."""
        sparse_state = self.layer_batch_sparse_states[layer_idx]
        ctx = get_context()
        is_dynamic_deltakv = self.is_deltakv_family
        if ((self.sparse_method == "omnikv" or is_dynamic_deltakv) and layer_idx in self.full_attn_layers) or \
            self.sparse_method in (
                'snapkv',
                'pyramidkv',
                'quest',
                'rkv',
                'skipkv',
                'streamingllm',
                'attention-sink',
                'attention_sink',
                '',
            ):
            return SparseSelection(
                kind="full",
                req_indices=sparse_state.global_req_indices
                if is_dynamic_deltakv and layer_idx in self.full_attn_layers
                else sparse_state.req_indices,
                context_lens=sparse_state.context_lens,
                max_context_len=sparse_state.max_context_len,
                attn_score=sparse_state.attn_score,
                global_req_indices=sparse_state.global_req_indices,
            )

        assert layer_idx not in self.full_attn_layers
        if is_dynamic_deltakv:
            # active_compressed_indices: (B, Kmax), padded with -1; may be None (treated as K=0)
            active = sparse_state.active_compressed_indices
            # For DeltaKV we always use a batch-major Req->slots table, so kernels use local req indices.
            chunk_lens = None
            if is_prefill:
                if ctx.cu_seqlens_q is None or ctx.cu_seqlens_q.numel() <= 1:
                    chunk_lens = None
                else:
                    chunk_lens = (ctx.cu_seqlens_q[1:] - ctx.cu_seqlens_q[:-1]).to(torch.int32)

            return SparseSelection(
                kind="deltakv",
                req_indices=sparse_state.global_req_indices,
                context_lens=sparse_state.context_lens,
                max_context_len=sparse_state.max_context_len,
                attn_score=sparse_state.attn_score,
                active_compressed_indices=active,
                global_req_indices=sparse_state.global_req_indices,
                chunk_lens=chunk_lens,
                release_temp_slots=sparse_state.deltakv_free_temp_slots,
            )

        if self.sparse_method == 'omnikv':
            if sparse_state.active_slots is not None:
                active_slots = sparse_state.active_slots
                logger.debug('active_slots 是被 omnikv 选到的 slots')
            else:
                active_slots = None
                logger.debug('active_slots is None')

            return SparseSelection(
                kind="slots",
                req_indices=sparse_state.req_indices,
                context_lens=sparse_state.context_lens,
                max_context_len=sparse_state.max_context_len,
                attn_score=sparse_state.attn_score,
                active_indices=sparse_state.active_indices,
                active_slots=active_slots,
                global_req_indices=sparse_state.global_req_indices,
            )

        raise RuntimeError(f"Unsupported sparse selection path: method={self.sparse_method!r} layer={layer_idx}")

    def get_prefill_selection(self, layer_idx: int) -> SparseSelection:
        return self._build_selection(layer_idx, is_prefill=True)

    def get_decode_selection(
        self,
        layer_idx: int,
        q: torch.Tensor,
        active_slots: torch.Tensor | None = None,
        req_indices: torch.Tensor | None = None,
        context_lens: torch.Tensor | None = None,
    ) -> SparseSelection:
        del active_slots, req_indices, context_lens
        return self._build_selection(layer_idx, is_prefill=False, q=q)

    def on_layer_end(self, layer_idx: int, context):
        """每一层结束后的动态策略 (如 OmniKV / DeltaKV)"""
        if get_context().is_long_text is False and not self.is_deltakv_family:
            self._debug_record_dynamic_selection("on_layer_end", layer_idx, skipped="short_text")
            return

        is_dynamic_deltakv = self.is_deltakv_family
        if self.sparse_method != 'omnikv' and not is_dynamic_deltakv:
            self._debug_record_dynamic_selection(
                "on_layer_end",
                layer_idx,
                skipped="method",
                method=str(self.sparse_method),
                is_dynamic_deltakv=bool(is_dynamic_deltakv),
            )
            return

        if context.is_prefill:
            self._debug_record_dynamic_selection("on_layer_end", layer_idx, skipped="prefill_full_attention")
            return

        if layer_idx not in self.obs_layer_ids:
            self._debug_record_dynamic_selection("on_layer_end", layer_idx, skipped="not_obs")
            return

        self._debug_record_dynamic_selection(
            "on_layer_end",
            layer_idx,
            skipped="",
            method=str(self.sparse_method),
            is_prefill=bool(context.is_prefill),
            is_dynamic_deltakv=bool(is_dynamic_deltakv),
        )
        with profiler.record("sparse_on_layer_end"):
            state = self.layer_batch_sparse_states[layer_idx]
            if state.attn_score is None:
                raise ValueError("Attn Score hasn't been initialized")

            if state.attn_score.dim() == 3:
                if context.is_prefill:
                    chunk_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                    state.attn_score /= chunk_lens.view(-1, 1, 1)  # 不除其实也无所谓
                    # HF DeltaKV prefill uses raw QK averaged over valid queries,
                    # then max over heads.
                    state.attn_score = state.attn_score.max(dim=1).values
                elif is_dynamic_deltakv:
                    # HF DeltaKV decode applies softmax over compressed candidates,
                    # then max over heads. The kernel returns raw QK logits.
                    scores = state.attn_score
                    compressed_lens = self.cache_manager.get_compressed_lens(state.req_indices)
                    state.attn_score = self._decode_softmax_token_scores(
                        scores,
                        candidate_start=self.num_sink,
                        candidate_lens=compressed_lens,
                    )
                else:
                    # OmniKV decode applies softmax over the searchable history
                    # excluding fixed sink and recent tokens, then max over heads.
                    hist_lens = (state.context_lens - self.num_recent).clamp_min(self.num_sink)
                    state.attn_score = self._decode_softmax_token_scores(
                        state.attn_score,
                        candidate_start=self.num_sink,
                        candidate_lens=hist_lens - self.num_sink,
                    )

            target_layers = []
            for j in range(layer_idx + 1, self.num_layers):
                if j in self.full_attn_layers: break
                target_layers.append(j)
            assert len(target_layers) > 0

            self._update_dynamic_omnikv_indices(layer_idx, target_layers)

    @torch.no_grad()
    def _deltakv_eviction(self, seqs: list[Sequence]):
        assert get_context().is_long_text or self.is_deltakv_family
        self.cache_manager.deltakv_evict(seqs)

    @torch.no_grad()
    def _snapkv_prefill_eviction(self, seqs: list[Sequence]):
        for layer_idx in range(self.num_layers):
            state = self.layer_batch_sparse_states[layer_idx]
            budget = self._get_layer_budget(layer_idx, is_prefill=True)
            if budget is None:
                continue
            for b_idx, seq in enumerate(seqs):
                if not seq.is_last_chunk_prefill:
                    continue
                kv_len = int(state.context_lens[b_idx])
                if kv_len <= budget:
                    continue
                seq_scores = self.cache_manager.pop_prefill_attention_score(layer_idx, seq)
                if seq_scores is None:
                    raise RuntimeError(
                        "SnapKV/PyramidKV prefill eviction requires prefill attention scores. "
                        f"method={self.sparse_method} layer={layer_idx} seq_id={seq.seq_id}"
                    )
                if seq_scores.dim() == 2:
                    seq_scores = seq_scores.max(dim=0).values
                if log_level == 'DEBUG':
                    logger.debug(
                        "[SnapKV] prefill eviction: "
                        f"layer={layer_idx} seq_id={seq.seq_id} kv_len={kv_len} budget={budget}"
                    )
                keep_indices = self._snapkv_select_indices(
                    seq_scores[:kv_len],
                    kv_len,
                    budget,
                    pool_kernel_size=int(getattr(self.config, "pool_kernel_size", 1) or 1),
                )
                self.cache_manager.free_part_slots(layer_idx, seq, keep_indices)

    @torch.no_grad()
    def _snapkv_decode_eviction(self, seqs: list[Sequence]):
        with profiler.record("snapkv_decode_eviction"):
            pending_compactions: dict[
                tuple[tuple[int, ...], tuple[int, ...]],
                list[tuple[int, list[Sequence], torch.Tensor]],
            ] = {}
            can_compact_layers = (
                self.sparse_method == "snapkv"
                and hasattr(self.cache_manager, "free_part_slots_batch_layers")
            )

            for layer_idx in range(self.num_layers):
                state = self.layer_batch_sparse_states[layer_idx]
                attn_scores = state.attn_score
                if attn_scores is None:
                    continue

                budget = self._get_layer_budget(layer_idx, is_prefill=False)
                if budget is None:
                    continue

                trigger_len = self._snapkv_decode_trigger_len(budget)
                max_context_len = state.max_context_len
                if max_context_len is not None and (
                    int(max_context_len) <= int(budget) or int(max_context_len) < int(trigger_len)
                ):
                    continue

                kv_len_fn = getattr(self.cache_manager, "decode_kv_lens_for_layer", None)
                if kv_len_fn is not None:
                    kv_lens = kv_len_fn(layer_idx, seqs)
                else:
                    kv_lens = [int(state.context_lens[b_idx]) for b_idx in range(len(seqs))]
                triggered: list[tuple[int, Sequence, int]] = []
                for b_idx, (seq, kv_len) in enumerate(zip(seqs, kv_lens)):
                    if kv_len <= budget or kv_len < trigger_len:
                        continue
                    triggered.append((b_idx, seq, kv_len))

                if not triggered:
                    continue
                if attn_scores.dim() == 3:
                    with profiler.record("snapkv_decode_score_reduce"):
                        attn_scores = attn_scores.max(dim=1).values

                by_kv_len: dict[int, list[tuple[int, Sequence]]] = {}
                for b_idx, seq, kv_len in triggered:
                    by_kv_len.setdefault(int(kv_len), []).append((b_idx, seq))

                for kv_len, group in by_kv_len.items():
                    if log_level == 'DEBUG':
                        for _b_idx, seq in group:
                            logger.debug(
                                "[SnapKV] decode eviction: "
                                f"layer={layer_idx} seq_id={seq.seq_id} kv_len={kv_len} budget={budget} trigger_len={trigger_len}"
                            )
                    if len(group) == 1:
                        b_idx, seq = group[0]
                        with profiler.record("snapkv_decode_select"):
                            keep_indices = self._snapkv_select_indices(
                                attn_scores[b_idx, :kv_len], kv_len, budget
                            )
                        with profiler.record("snapkv_decode_compact"):
                            self.cache_manager.free_part_slots(layer_idx, seq, keep_indices)
                        continue

                    batch_indices = torch.tensor(
                        [b_idx for b_idx, _seq in group],
                        dtype=torch.long,
                        device=attn_scores.device,
                    )
                    with profiler.record("snapkv_decode_select"):
                        keep_indices = self._snapkv_select_indices_batch(
                            attn_scores.index_select(0, batch_indices)[:, :kv_len],
                            kv_len,
                            budget,
                    )
                    free_batch = getattr(self.cache_manager, "free_part_slots_batch", None)
                    group_seqs = [seq for _b_idx, seq in group]
                    if can_compact_layers:
                        key = (
                            tuple(int(seq.seq_id) for seq in group_seqs),
                            tuple(int(dim) for dim in keep_indices.shape),
                        )
                        pending_compactions.setdefault(key, []).append((layer_idx, group_seqs, keep_indices))
                    else:
                        with profiler.record("snapkv_decode_compact"):
                            if free_batch is None:
                                for row_idx, (_b_idx, seq) in enumerate(group):
                                    self.cache_manager.free_part_slots(layer_idx, seq, keep_indices[row_idx])
                            else:
                                free_batch(layer_idx, group_seqs, keep_indices)

            if pending_compactions:
                free_layers = getattr(self.cache_manager, "free_part_slots_batch_layers")
                for entries in pending_compactions.values():
                    if len(entries) == 1:
                        layer_idx, group_seqs, keep_indices = entries[0]
                        with profiler.record("snapkv_decode_compact"):
                            self.cache_manager.free_part_slots_batch(layer_idx, group_seqs, keep_indices)
                        continue
                    layer_indices = [layer_idx for layer_idx, _group_seqs, _keep_indices in entries]
                    group_seqs = entries[0][1]
                    keep_indices = torch.stack([entry[2] for entry in entries], dim=0)
                    with profiler.record("snapkv_decode_compact_layers"):
                        free_layers(layer_indices, group_seqs, keep_indices)

    @torch.no_grad()
    def _rkv_decode_eviction(self, seqs: list[Sequence]):
        self._joint_decode_eviction(
            seqs,
            profiler_name="rkv_decode_eviction",
            select_fn_name="select_rkv_indices",
            interval=int(self.config.rkv_compression_interval),
        )

    @torch.no_grad()
    def _skipkv_decode_eviction(self, seqs: list[Sequence]):
        self._joint_decode_eviction(
            seqs,
            profiler_name="skipkv_decode_eviction",
            select_fn_name="select_skipkv_indices",
            interval=int(self.config.skipkv_compression_interval),
        )

    @torch.no_grad()
    def _joint_decode_eviction(
        self,
        seqs: list[Sequence],
        *,
        profiler_name: str,
        select_fn_name: str,
        interval: int,
    ):
        budget = self._get_joint_decode_budget()
        if budget is None:
            return
        trigger_len = int(budget) + int(interval)
        select_fn = getattr(self.cache_manager, select_fn_name, None)
        if select_fn is None:
            raise RuntimeError(
                f"Cache manager {type(self.cache_manager).__name__} does not implement {select_fn_name}."
            )
        use_query_cache_scores = self.sparse_method == "rkv"
        query_score_fn = getattr(self.cache_manager, "rkv_query_attention_scores", None)
        query_score_batch_fn = getattr(self.cache_manager, "rkv_query_attention_scores_batch", None)
        if use_query_cache_scores and query_score_fn is None:
            raise RuntimeError(
                f"Cache manager {type(self.cache_manager).__name__} does not implement rkv_query_attention_scores."
            )
        select_batch_fn = getattr(self.cache_manager, f"{select_fn_name}_batch", None)
        free_batch = getattr(self.cache_manager, "free_part_slots_batch", None)
        free_layers = getattr(self.cache_manager, "free_part_slots_batch_layers", None)

        with profiler.record(profiler_name):
            pending_layer_compactions: dict[
                tuple[tuple[int, ...], int],
                tuple[list[Sequence], list[int], list[torch.Tensor]],
            ] = {}
            for layer_idx in range(self.num_layers):
                state = self.layer_batch_sparse_states[layer_idx]
                attn_scores = None
                if not use_query_cache_scores:
                    attn_scores = state.attn_score
                    if attn_scores is None:
                        continue

                triggered: list[tuple[int, Sequence, int]] = []
                for b_idx, seq in enumerate(seqs):
                    kv_len = int(state.context_lens[b_idx])
                    if kv_len <= budget or kv_len < trigger_len:
                        continue
                    triggered.append((b_idx, seq, kv_len))

                if not triggered:
                    continue

                if attn_scores is not None and attn_scores.dim() == 3:
                    attn_scores = self._decode_softmax_token_scores(
                        attn_scores,
                        candidate_start=self.num_sink,
                        candidate_lens=(state.context_lens - self.num_sink).clamp_min(0),
                    )

                batch_importance_scores = None
                if use_query_cache_scores and query_score_batch_fn is not None:
                    batch_importance_scores = query_score_batch_fn(
                        layer_idx,
                        [seq for _, seq, _ in triggered],
                        [kv_len for _, _, kv_len in triggered],
                        candidate_start=self.num_sink,
                        num_recent_tokens=self.num_recent,
                    )
                batch_keep_indices = None
                triggered_seqs = [seq for _, seq, _ in triggered]
                triggered_kv_lens = [kv_len for _, _, kv_len in triggered]
                if (
                    select_batch_fn is not None
                    and len(set(int(kv_len) for kv_len in triggered_kv_lens)) == 1
                ):
                    if batch_importance_scores is not None:
                        select_importance_scores = batch_importance_scores
                    elif attn_scores is not None:
                        batch_indices = torch.tensor(
                            [b_idx for b_idx, _seq, _kv_len in triggered],
                            dtype=torch.long,
                            device=attn_scores.device,
                        )
                        select_importance_scores = attn_scores.index_select(0, batch_indices)
                    else:
                        select_importance_scores = None
                else:
                    select_importance_scores = None
                if select_importance_scores is not None:
                    batch_keep_indices = select_batch_fn(
                        layer_idx,
                        triggered_seqs,
                        select_importance_scores,
                        triggered_kv_lens,
                        budget,
                    )
                keep_batch: list[torch.Tensor] = []
                seq_batch: list[Sequence] = []
                for local_trigger_idx, (b_idx, seq, kv_len) in enumerate(triggered):
                    if log_level == 'DEBUG':
                        logger.debug(
                            "[{}] decode eviction: layer={} seq_id={} kv_len={} budget={} trigger_len={}",
                            self.sparse_method,
                            layer_idx,
                            seq.seq_id,
                            kv_len,
                            budget,
                            trigger_len,
                        )
                    if batch_keep_indices is not None:
                        keep_indices = batch_keep_indices[local_trigger_idx]
                    else:
                        if batch_importance_scores is not None:
                            importance_scores = batch_importance_scores[local_trigger_idx, :kv_len]
                        elif use_query_cache_scores:
                            importance_scores = query_score_fn(
                                layer_idx,
                                seq,
                                kv_len,
                                candidate_start=self.num_sink,
                                num_recent_tokens=self.num_recent,
                            )
                        else:
                            importance_scores = attn_scores[b_idx, :kv_len]
                        keep_indices = select_fn(
                            layer_idx,
                            seq,
                            importance_scores,
                            kv_len,
                            budget,
                        )
                    keep_batch.append(keep_indices)
                    seq_batch.append(seq)

                use_layer_batch = (
                    use_query_cache_scores
                    and free_layers is not None
                    and len(seq_batch) > 1
                    and all(int(keep.numel()) == int(keep_batch[0].numel()) for keep in keep_batch)
                )
                if use_layer_batch:
                    key = (
                        tuple(int(seq.seq_id) for seq in seq_batch),
                        int(keep_batch[0].numel()),
                    )
                    entry = pending_layer_compactions.get(key)
                    if entry is None:
                        entry = (list(seq_batch), [], [])
                        pending_layer_compactions[key] = entry
                    entry[1].append(int(layer_idx))
                    entry[2].append(torch.stack(keep_batch, dim=0))
                elif free_batch is not None and len(seq_batch) > 1:
                    keep_indices = torch.stack(keep_batch, dim=0)
                    free_batch(layer_idx, seq_batch, keep_indices)
                else:
                    for seq, keep_indices in zip(seq_batch, keep_batch):
                        self.cache_manager.free_part_slots(layer_idx, seq, keep_indices)

            for seq_batch, layer_indices, keep_batches in pending_layer_compactions.values():
                keep_indices = torch.stack(keep_batches, dim=0)
                free_layers(layer_indices, seq_batch, keep_indices)

    @torch.no_grad()
    def _streamingllm_prefill_eviction(self, seqs: list[Sequence]):
        budget = self._get_streamingllm_budget()
        if budget is None:
            return

        with profiler.record("streamingllm_prefill_eviction"):
            free_prefix_recent = getattr(self.cache_manager, "free_prefix_recent_slots_batch_layers", None)
            free_layers = getattr(self.cache_manager, "free_part_slots_batch_layers", None)
            free_batch = getattr(self.cache_manager, "free_part_slots_batch", None)
            pending_prefix_recent: dict[
                tuple[tuple[int, ...], int],
                list[tuple[int, list[Sequence]]],
            ] = {}
            pending_compactions: dict[
                tuple[tuple[int, ...], int],
                list[tuple[int, list[Sequence], torch.Tensor]],
            ] = {}

            for layer_idx in range(self.num_layers):
                state = self.layer_batch_sparse_states[layer_idx]
                triggered: list[tuple[int, Sequence, int]] = []
                for b_idx, seq in enumerate(seqs):
                    if not seq.is_last_chunk_prefill:
                        continue
                    kv_len = int(state.context_lens[b_idx])
                    if kv_len <= budget:
                        continue
                    triggered.append((b_idx, seq, kv_len))

                if not triggered:
                    continue

                by_kv_len: dict[int, list[tuple[int, Sequence]]] = {}
                for b_idx, seq, kv_len in triggered:
                    by_kv_len.setdefault(int(kv_len), []).append((b_idx, seq))

                for kv_len, group in by_kv_len.items():
                    group_seqs = [seq for _b_idx, seq in group]
                    if free_prefix_recent is not None:
                        key = (tuple(int(seq.seq_id) for seq in group_seqs), int(kv_len))
                        pending_prefix_recent.setdefault(key, []).append((layer_idx, group_seqs))
                        continue
                    keep_indices = self._streamingllm_select_indices(kv_len).expand(len(group), -1)
                    if free_layers is not None:
                        key = (tuple(int(seq.seq_id) for seq in group_seqs), int(kv_len))
                        pending_compactions.setdefault(key, []).append((layer_idx, group_seqs, keep_indices))
                    elif free_batch is not None:
                        free_batch(layer_idx, group_seqs, keep_indices)
                    else:
                        for row_idx, (_b_idx, seq) in enumerate(group):
                            self.cache_manager.free_part_slots(layer_idx, seq, keep_indices[row_idx])

            if pending_prefix_recent:
                for (_seq_ids, kv_len), entries in pending_prefix_recent.items():
                    layer_indices = [layer_idx for layer_idx, _group_seqs in entries]
                    group_seqs = entries[0][1]
                    free_prefix_recent(
                        layer_indices,
                        group_seqs,
                        kv_len=kv_len,
                        num_sink_tokens=self.num_sink,
                        num_recent_tokens=self.num_recent,
                    )
            if pending_compactions:
                for entries in pending_compactions.values():
                    if len(entries) == 1:
                        layer_idx, group_seqs, keep_indices = entries[0]
                        if free_batch is not None:
                            free_batch(layer_idx, group_seqs, keep_indices)
                        else:
                            for row_idx, seq in enumerate(group_seqs):
                                self.cache_manager.free_part_slots(layer_idx, seq, keep_indices[row_idx])
                        continue
                    layer_indices = [layer_idx for layer_idx, _group_seqs, _keep_indices in entries]
                    group_seqs = entries[0][1]
                    keep_indices = torch.stack([entry[2] for entry in entries], dim=0)
                    free_layers(layer_indices, group_seqs, keep_indices)

    @torch.no_grad()
    def _streamingllm_decode_eviction(self, seqs: list[Sequence]):
        budget = self._get_streamingllm_budget()
        if budget is None:
            return
        trigger_len = int(2.0 * budget)

        with profiler.record("streamingllm_decode_eviction"):
            free_prefix_recent = getattr(self.cache_manager, "free_prefix_recent_slots_batch_layers", None)
            free_layers = getattr(self.cache_manager, "free_part_slots_batch_layers", None)
            free_batch = getattr(self.cache_manager, "free_part_slots_batch", None)
            pending_prefix_recent: dict[
                tuple[tuple[int, ...], int],
                list[tuple[int, list[Sequence]]],
            ] = {}
            pending_compactions: dict[
                tuple[tuple[int, ...], int],
                list[tuple[int, list[Sequence], torch.Tensor]],
            ] = {}

            for layer_idx in range(self.num_layers):
                state = self.layer_batch_sparse_states[layer_idx]
                max_context_len = state.max_context_len
                if max_context_len is not None and (
                    int(max_context_len) <= int(budget) or int(max_context_len) < int(trigger_len)
                ):
                    continue
                kv_len_fn = getattr(self.cache_manager, "decode_kv_lens_for_layer", None)
                if kv_len_fn is not None:
                    kv_lens = kv_len_fn(layer_idx, seqs)
                else:
                    kv_lens = [int(state.context_lens[b_idx]) for b_idx in range(len(seqs))]

                triggered: list[tuple[int, Sequence, int]] = []
                for b_idx, (seq, kv_len) in enumerate(zip(seqs, kv_lens)):
                    if kv_len <= budget or kv_len < trigger_len:
                        continue
                    triggered.append((b_idx, seq, int(kv_len)))

                if not triggered:
                    continue

                by_kv_len: dict[int, list[tuple[int, Sequence]]] = {}
                for b_idx, seq, kv_len in triggered:
                    by_kv_len.setdefault(int(kv_len), []).append((b_idx, seq))

                for kv_len, group in by_kv_len.items():
                    if log_level == 'DEBUG':
                        for _b_idx, seq in group:
                            logger.debug(
                                "[StreamingLLM] decode eviction: "
                                f"layer={layer_idx} seq_id={seq.seq_id} kv_len={kv_len} "
                                f"budget={budget} trigger_len={trigger_len}"
                            )
                    group_seqs = [seq for _b_idx, seq in group]
                    if free_prefix_recent is not None:
                        key = (tuple(int(seq.seq_id) for seq in group_seqs), int(kv_len))
                        pending_prefix_recent.setdefault(key, []).append((layer_idx, group_seqs))
                        continue
                    keep_indices = self._streamingllm_select_indices(kv_len).expand(len(group), -1)
                    if free_layers is not None:
                        key = (tuple(int(seq.seq_id) for seq in group_seqs), int(kv_len))
                        pending_compactions.setdefault(key, []).append((layer_idx, group_seqs, keep_indices))
                    elif free_batch is not None:
                        free_batch(layer_idx, group_seqs, keep_indices)
                    else:
                        for row_idx, (_b_idx, seq) in enumerate(group):
                            self.cache_manager.free_part_slots(layer_idx, seq, keep_indices[row_idx])

            if pending_prefix_recent:
                for (_seq_ids, kv_len), entries in pending_prefix_recent.items():
                    layer_indices = [layer_idx for layer_idx, _group_seqs in entries]
                    group_seqs = entries[0][1]
                    free_prefix_recent(
                        layer_indices,
                        group_seqs,
                        kv_len=kv_len,
                        num_sink_tokens=self.num_sink,
                        num_recent_tokens=self.num_recent,
                    )
            if pending_compactions:
                for entries in pending_compactions.values():
                    if len(entries) == 1:
                        layer_idx, group_seqs, keep_indices = entries[0]
                        if free_batch is not None:
                            free_batch(layer_idx, group_seqs, keep_indices)
                        else:
                            for row_idx, seq in enumerate(group_seqs):
                                self.cache_manager.free_part_slots(layer_idx, seq, keep_indices[row_idx])
                        continue
                    layer_indices = [layer_idx for layer_idx, _group_seqs, _keep_indices in entries]
                    group_seqs = entries[0][1]
                    keep_indices = torch.stack([entry[2] for entry in entries], dim=0)
                    free_layers(layer_indices, group_seqs, keep_indices)

    def _get_streamingllm_budget(self) -> int | None:
        budget = self.num_sink + self.num_recent
        if budget <= 0:
            return None
        return budget

    def _streamingllm_select_indices(self, kv_len: int) -> torch.Tensor:
        assert kv_len > 0
        device = self.device
        sink_end = min(self.num_sink, kv_len)
        recent_start = max(sink_end, kv_len - self.num_recent)
        sink_indices = torch.arange(sink_end, device=device, dtype=torch.long)
        recent_indices = torch.arange(recent_start, kv_len, device=device, dtype=torch.long)
        return torch.cat([sink_indices, recent_indices], dim=0)

    def _snapkv_select_indices(
        self,
        scores: torch.Tensor,
        kv_len: int,
        budget: int,
        *,
        pool_kernel_size: int = 1,
    ) -> torch.Tensor:
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
            pool_kernel_size = int(pool_kernel_size)
            if pool_kernel_size > 1:
                middle_scores = F.max_pool1d(
                    middle_scores[None, None, :],
                    kernel_size=pool_kernel_size,
                    padding=pool_kernel_size // 2,
                    stride=1,
                ).squeeze(0).squeeze(0)
            topk_indices_relative = middle_scores.topk(min(num_topk, middle_scores.shape[0]), dim=-1).indices
            topk_indices = topk_indices_relative + self.num_sink
            keep_indices = torch.cat([sink_indices, topk_indices, recent_indices])
        else:
            keep_indices = torch.cat([sink_indices, recent_indices])
            
        return keep_indices

    def _snapkv_select_indices_batch(
        self,
        scores: torch.Tensor,
        kv_len: int,
        budget: int,
        *,
        pool_kernel_size: int = 1,
    ) -> torch.Tensor:
        if scores.dim() != 2:
            raise ValueError(f"Expected batched SnapKV scores with shape [B, L], got {tuple(scores.shape)}.")
        assert kv_len > budget
        if int(scores.shape[1]) < int(kv_len):
            raise ValueError(
                f"SnapKV batched scores are shorter than kv_len: scores={tuple(scores.shape)} kv_len={kv_len}."
            )
        device = scores.device
        batch_size = int(scores.shape[0])

        sink_indices = torch.arange(self.num_sink, device=device).expand(batch_size, -1)
        recent_start = kv_len - self.num_recent
        recent_indices = torch.arange(recent_start, kv_len, device=device).expand(batch_size, -1)

        num_topk = budget - self.num_sink - self.num_recent
        if num_topk > 0 and recent_start > self.num_sink:
            middle_scores = scores[:, self.num_sink:recent_start]
            pool_kernel_size = int(pool_kernel_size)
            if pool_kernel_size > 1:
                middle_scores = F.max_pool1d(
                    middle_scores[:, None, :],
                    kernel_size=pool_kernel_size,
                    padding=pool_kernel_size // 2,
                    stride=1,
                ).squeeze(1)
            topk_indices_relative = middle_scores.topk(
                min(num_topk, middle_scores.shape[1]),
                dim=-1,
            ).indices
            topk_indices = topk_indices_relative + self.num_sink
            return torch.cat([sink_indices, topk_indices, recent_indices], dim=1)
        return torch.cat([sink_indices, recent_indices], dim=1)

    def _get_joint_decode_budget(self) -> int | None:
        budget = int(self.num_sink) + int(self.decode_keep_tokens) + int(self.num_recent)
        if budget <= 0:
            return None
        return budget

    def _update_dynamic_omnikv_indices(self, obs_layer_idx, target_layers):
        assert get_context().is_long_text or self.is_deltakv_family

        with profiler.record("sparse_update_dynamic_indices"):
            ctx = get_context()
            is_dynamic_deltakv = self.is_deltakv_family
            self._debug_record_dynamic_selection(
                "update_dynamic",
                obs_layer_idx,
                method=str(self.sparse_method),
                is_prefill=bool(ctx.is_prefill),
                is_dynamic_deltakv=bool(is_dynamic_deltakv),
                target_layers=[int(x) for x in target_layers],
            )
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
            elif is_dynamic_deltakv:
                rel_hist_lens = self.cache_manager.get_compressed_lens(obs_sparse_state.req_indices)
            else:
                raise ValueError

            # 2. 掩码处理 (处理不等长 + 防止 topk 选到 buffer/chunk 区域)
            mask = torch.arange(search_scores.size(1), device=self.device) >= rel_hist_lens.unsqueeze(1)
            search_scores.masked_fill_(mask, -1e10)
            if (
                self.dynamic_deltakv_topk_tiebreak
                and is_dynamic_deltakv
                and not ctx.is_prefill
                and search_scores.numel() > 0
            ):
                # BF16 score matching creates many exact ties.  CUDA graph replay
                # and eager topk can pick different tied tokens, so add a tiny
                # deterministic position key that is far below the BF16 score
                # quantum but visible to float32 topk.
                pos_key = torch.arange(search_scores.size(1), device=search_scores.device, dtype=torch.float32)
                pos_key = pos_key / max(1, int(search_scores.size(1)))
                score_scale = search_scores.detach().abs().float().clamp_min(1.0)
                search_scores = search_scores.float() + score_scale * (pos_key.unsqueeze(0) * 1.0e-6)
                search_scores.masked_fill_(mask, -1e10)

            # 3. 提取 Top-K. DeltaKV keeps the original ragged per-row budget path;
            # OmniKV below uses a fixed padded K to avoid per-decode CPU/GPU syncs.
            decode_keep = self.decode_keep_tokens
            if is_dynamic_deltakv:
                if not ctx.is_prefill:
                    k_max = min(int(decode_keep), int(search_scores.size(1)))
                    if k_max > 0:
                        topk_indices = search_scores.topk(k_max, dim=1, sorted=True).indices.to(torch.int32)
                    else:
                        topk_indices = torch.empty((batch_size, 0), device=self.device, dtype=torch.int32)
                else:
                    topk_list = []
                    k_list = []
                    for b in range(batch_size):
                        avail = int(rel_hist_lens[b].item())
                        k_b = min(int(decode_keep), int(search_scores.size(1)), max(0, avail))
                        k_list.append(k_b)
                        if k_b <= 0:
                            topk_list.append(torch.empty((0,), device=self.device, dtype=torch.int32))
                        else:
                            idx = search_scores[b].topk(k_b, dim=0).indices.to(torch.int32)
                            topk_list.append(idx)
                    k_max = max(k_list) if k_list else 0
                    if k_max > 0:
                        topk_indices = torch.full((batch_size, k_max), -1, device=self.device, dtype=torch.int32)
                        for b in range(batch_size):
                            k_b = k_list[b]
                            if k_b > 0:
                                topk_indices[b, :k_b] = topk_list[b]
                    else:
                        topk_indices = torch.empty((batch_size, 0), device=self.device, dtype=torch.int32)
                if self.debug_dynamic_selection_detail:
                    debug_k = min(16, int(search_scores.shape[1]))
                    detail = {
                        "rel_hist_lens_preview": self._debug_tensor_preview(rel_hist_lens, 16),
                        "search_scores_shape": tuple(int(x) for x in search_scores.shape),
                        "topk_shape": tuple(int(x) for x in topk_indices.shape),
                        "topk_rel_preview": self._debug_tensor_preview(topk_indices, 32),
                        "topk_abs_preview": self._debug_tensor_preview(topk_indices + int(self.num_sink), 32),
                    }
                    if debug_k > 0:
                        debug_scores, debug_rel = search_scores.topk(debug_k, dim=1, sorted=True)
                        detail.update(
                            search_top_rel_preview=self._debug_tensor_preview(debug_rel, 32),
                            search_top_abs_preview=self._debug_tensor_preview(debug_rel + int(self.num_sink), 32),
                            search_top_score_preview=self._debug_tensor_preview(debug_scores, 32),
                        )
                    self._debug_record_dynamic_selection("dynamic_topk_detail", obs_layer_idx, **detail)
            else:
                topk_indices = None

            # 4. 根据方法更新目标层状态
            if self.sparse_method == 'omnikv':
                local_req_indices = torch.arange(batch_size, dtype=torch.int32, device=self.device)
                decode_keep = int(decode_keep)
                k_max = min(decode_keep, int(search_scores.size(1)))
                if k_max > 0:
                    topk_lens = rel_hist_lens.clamp(min=0, max=k_max).to(torch.int32)
                    topk_indices = search_scores.topk(k_max, dim=1, sorted=False).indices.to(torch.int32) + self.num_sink
                else:
                    topk_lens = torch.zeros((batch_size,), dtype=torch.int32, device=self.device)
                    topk_indices = torch.empty((batch_size, 0), device=self.device, dtype=torch.int32)

                if ctx.is_prefill:
                    chunk_lens = ctx.cu_seqlens_q[1:] - ctx.cu_seqlens_q[:-1]
                    max_recent_or_chunk = int(chunk_lens.max().item()) + int(self.num_recent)
                else:
                    max_recent_or_chunk = int(self.num_recent)
                max_sparse_context_len = int(self.num_sink) + int(k_max) + max_recent_or_chunk
                keep_indices, active_slots, new_context_lens = build_omnikv_keep_and_slots(
                    topk_indices,
                    topk_lens,
                    hist_lens,
                    obs_sparse_state.context_lens - hist_lens,  # lens of recent and chunk
                    self.cache_manager.get_layer_buffer_req_to_token_slots(obs_layer_idx + 1),
                    obs_sparse_state.req_indices,
                    self.num_sink,
                    max_s=max_sparse_context_len,
                )

                for l_idx in target_layers:
                    target_sparse_state = self.layer_batch_sparse_states[l_idx]
                    target_sparse_state.active_indices = keep_indices
                    target_sparse_state.active_slots = active_slots
                    target_sparse_state.context_lens = new_context_lens
                    target_sparse_state.max_context_len = max_sparse_context_len
                    target_sparse_state.req_indices = local_req_indices
            
            elif is_dynamic_deltakv:
                for l_idx in target_layers:
                    target_sparse_state = self.layer_batch_sparse_states[l_idx]
                    target_sparse_state.active_compressed_indices = topk_indices
                    # context_lens is finalized when cache_manager builds the DeltaKV compute view.
                    target_sparse_state.context_lens = obs_sparse_state.context_lens
                    target_sparse_state.req_indices = obs_sparse_state.req_indices
                    target_sparse_state.global_req_indices = obs_sparse_state.req_indices
                    target_sparse_state.deltakv_free_temp_slots = (l_idx == target_layers[-1])
            else:
                raise ValueError

    def _needs_attn_score(self, layer_idx: int, is_prefill: bool, seqs: list[Sequence]) -> bool:
        is_dynamic_deltakv = self.is_deltakv_family
        if self.sparse_method == 'omnikv' and get_context().is_long_text is False:
            return False
        if (self.sparse_method == 'omnikv' or is_dynamic_deltakv) and layer_idx in self.obs_layer_ids:
            if is_prefill:
                return False
            return True
        if self.sparse_method in ('snapkv', 'pyramidkv'):
            if is_prefill:
                return False

            # Decode: only collect scores when we're about to evict.
            budget = self._get_layer_budget(layer_idx, is_prefill=False)
            if budget is None:
                return False
            trigger_len = self._snapkv_decode_trigger_len(budget)
            kv_lens_fn = getattr(self.cache_manager, "decode_kv_lens_for_layer", None)
            if kv_lens_fn is not None:
                kv_lens = kv_lens_fn(layer_idx, seqs)
                return any(int(kv_len) >= int(trigger_len) and int(kv_len) > int(budget) for kv_len in kv_lens)

            state = self.layer_batch_sparse_states[layer_idx]
            if state.context_lens is None:
                return False
            max_context_len = state.max_context_len
            if max_context_len is not None:
                return int(max_context_len) >= int(trigger_len) and int(max_context_len) > int(budget)
            return bool(((state.context_lens >= trigger_len) & (state.context_lens > budget)).any())
        if self.sparse_method == "rkv":
            return False
        if self.sparse_method == "skipkv":
            if is_prefill:
                return False
            budget = self._get_joint_decode_budget()
            if budget is None:
                return False
            if bool(getattr(self.config, "decode_cuda_graph", False)):
                # Graph replay reuses the tensors captured on the first decode
                # step.  SkipKV eviction may trigger only after more tokens are
                # generated, so capture the score path up front instead of
                # silently losing later score-dependent evictions.
                return True
            state = self.layer_batch_sparse_states[layer_idx]
            if state.context_lens is None:
                return False
            interval = int(self.config.skipkv_compression_interval)
            trigger_len = int(budget) + int(interval)
            return bool(((state.context_lens >= trigger_len) & (state.context_lens > budget)).any())
        return False

    def _get_layer_budget(self, layer_idx: int, is_prefill: bool) -> int | None:
        if layer_idx < self.config.snapkv_num_full_layers:
            return None
        decode_keep = self.decode_keep_tokens
        if self.config.pyramid_layer_ratios is not None:
            ratio = self.config.pyramid_layer_ratios[layer_idx]
            base_ratio = self.config.pyramid_layer_ratios[0]
            scaled_top_tokens = int(decode_keep * ratio / base_ratio)
            return self.num_sink + scaled_top_tokens + self.num_recent
        elif self.sparse_method == 'snapkv':
            return self.num_sink + decode_keep + self.num_recent
        return None

    def _snapkv_decode_trigger_len(self, budget: int) -> int:
        top_budget = int(budget) - int(self.num_sink) - int(self.num_recent)
        if self.sparse_method == "pyramidkv":
            return int(budget) + int(top_budget)
        return int(2.0 * top_budget)
