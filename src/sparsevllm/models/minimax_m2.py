from __future__ import annotations

import re
from dataclasses import replace

import torch
import torch.nn.functional as F
from torch import nn

from sparsevllm.distributed import get_parallel_context
from sparsevllm.layers.attention import Attention
from sparsevllm.layers.embed_head import ParallelLMHead
from sparsevllm.layers.layernorm import FlashInferRMSNorm
from sparsevllm.layers.linear import QKVParallelLinear, RowParallelLinear
from sparsevllm.layers.rotary_embedding import (
    apply_partial_rotary_emb,
    get_rope,
)
from sparsevllm.models.qwen3 import Qwen3ModelBase
from sparsevllm.platforms import device_runtime
from sparsevllm.quantization.fp8 import (
    Fp8BlockScaledLinearBackend,
    fp8_blockwise_linear_reference,
)
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger


_EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."
    r"(w1|w2|w3)\.weight$"
)
_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."
    r"(w1|w2|w3)\.expert_weight$"
)


def _execution_backend(config) -> str:
    backend = str(getattr(config, "moe_backend", "pytorch")).strip().lower()
    if backend not in {"pytorch", "native", "triton"}:
        raise ValueError(
            "MiniMax M2 execution backend must be 'pytorch', 'native', or "
            f"'triton', got {backend!r}."
        )
    return backend


def _dense_quantization_config(config):
    quantization = config.quantization_config
    if _execution_backend(config) == "pytorch":
        return replace(quantization, backend="reference")
    return quantization


def _torch_biased_sigmoid_topk(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    routing_weights = torch.sigmoid(router_logits.float())
    scores = routing_weights + correction_bias
    _, topk_ids = torch.topk(scores, top_k, dim=-1, sorted=False)
    topk_weights = routing_weights.gather(1, topk_ids)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
    return topk_weights, topk_ids


class MiniMaxM2Router(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_local_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.backend = _execution_backend(config)
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size, dtype=torch.float32)
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32)
        )
        if self.backend == "triton":
            from sparsevllm.triton_kernel.minimax_m2_router import (
                topk_biased_sigmoid,
            )

            self.topk_impl = topk_biased_sigmoid
        else:
            self.topk_impl = _torch_biased_sigmoid_topk

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states.float(), self.weight)
        topk_weights, topk_ids = self.topk_impl(
            router_logits,
            self.e_score_correction_bias,
            top_k=self.top_k,
        )
        return router_logits, topk_weights, topk_ids


class MiniMaxM2PackedExperts(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        parallel_context = get_parallel_context()
        self.ep_rank = int(parallel_context.ep_rank)
        self.ep_size = int(parallel_context.ep_size)
        self.num_experts = int(config.num_local_experts)
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.intermediate_size)
        if self.num_experts % self.ep_size:
            raise ValueError(
                f"MiniMax experts={self.num_experts} must be divisible by EP={self.ep_size}."
            )
        if self.hidden_size % 128 or self.intermediate_size % 128:
            raise ValueError(
                "MiniMax packed FP8 experts require hidden/intermediate dimensions "
                f"aligned to 128, got {self.hidden_size}/{self.intermediate_size}."
            )
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size,
                dtype=torch.float8_e4m3fn,
            ),
            requires_grad=False,
        )
        self.register_buffer(
            "w13_scale_inv",
            torch.empty(
                self.num_local_experts,
                2 * self.intermediate_size // 128,
                self.hidden_size // 128,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "w2_scale_inv",
            torch.empty(
                self.num_local_experts,
                self.hidden_size // 128,
                self.intermediate_size // 128,
                dtype=torch.float32,
            ),
        )
        self._loaded_expert_shards: set[tuple[int, str]] = set()
        self.backend = _execution_backend(config)
        if self.backend == "pytorch":
            self.forward_impl = self.forward_reference
        elif self.backend == "native":
            quantization = config.quantization_config
            self.native_linear = Fp8BlockScaledLinearBackend(
                block_size=tuple(quantization.weight_block_size),
                backend=str(quantization.backend),
                model_name="MiniMax M2.7",
            )
            self.forward_impl = self.forward_native
        else:
            self.forward_impl = self.forward_triton

    def is_local_expert(self, global_expert_id: int) -> bool:
        return self.local_expert_start <= int(global_expert_id) < self.local_expert_end

    def load_expert_weight(
        self,
        global_expert_id: int,
        projection: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> None:
        global_expert_id = int(global_expert_id)
        if not self.is_local_expert(global_expert_id):
            raise ValueError(
                f"Expert {global_expert_id} is outside local range "
                f"[{self.local_expert_start}, {self.local_expert_end})."
            )
        if projection not in {"w1", "w2", "w3"}:
            raise ValueError(f"Unsupported MiniMax expert projection {projection!r}.")
        load_key = (global_expert_id, projection)
        if load_key in self._loaded_expert_shards:
            raise ValueError(
                f"Duplicate MiniMax expert weight for expert={global_expert_id}, "
                f"projection={projection}."
            )
        if loaded_scale is None:
            raise ValueError(
                f"Missing FP8 weight_scale_inv for MiniMax expert={global_expert_id}, "
                f"projection={projection}."
            )
        if loaded_weight.dtype != torch.float8_e4m3fn:
            raise TypeError(
                f"MiniMax expert weight must be FP8 E4M3, got {loaded_weight.dtype}."
            )
        if loaded_scale.dtype != torch.float32:
            raise TypeError(
                "MiniMax expert weight_scale_inv must be FP32, "
                f"got {loaded_scale.dtype}."
            )

        local_expert_id = global_expert_id - self.local_expert_start
        if projection == "w2":
            weight_target = self.w2_weight.data[local_expert_id]
            scale_target = self.w2_scale_inv[local_expert_id]
        else:
            weight_offset = 0 if projection == "w1" else self.intermediate_size
            scale_rows = self.intermediate_size // 128
            scale_offset = 0 if projection == "w1" else scale_rows
            weight_target = self.w13_weight.data[
                local_expert_id,
                weight_offset : weight_offset + self.intermediate_size,
            ]
            scale_target = self.w13_scale_inv[
                local_expert_id,
                scale_offset : scale_offset + scale_rows,
            ]
        if tuple(loaded_weight.shape) != tuple(weight_target.shape):
            raise ValueError(
                f"MiniMax expert weight shape mismatch for expert={global_expert_id}, "
                f"projection={projection}: expected={tuple(weight_target.shape)}, "
                f"got={tuple(loaded_weight.shape)}."
            )
        if tuple(loaded_scale.shape) != tuple(scale_target.shape):
            raise ValueError(
                f"MiniMax expert scale shape mismatch for expert={global_expert_id}, "
                f"projection={projection}: expected={tuple(scale_target.shape)}, "
                f"got={tuple(loaded_scale.shape)}."
            )
        weight_target.copy_(loaded_weight)
        scale_target.copy_(loaded_scale)
        self._loaded_expert_shards.add(load_key)

    def validate_loaded_weights(self) -> None:
        expected = {
            (expert_id, projection)
            for expert_id in range(self.local_expert_start, self.local_expert_end)
            for projection in ("w1", "w2", "w3")
        }
        missing = sorted(expected - self._loaded_expert_shards)
        if missing:
            raise ValueError(
                "Missing local MiniMax expert weights/scales: "
                f"local_range=[{self.local_expert_start}, {self.local_expert_end}), "
                f"missing={missing[:8]}."
            )

    def _dispatch_loop(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        linear,
    ) -> torch.Tensor:
        local_output = torch.zeros_like(hidden_states)
        for local_expert_id in range(self.num_local_experts):
            global_expert_id = self.local_expert_start + local_expert_id
            token_ids, topk_slots = torch.where(topk_ids == global_expert_id)
            if token_ids.numel() == 0:
                continue
            expert_input = hidden_states[token_ids]
            gate_up = linear(
                expert_input,
                self.w13_weight[local_expert_id],
                self.w13_scale_inv[local_expert_id],
            )
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = linear(
                F.silu(gate) * up,
                self.w2_weight[local_expert_id],
                self.w2_scale_inv[local_expert_id],
            )
            expert_output.mul_(
                topk_weights[token_ids, topk_slots, None].to(expert_output.dtype)
            )
            local_output.index_add_(0, token_ids, expert_output.to(local_output.dtype))
        return local_output

    def forward_reference(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._dispatch_loop(
            hidden_states,
            topk_ids,
            topk_weights,
            fp8_blockwise_linear_reference,
        )

    def forward_native(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self._dispatch_loop(
            hidden_states,
            topk_ids,
            topk_weights,
            self.native_linear,
        )

    def forward_triton(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        from sparsevllm.triton_kernel.minimax_m2_moe_fp8 import fused_moe_fp8

        return fused_moe_fp8(
            hidden_states,
            self.w13_weight,
            self.w13_scale_inv,
            self.w2_weight,
            self.w2_scale_inv,
            topk_ids,
            topk_weights,
            num_experts=self.num_experts,
            local_expert_start=self.local_expert_start,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward_impl(hidden_states, topk_ids, topk_weights)


class MiniMaxM2SparseMoeBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.gate = MiniMaxM2Router(config)
        self.experts = MiniMaxM2PackedExperts(config)

    @property
    def e_score_correction_bias(self) -> nn.Parameter:
        return self.gate.e_score_correction_bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError(
                f"MiniMax M2 MoE expects [tokens, hidden], got {tuple(hidden_states.shape)}."
            )
        _, topk_weights, topk_ids = self.gate(hidden_states)
        local_output = self.experts(hidden_states, topk_ids, topk_weights)
        return self.parallel_context.ep_all_reduce(local_output)


class MiniMaxM2Attention(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        tp_size = int(get_parallel_context().tp_size)
        self.total_num_heads = int(config.num_attention_heads)
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_heads % tp_size or self.total_num_kv_heads % tp_size:
            raise ValueError("MiniMax attention heads must be divisible by TP size.")
        self.num_heads = self.total_num_heads // tp_size
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = int(config.head_dim)
        self.rotary_dim = int(config.rotary_dim)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        quantization = _dense_quantization_config(config)
        self.qkv_proj = QKVParallelLinear(
            int(config.hidden_size),
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quantization=quantization,
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            int(config.hidden_size),
            bias=False,
            quantization=quantization,
        )
        self.q_norm = FlashInferRMSNorm(
            self.q_size,
            eps=float(config.rms_norm_eps),
        )
        self.k_norm = FlashInferRMSNorm(
            self.kv_size,
            eps=float(config.rms_norm_eps),
        )
        self.rotary_emb = get_rope(
            self.rotary_dim,
            rotary_dim=self.rotary_dim,
            max_position=int(config.max_position_embeddings),
            base=float(config.rope_theta),
            rope_scaling=None,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.head_dim**-0.5,
            self.num_kv_heads,
        )

    def forward(self, positions: torch.Tensor, hidden_states: torch.Tensor) -> torch.Tensor:
        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        context = get_context()
        layer_idx = context.now_layer_idx
        raw_k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context.cache_manager.save_raw_kv_if_needed(layer_idx, raw_k, v)
        q = self.q_norm(q).view(-1, self.num_heads, self.head_dim)
        k = self.k_norm(k).view(-1, self.num_kv_heads, self.head_dim)
        q, k = apply_partial_rotary_emb(
            self.rotary_emb,
            positions,
            q,
            k,
            self.rotary_dim,
        )
        context.cache_manager.save_rope_kv_if_needed(layer_idx, k, v)
        output = self.attn(q, k, v).flatten(1, -1)
        return self.o_proj(output)


class MiniMaxM2DecoderLayer(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.self_attn = MiniMaxM2Attention(config)
        self.block_sparse_moe = MiniMaxM2SparseMoeBlock(config)
        self.input_layernorm = FlashInferRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = FlashInferRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        if self.parallel_context.ep_size > 1:
            self.parallel_context.ep_broadcast(hidden_states, src_ep_rank=0)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.block_sparse_moe(hidden_states)
        return hidden_states, residual


class MiniMaxM2Model(Qwen3ModelBase):
    def __init__(self, config) -> None:
        super().__init__(config, MiniMaxM2DecoderLayer)
        self.norm = FlashInferRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )


class MiniMaxM2ForCausalLM(nn.Module):
    special_weight_loaders = (".expert_weight",)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.parallel_context = get_parallel_context()
        self.model = MiniMaxM2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        self._intentionally_skipped_expert_weights: set[str] = set()
        self._intentionally_skipped_expert_scales: set[str] = set()

    @torch.inference_mode()
    def warmup_moe_backend(self) -> None:
        if _execution_backend(self.config) != "triton":
            return
        layer = self.model.layers[0]
        experts = layer.block_sparse_moe.experts
        device = experts.w13_weight.device
        hidden_states = torch.zeros(
            (1, experts.hidden_size),
            dtype=torch.bfloat16,
            device=device,
        )
        layer.self_attn.qkv_proj(hidden_states)
        layer.self_attn.o_proj(
            torch.zeros(
                (1, layer.self_attn.q_size),
                dtype=hidden_states.dtype,
                device=device,
            )
        )
        layer.block_sparse_moe.gate(hidden_states)
        top_k = int(self.config.num_experts_per_tok)
        topk_ids = (
            torch.arange(top_k, dtype=torch.int64, device=device)
            .remainder(experts.num_local_experts)
            .add(experts.local_expert_start)
            .view(1, top_k)
        )
        topk_weights = torch.full(
            (1, top_k),
            1.0 / top_k,
            dtype=torch.float32,
            device=device,
        )
        experts.forward_triton(hidden_states, topk_ids, topk_weights)
        device_runtime.synchronize()

    def map_weight_name(self, source_weight_name: str) -> str | None:
        if source_weight_name.endswith(".block_sparse_moe.e_score_correction_bias"):
            return source_weight_name.replace(
                ".block_sparse_moe.e_score_correction_bias",
                ".block_sparse_moe.gate.e_score_correction_bias",
            )
        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is None:
            return source_weight_name
        layer_idx, global_expert_id, projection = match.groups()
        global_expert_id = int(global_expert_id)
        experts = self.model.layers[int(layer_idx)].block_sparse_moe.experts
        if not experts.is_local_expert(global_expert_id):
            return None
        return (
            f"model.layers.{layer_idx}.block_sparse_moe.experts."
            f"{global_expert_id}.{projection}.expert_weight"
        )

    def record_skipped_weight(
        self,
        source_weight_name: str,
        loaded_weight_shape: tuple[int, ...] | None,
        loaded_weight_dtype: str | None,
        loaded_scale_shape: tuple[int, ...] | None,
        loaded_scale_dtype: str | None,
    ) -> None:
        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is None:
            raise ValueError(
                f"MiniMax loader unexpectedly skipped {source_weight_name!r}."
            )
        layer_idx, global_expert_id, projection = match.groups()
        experts = self.model.layers[int(layer_idx)].block_sparse_moe.experts
        if experts.is_local_expert(int(global_expert_id)):
            raise ValueError(
                f"MiniMax loader skipped local expert weight {source_weight_name!r}."
            )
        if loaded_weight_shape is None or loaded_weight_dtype is None:
            raise ValueError(
                f"Skipped remote MiniMax expert is missing weight metadata: "
                f"{source_weight_name!r}."
            )
        if loaded_weight_dtype != "F8_E4M3":
            raise TypeError(
                "Remote MiniMax expert weight must be FP8 E4M3, got "
                f"safetensors dtype {loaded_weight_dtype}."
            )
        if loaded_scale_shape is None or loaded_scale_dtype is None:
            raise ValueError(
                f"Skipped remote MiniMax expert is missing weight_scale_inv: "
                f"{source_weight_name!r}."
            )
        if loaded_scale_dtype != "F32":
            raise TypeError(
                "Remote MiniMax expert scale must be FP32, got safetensors dtype "
                f"{loaded_scale_dtype}."
            )
        expected_shape = (
            (experts.hidden_size // 128, experts.intermediate_size // 128)
            if projection == "w2"
            else (experts.intermediate_size // 128, experts.hidden_size // 128)
        )
        expected_weight_shape = (
            (experts.hidden_size, experts.intermediate_size)
            if projection == "w2"
            else (experts.intermediate_size, experts.hidden_size)
        )
        if loaded_weight_shape != expected_weight_shape:
            raise ValueError(
                "Remote MiniMax expert weight shape mismatch for "
                f"{source_weight_name!r}: expected={expected_weight_shape}, "
                f"got={loaded_weight_shape}."
            )
        if loaded_scale_shape != expected_shape:
            raise ValueError(
                "Remote MiniMax expert scale shape mismatch for "
                f"{source_weight_name!r}: expected={expected_shape}, "
                f"got={loaded_scale_shape}."
            )
        self._intentionally_skipped_expert_weights.add(source_weight_name)
        self._intentionally_skipped_expert_scales.add(
            source_weight_name[: -len(".weight")] + ".weight_scale_inv"
        )

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        match = _EXPERT_TARGET_RE.match(target_weight_name)
        if match is None:
            return 0
        layer_idx, global_expert_id, projection = match.groups()
        self.model.layers[int(layer_idx)].block_sparse_moe.experts.load_expert_weight(
            int(global_expert_id),
            projection,
            loaded_weight,
            loaded_scale,
        )
        return 1

    def validate_loaded_weights(self, loaded_parameter_names: set[str]) -> None:
        packed_expert_parameters = {
            name
            for name, _ in self.named_parameters()
            if name.endswith(".block_sparse_moe.experts.w13_weight")
            or name.endswith(".block_sparse_moe.experts.w2_weight")
        }
        expected_dense = {
            name for name, _ in self.named_parameters()
        } - packed_expert_parameters
        missing_dense = sorted(expected_dense - loaded_parameter_names)
        if missing_dense:
            raise ValueError(
                f"Missing replicated MiniMax M2 weights: {missing_dense[:8]}."
            )
        for layer in self.model.layers:
            layer.block_sparse_moe.experts.validate_loaded_weights()

        expected_skipped_weights = {
            f"model.layers.{layer_idx}.block_sparse_moe.experts.{expert_id}."
            f"{projection}.weight"
            for layer_idx in range(int(self.config.num_hidden_layers))
            for expert_id in range(int(self.config.num_local_experts))
            if not self.model.layers[
                layer_idx
            ].block_sparse_moe.experts.is_local_expert(expert_id)
            for projection in ("w1", "w2", "w3")
        }
        expected_skipped_scales = {
            name[: -len(".weight")] + ".weight_scale_inv"
            for name in expected_skipped_weights
        }
        missing_skips = sorted(
            expected_skipped_weights - self._intentionally_skipped_expert_weights
        )
        unexpected_skips = sorted(
            self._intentionally_skipped_expert_weights - expected_skipped_weights
        )
        missing_scale_skips = sorted(
            expected_skipped_scales - self._intentionally_skipped_expert_scales
        )
        unexpected_scale_skips = sorted(
            self._intentionally_skipped_expert_scales - expected_skipped_scales
        )
        if missing_skips or missing_scale_skips:
            raise ValueError(
                "Checkpoint is missing expected remote MiniMax expert entries: "
                f"weights={missing_skips[:4]}, scales={missing_scale_skips[:4]}."
            )
        if unexpected_skips or unexpected_scale_skips:
            raise ValueError(
                "Unexpectedly skipped MiniMax expert entries: "
                f"weights={unexpected_skips[:4]}, scales={unexpected_scale_skips[:4]}."
            )
        logger.info(
            "Loaded MiniMax M2 rank {} local experts [{}, {}) across {} layers; "
            "intentionally skipped {} remote expert weight/scale pairs.",
            self.parallel_context.world_rank,
            self.model.layers[0].block_sparse_moe.experts.local_expert_start,
            self.model.layers[0].block_sparse_moe.experts.local_expert_end,
            len(self.model.layers),
            len(self._intentionally_skipped_expert_weights),
        )

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
