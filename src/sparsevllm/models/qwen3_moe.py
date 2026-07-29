from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Qwen3MoeConfig

from sparsevllm.distributed import get_parallel_context
from sparsevllm.layers.embed_head import ParallelLMHead
from sparsevllm.models.qwen3 import Qwen3DecoderLayerBase, Qwen3ModelBase
from sparsevllm.operators.moe import (
    MoeOpSpec,
    model_activation_dtype,
    resolve_moe_provider,
)
from sparsevllm.platforms import device_runtime
from sparsevllm.quantization.fp8_tp import Fp8ExpertTpShard
from sparsevllm.utils.log import logger


_EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.expert_weight$"
)


class Qwen3MoeRouter(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.norm_topk_prob = bool(config.norm_topk_prob)
        from sparsevllm.triton_kernel.moe_topk import topk_softmax

        self.topk_impl = topk_softmax
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states, self.weight)
        topk_weights, topk_ids = self.topk_impl(
            router_logits,
            top_k=self.top_k,
            norm_topk_prob=self.norm_topk_prob,
        )
        return router_logits, topk_weights, topk_ids


class Qwen3MoePackedExperts(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        parallel_context = get_parallel_context()
        self.tp_rank = parallel_context.moe_tp_rank
        self.tp_size = parallel_context.moe_tp_size
        self.ep_rank = parallel_context.ep_rank
        self.ep_size = parallel_context.ep_size
        self.num_experts = int(config.num_experts)
        self.hidden_size = int(config.hidden_size)
        self.global_intermediate_size = int(config.moe_intermediate_size)
        if self.global_intermediate_size % self.tp_size:
            raise ValueError(
                "Qwen3MoE moe_intermediate_size must be divisible by "
                f"tensor_parallel_size, got {self.global_intermediate_size} and "
                f"{self.tp_size}."
            )
        self.logical_intermediate_size = self.global_intermediate_size // self.tp_size
        self.fp8_enabled = bool(
            getattr(getattr(config, "quantization_config", None), "enabled", False)
        )
        if self.fp8_enabled and (
            self.hidden_size % 128 or self.global_intermediate_size % 128
        ):
            raise ValueError(
                "Qwen3MoE FP8 requires hidden_size and moe_intermediate_size "
                f"aligned to 128, got {self.hidden_size}/{self.global_intermediate_size}."
            )
        self.fp8_tp_shard = (
            Fp8ExpertTpShard(
                self.global_intermediate_size,
                self.tp_rank,
                self.tp_size,
            )
            if self.fp8_enabled
            else None
        )
        self.intermediate_size = (
            self.fp8_tp_shard.physical_size
            if self.fp8_tp_shard is not None
            else self.logical_intermediate_size
        )
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        activation_dtype = model_activation_dtype(config)
        self.op_spec = MoeOpSpec(
            num_experts=self.num_experts,
            num_local_experts=self.num_local_experts,
            hidden_size=self.hidden_size,
            intermediate_size=self.intermediate_size,
            top_k=int(config.num_experts_per_tok),
            activation_dtype=activation_dtype,
            weight_dtype=(
                torch.float8_e4m3fn if self.fp8_enabled else activation_dtype
            ),
            block_shape=(128, 128) if self.fp8_enabled else None,
            ep_size=int(self.ep_size),
            cuda_graph=bool(getattr(config, "decode_cuda_graph", False)),
            tp_size=int(self.tp_size),
        )
        self.provider = resolve_moe_provider(self.op_spec)
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * self.intermediate_size,
                self.hidden_size,
                dtype=torch.float8_e4m3fn if self.fp8_enabled else None,
            ),
            requires_grad=not self.fp8_enabled,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size,
                dtype=torch.float8_e4m3fn if self.fp8_enabled else None,
            ),
            requires_grad=not self.fp8_enabled,
        )
        if self.fp8_enabled:
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
        else:
            self.register_buffer("w13_scale_inv", None)
            self.register_buffer("w2_scale_inv", None)
        self._loaded_expert_shards: set[tuple[int, str]] = set()

    def is_local_expert(self, global_expert_id: int) -> bool:
        return self.local_expert_start <= int(global_expert_id) < self.local_expert_end

    def load_expert_weight(
        self,
        global_expert_id: int,
        projection: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None = None,
    ) -> None:
        global_expert_id = int(global_expert_id)
        if not self.is_local_expert(global_expert_id):
            raise ValueError(
                f"Expert {global_expert_id} is outside local range "
                f"[{self.local_expert_start}, {self.local_expert_end})."
            )
        if projection not in {"gate_proj", "up_proj", "down_proj"}:
            raise ValueError(f"Unsupported expert projection {projection!r}.")
        load_key = (global_expert_id, projection)
        if load_key in self._loaded_expert_shards:
            raise ValueError(
                f"Duplicate Qwen3MoE expert weight for expert={global_expert_id}, "
                f"projection={projection}."
            )

        if self.fp8_enabled:
            if loaded_scale is None:
                raise ValueError(
                    "Missing FP8 weight_scale_inv for Qwen3MoE "
                    f"expert={global_expert_id}, projection={projection}."
                )
            if loaded_weight.dtype != torch.float8_e4m3fn:
                raise TypeError(
                    "Qwen3MoE expert weight must be FP8 E4M3, "
                    f"got {loaded_weight.dtype}."
                )
            if loaded_scale.dtype != torch.bfloat16:
                raise TypeError(
                    "Qwen3MoE expert weight_scale_inv must be BF16, "
                    f"got {loaded_scale.dtype}."
                )
        elif loaded_scale is not None:
            raise ValueError(
                "Unexpected weight_scale_inv for unquantized Qwen3MoE "
                f"expert={global_expert_id}, projection={projection}."
            )

        if self.fp8_tp_shard is not None:
            loaded_weight, loaded_scale = self.fp8_tp_shard.prepare_projection(
                loaded_weight,
                loaded_scale,
                hidden_size=self.hidden_size,
                down_projection=projection == "down_proj",
            )
        else:
            loaded_weight = self._local_projection_shard(projection, loaded_weight)
        local_expert_id = global_expert_id - self.local_expert_start
        logical_projection = {
            "gate_proj": "gate",
            "up_proj": "up",
            "down_proj": "down",
        }[projection]
        self.provider.load_expert_projection(
            self.op_spec,
            local_expert_id=local_expert_id,
            projection=logical_projection,
            loaded_weight=loaded_weight,
            loaded_scale=loaded_scale,
            w13_weight=self.w13_weight.data,
            w2_weight=self.w2_weight.data,
            w13_scale_inv=self.w13_scale_inv,
            w2_scale_inv=self.w2_scale_inv,
        )
        self._loaded_expert_shards.add(load_key)

    def _local_projection_shard(
        self,
        projection: str,
        loaded_weight: torch.Tensor,
    ) -> torch.Tensor:
        local_shape = (
            (self.hidden_size, self.intermediate_size)
            if projection == "down_proj"
            else (self.intermediate_size, self.hidden_size)
        )
        if tuple(loaded_weight.shape) == local_shape:
            return loaded_weight

        global_shape = (
            (self.hidden_size, self.global_intermediate_size)
            if projection == "down_proj"
            else (self.global_intermediate_size, self.hidden_size)
        )
        if tuple(loaded_weight.shape) != global_shape:
            raise ValueError(
                "Qwen3MoE expert projection shape mismatch: "
                f"projection={projection}, expected local={local_shape} or "
                f"global={global_shape}, got={tuple(loaded_weight.shape)}."
            )
        shard_dim = 1 if projection == "down_proj" else 0
        return loaded_weight.chunk(self.tp_size, dim=shard_dim)[self.tp_rank]

    def validate_loaded_weights(self) -> None:
        expected = {
            (global_expert_id, projection)
            for global_expert_id in range(
                self.local_expert_start, self.local_expert_end
            )
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        missing = sorted(expected - self._loaded_expert_shards)
        if missing:
            raise ValueError(
                "Missing local Qwen3MoE expert weights: "
                f"local_range=[{self.local_expert_start}, {self.local_expert_end}), "
                f"missing={missing[:8]}."
            )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        return self.provider.run(
            self.op_spec,
            hidden_states,
            topk_ids,
            topk_weights,
            self.w13_weight,
            self.w2_weight,
            self.w13_scale_inv,
            self.w2_scale_inv,
            local_expert_start=self.local_expert_start,
            ep_rank=int(self.ep_rank),
        )


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.gate = Qwen3MoeRouter(config)
        self.experts = Qwen3MoePackedExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(
                f"Qwen3MoeSparseMoeBlock expects [tokens, hidden], got {tuple(hidden_states.shape)}."
            )
        debug_enabled = os.getenv("SPARSEVLLM_DEBUG_MOE", "0") == "1"
        if debug_enabled:
            self.debug_last_input = hidden_states.detach().clone()
        router_logits, topk_weights, topk_ids = self.gate(hidden_states)
        local_output = self.experts(
            hidden_states,
            topk_ids,
            topk_weights,
        )

        if debug_enabled:
            self.debug_last_router_logits = router_logits.detach().clone()
            self.debug_last_topk_ids = topk_ids.detach().clone()
            self.debug_last_topk_weights = topk_weights.detach().clone()
            self.debug_last_local_output = local_output.detach().clone()
            local_mask = (topk_ids >= self.experts.local_expert_start) & (
                topk_ids < self.experts.local_expert_end
            )
            local_hit_count = local_mask.sum()
            self.debug_last_local_hit_count = (
                local_hit_count
                if torch.cuda.is_available()
                and torch.cuda.is_current_stream_capturing()
                else int(local_hit_count.item())
            )

        output = self.parallel_context.world_all_reduce(local_output)
        if debug_enabled:
            self.debug_last_output = output.detach().clone()
        return output


class Qwen3MoeDecoderLayer(Qwen3DecoderLayerBase):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__(config)
        self.parallel_context = get_parallel_context()
        self.mlp = Qwen3MoeSparseMoeBlock(config)

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

        if self.parallel_context.tp_size == 1 and self.parallel_context.ep_size > 1:
            # The incoming residual is already replicated, so syncing attention
            # output before RMSNorm preserves the old post-norm state with half
            # the broadcast payload.
            self.parallel_context.ep_broadcast(hidden_states, src_ep_rank=0)

        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(Qwen3ModelBase):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__(config, Qwen3MoeDecoderLayer)


class Qwen3MoeForCausalLM(nn.Module):
    special_weight_loaders = (".expert_weight",)
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
    }

    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        self.config = config
        self.parallel_context = get_parallel_context()
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self._intentionally_skipped_expert_weights: set[str] = set()
        self._intentionally_skipped_expert_scales: set[str] = set()

    @torch.inference_mode()
    def warmup_moe(self) -> None:
        block = self.model.layers[0].mlp
        experts = block.experts
        top_k = int(self.config.num_experts_per_tok)
        device = experts.w13_weight.device
        dtype = block.gate.weight.dtype
        hidden_states = torch.zeros(
            (1, experts.hidden_size),
            dtype=dtype,
            device=device,
        )
        if experts.fp8_enabled:
            layer = self.model.layers[0]
            layer.self_attn.qkv_proj(hidden_states)
            layer.self_attn.o_proj(
                torch.zeros(
                    (1, layer.self_attn.q_size),
                    dtype=dtype,
                    device=device,
                )
            )
            block.gate(hidden_states)
        topk_ids = (
            torch.arange(top_k, dtype=torch.int64, device=device)
            .remainder(experts.num_local_experts)
            .add(experts.local_expert_start)
            .view(1, top_k)
        )
        topk_weights = torch.full(
            (1, top_k),
            1.0 / top_k,
            dtype=dtype,
            device=device,
        )
        experts(hidden_states, topk_ids, topk_weights)
        device_runtime.synchronize()

    def map_weight_name(self, source_weight_name: str) -> str | None:
        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is None:
            return source_weight_name
        layer_idx, global_expert_id, projection = match.groups()
        global_expert_id = int(global_expert_id)
        experts = self.model.layers[int(layer_idx)].mlp.experts
        if not experts.is_local_expert(global_expert_id):
            self._intentionally_skipped_expert_weights.add(source_weight_name)
            return None
        return (
            f"model.layers.{layer_idx}.mlp.experts.{global_expert_id}."
            f"{projection}.expert_weight"
        )

    def rank_local_special_weight_slice(
        self,
        source_weight_name: str,
        source_shape: tuple[int, ...],
        *,
        is_scale: bool = False,
    ) -> tuple[slice, ...] | None:
        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is None:
            return None
        _, _, projection = match.groups()
        experts = self.model.layers[0].mlp.experts
        if experts.tp_size == 1:
            return None
        if experts.fp8_tp_shard is not None:
            return experts.fp8_tp_shard.checkpoint_slice(
                source_shape,
                hidden_size=experts.hidden_size,
                down_projection=projection == "down_proj",
                is_scale=is_scale,
            )
        if is_scale:
            raise ValueError("Unquantized Qwen3MoE checkpoint has an expert scale.")
        global_shape = (
            (experts.hidden_size, experts.global_intermediate_size)
            if projection == "down_proj"
            else (experts.global_intermediate_size, experts.hidden_size)
        )
        if tuple(source_shape) != global_shape:
            raise ValueError(
                "Qwen3MoE checkpoint expert shape mismatch: "
                f"projection={projection}, expected={global_shape}, got={source_shape}."
            )
        start = experts.tp_rank * experts.logical_intermediate_size
        stop = start + experts.logical_intermediate_size
        return (
            (slice(None), slice(start, stop))
            if projection == "down_proj"
            else (slice(start, stop), slice(None))
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
                f"Qwen3MoE loader unexpectedly skipped {source_weight_name!r}."
            )
        layer_idx, global_expert_id, projection = match.groups()
        experts = self.model.layers[int(layer_idx)].mlp.experts
        if experts.is_local_expert(int(global_expert_id)):
            raise ValueError(
                f"Qwen3MoE loader skipped local expert weight {source_weight_name!r}."
            )
        expected_weight_shape = (
            (experts.hidden_size, experts.global_intermediate_size)
            if projection == "down_proj"
            else (experts.global_intermediate_size, experts.hidden_size)
        )
        if loaded_weight_shape != expected_weight_shape:
            raise ValueError(
                "Remote Qwen3MoE expert weight shape mismatch for "
                f"{source_weight_name!r}: expected={expected_weight_shape}, "
                f"got={loaded_weight_shape}."
            )
        if experts.fp8_enabled:
            if loaded_weight_dtype != "F8_E4M3":
                raise TypeError(
                    "Remote Qwen3MoE expert weight must be FP8 E4M3, got "
                    f"safetensors dtype {loaded_weight_dtype}."
                )
            expected_scale_shape = (
                (experts.hidden_size // 128, experts.global_intermediate_size // 128)
                if projection == "down_proj"
                else (
                    experts.global_intermediate_size // 128,
                    experts.hidden_size // 128,
                )
            )
            if loaded_scale_shape != expected_scale_shape:
                raise ValueError(
                    "Remote Qwen3MoE expert scale shape mismatch for "
                    f"{source_weight_name!r}: expected={expected_scale_shape}, "
                    f"got={loaded_scale_shape}."
                )
            if loaded_scale_dtype != "BF16":
                raise TypeError(
                    "Remote Qwen3MoE expert scale must be BF16, got "
                    f"safetensors dtype {loaded_scale_dtype}."
                )
            self._intentionally_skipped_expert_scales.add(
                source_weight_name[: -len(".weight")] + ".weight_scale_inv"
            )
        elif loaded_scale_shape is not None or loaded_scale_dtype is not None:
            raise ValueError(
                "Unquantized remote Qwen3MoE expert unexpectedly has "
                f"weight_scale_inv: {source_weight_name!r}."
            )
        elif loaded_weight_dtype not in {"BF16", "F16", "F32"}:
            raise TypeError(
                "Remote unquantized Qwen3MoE expert has unsupported safetensors "
                f"dtype {loaded_weight_dtype}."
            )
        self._intentionally_skipped_expert_weights.add(source_weight_name)

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
        self.model.layers[int(layer_idx)].mlp.experts.load_expert_weight(
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
            if name.endswith(".mlp.experts.w13_weight")
            or name.endswith(".mlp.experts.w2_weight")
        }
        expected_dense_parameters = {
            name for name, _ in self.named_parameters()
        } - packed_expert_parameters
        missing_dense = sorted(expected_dense_parameters - loaded_parameter_names)
        if missing_dense:
            raise ValueError(
                f"Missing replicated Qwen3MoE weights: {missing_dense[:8]}."
            )

        for layer in self.model.layers:
            layer.mlp.experts.validate_loaded_weights()

        expected_skipped = {
            f"model.layers.{layer_idx}.mlp.experts.{expert_id}.{projection}.weight"
            for layer_idx in range(int(self.config.num_hidden_layers))
            for expert_id in range(int(self.config.num_experts))
            if not self.model.layers[layer_idx].mlp.experts.is_local_expert(expert_id)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        expected_skipped_scales = (
            {name[: -len(".weight")] + ".weight_scale_inv" for name in expected_skipped}
            if self.model.layers[0].mlp.experts.fp8_enabled
            else set()
        )
        missing_skips = sorted(
            expected_skipped - self._intentionally_skipped_expert_weights
        )
        if missing_skips:
            raise ValueError(
                "Checkpoint is missing expected remote expert entries: "
                f"{missing_skips[:8]}."
            )
        unexpected_skips = sorted(
            self._intentionally_skipped_expert_weights - expected_skipped
        )
        missing_scale_skips = sorted(
            expected_skipped_scales - self._intentionally_skipped_expert_scales
        )
        unexpected_scale_skips = sorted(
            self._intentionally_skipped_expert_scales - expected_skipped_scales
        )
        if missing_scale_skips:
            raise ValueError(
                "Checkpoint is missing expected remote expert scales: "
                f"{missing_scale_skips[:8]}."
            )
        if unexpected_skips or unexpected_scale_skips:
            raise ValueError(
                "Unexpectedly skipped Qwen3MoE expert entries: "
                f"weights={unexpected_skips[:4]}, "
                f"scales={unexpected_scale_skips[:4]}."
            )
        logger.info(
            "Loaded Qwen3MoE rank {} provider={} attention TP {}/{} MoE TP {}/{} "
            "and local experts "
            "[{}, {}) across {} layers; intentionally skipped {} remote expert tensors.",
            self.parallel_context.world_rank,
            self.model.layers[0].mlp.experts.provider.name,
            self.parallel_context.tp_rank,
            self.parallel_context.tp_size,
            self.parallel_context.moe_tp_rank,
            self.parallel_context.moe_tp_size,
            self.model.layers[0].mlp.experts.local_expert_start,
            self.model.layers[0].mlp.experts.local_expert_end,
            len(self.model.layers),
            len(self._intentionally_skipped_expert_weights),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)
