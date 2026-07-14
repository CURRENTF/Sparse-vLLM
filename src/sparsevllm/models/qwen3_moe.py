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
from sparsevllm.utils.log import logger
from sparsevllm.utils.profiler import profiler


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
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_size))

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dtype=torch.float32, dim=-1)
        topk_weights, topk_ids = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        return router_logits, topk_weights.to(router_logits.dtype), topk_ids


class Qwen3MoePackedExperts(nn.Module):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__()
        parallel_context = get_parallel_context()
        self.ep_rank = parallel_context.ep_rank
        self.ep_size = parallel_context.ep_size
        self.num_experts = int(config.num_experts)
        self.hidden_size = int(config.hidden_size)
        self.intermediate_size = int(config.moe_intermediate_size)
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.w13_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                2 * self.intermediate_size,
                self.hidden_size,
            )
        )
        self.w2_weight = nn.Parameter(
            torch.empty(
                self.num_local_experts,
                self.hidden_size,
                self.intermediate_size,
            )
        )
        self._loaded_expert_shards: set[tuple[int, str]] = set()

    def is_local_expert(self, global_expert_id: int) -> bool:
        return self.local_expert_start <= int(global_expert_id) < self.local_expert_end

    def load_expert_weight(
        self,
        global_expert_id: int,
        projection: str,
        loaded_weight: torch.Tensor,
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

        local_expert_id = global_expert_id - self.local_expert_start
        if projection == "down_proj":
            target = self.w2_weight.data[local_expert_id]
        else:
            offset = 0 if projection == "gate_proj" else self.intermediate_size
            target = self.w13_weight.data[local_expert_id, offset : offset + self.intermediate_size]
        if tuple(target.shape) != tuple(loaded_weight.shape):
            raise ValueError(
                f"Qwen3MoE expert weight shape mismatch for expert={global_expert_id}, "
                f"projection={projection}: expected={tuple(target.shape)}, "
                f"got={tuple(loaded_weight.shape)}."
            )
        target.copy_(loaded_weight)
        self._loaded_expert_shards.add(load_key)

    def validate_loaded_weights(self) -> None:
        expected = {
            (global_expert_id, projection)
            for global_expert_id in range(self.local_expert_start, self.local_expert_end)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        missing = sorted(expected - self._loaded_expert_shards)
        if missing:
            raise ValueError(
                "Missing local Qwen3MoE expert weights: "
                f"local_range=[{self.local_expert_start}, {self.local_expert_end}), "
                f"missing={missing[:8]}."
            )

    def forward_pytorch(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        local_output = torch.zeros_like(hidden_states)
        for local_expert_id in range(self.num_local_experts):
            global_expert_id = self.local_expert_start + local_expert_id
            token_ids, topk_slots = torch.where(topk_ids == global_expert_id)
            if token_ids.numel() == 0:
                continue
            expert_input = hidden_states[token_ids]
            gate_up = F.linear(expert_input, self.w13_weight[local_expert_id])
            gate, up = gate_up.chunk(2, dim=-1)
            expert_output = F.linear(
                F.silu(gate) * up,
                self.w2_weight[local_expert_id],
            )
            expert_output = expert_output * topk_weights[token_ids, topk_slots, None]
            local_output.index_add_(0, token_ids, expert_output.to(local_output.dtype))
        return local_output


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
        with profiler.record("moe_router"):
            router_logits, topk_weights, topk_ids = self.gate(hidden_states)
        with profiler.record("moe_experts_pytorch"):
            local_output = self.experts.forward_pytorch(
                hidden_states,
                topk_ids,
                topk_weights,
            )

        debug_enabled = os.getenv("SPARSEVLLM_DEBUG_MOE", "0") == "1"
        if debug_enabled:
            self.debug_last_router_logits = router_logits.detach().clone()
            self.debug_last_topk_ids = topk_ids.detach().clone()
            self.debug_last_topk_weights = topk_weights.detach().clone()
            self.debug_last_local_output = local_output.detach().clone()
            local_mask = (topk_ids >= self.experts.local_expert_start) & (
                topk_ids < self.experts.local_expert_end
            )
            self.debug_last_local_hit_count = int(local_mask.sum().item())

        with profiler.record("moe_ep_all_reduce"):
            output = self.parallel_context.ep_all_reduce(local_output)
        if debug_enabled:
            self.debug_last_output = output.detach().clone()
        return output


class Qwen3MoeDecoderLayer(Qwen3DecoderLayerBase):
    def __init__(self, config: Qwen3MoeConfig) -> None:
        super().__init__(config)
        self.mlp = Qwen3MoeSparseMoeBlock(config)


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

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        match = _EXPERT_TARGET_RE.match(target_weight_name)
        if match is None:
            return 0
        if loaded_scale is not None:
            raise NotImplementedError("Qwen3MoE v1 does not support quantized expert weights.")
        layer_idx, global_expert_id, projection = match.groups()
        self.model.layers[int(layer_idx)].mlp.experts.load_expert_weight(
            int(global_expert_id),
            projection,
            loaded_weight,
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
        if unexpected_skips:
            raise ValueError(
                f"Unexpectedly skipped Qwen3MoE expert weights: {unexpected_skips[:8]}."
            )
        logger.info(
            "Loaded Qwen3MoE rank {} local experts [{}, {}) across {} layers; "
            "intentionally skipped {} remote expert tensors.",
            self.parallel_context.world_rank,
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
