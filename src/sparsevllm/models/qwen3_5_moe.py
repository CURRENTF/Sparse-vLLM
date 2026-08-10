from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F
from torch import nn

from sparsevllm.distributed import get_parallel_context
from sparsevllm.engine.recurrent_state_manager import (
    RecurrentStateSpec,
    RecurrentTensorSpec,
)
from sparsevllm.layers.embed_head import ParallelLMHead
from sparsevllm.models.qwen3_5 import (
    Qwen35DecoderLayer,
    Qwen35ForCausalLM,
    Qwen35MLP,
    Qwen35Model,
)
from sparsevllm.models.qwen3_moe import Qwen3MoePackedExperts
from sparsevllm.operators.moe import model_activation_dtype
from sparsevllm.operators.moe_router import (
    MoeRouterOpSpec,
    resolve_moe_router_provider,
)
from sparsevllm.platforms import device_runtime
from sparsevllm.utils.log import logger
from sparsevllm.utils.weight_target import WeightTarget


_PACKED_EXPERT_SOURCE_RE = re.compile(
    r"^model\.language_model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)$"
)
_PACKED_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\."
    r"(gate_up_proj|down_proj)\.packed_expert_weight$"
)


class Qwen35MoeRouter(nn.Module):
    """Replicated router with the checkpoint's FP32 softmax semantics."""

    def __init__(self, config) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.num_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.op_spec = MoeRouterOpSpec(
            num_experts=self.num_experts,
            top_k=self.top_k,
            activation_dtype=model_activation_dtype(config),
            norm_topk_prob=True,
            cuda_graph=bool(getattr(config, "decode_cuda_graph", False)),
        )
        self.provider = resolve_moe_router_provider(self.op_spec)
        self.weight = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_size)
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states, self.weight)
        topk_weights, topk_ids = self.provider.run(
            self.op_spec, router_logits
        )
        return router_logits, topk_weights, topk_ids


class Qwen35MoePackedExperts(Qwen3MoePackedExperts):
    """Qwen3.6 packed-3D checkpoint adapter for the shared MoE provider."""

    checkpoint_projection_map = {
        "gate_up_proj": "gate_up",
        "down_proj": "down",
    }

    def __init__(self, config) -> None:
        super().__init__(config)
        if self.fp8_enabled:
            raise NotImplementedError(
                "Qwen3.6 MoE v1 accepts BF16 expert weights only."
            )
        self._loaded_packed_projections: set[str] = set()

    def rank_local_weight_slice(
        self,
        source_shape: tuple[int, ...],
        *,
        loaded_shard_id: str,
        is_scale: bool = False,
    ) -> tuple[slice, ...] | None:
        if is_scale:
            raise ValueError("Qwen3.6 MoE BF16 experts do not have weight scales.")
        expected = {
            "gate_up_proj": (
                self.num_experts,
                2 * self.global_intermediate_size,
                self.hidden_size,
            ),
            "down_proj": (
                self.num_experts,
                self.hidden_size,
                self.global_intermediate_size,
            ),
        }.get(str(loaded_shard_id))
        if expected is None:
            raise ValueError(
                f"Unsupported Qwen3.6 packed expert projection {loaded_shard_id!r}."
            )
        if tuple(source_shape) != expected:
            raise ValueError(
                "Qwen3.6 packed expert checkpoint shape mismatch: "
                f"projection={loaded_shard_id} expected={expected} "
                f"got={tuple(source_shape)}."
            )
        return (
            slice(self.local_expert_start, self.local_expert_end),
            slice(None),
            slice(None),
        )

    def load_packed_expert_weight(
        self,
        projection: str,
        loaded_weight: torch.Tensor,
    ) -> None:
        projection = str(projection)
        if projection in self._loaded_packed_projections:
            raise ValueError(
                f"Duplicate Qwen3.6 packed expert projection {projection!r}."
            )
        if loaded_weight.dtype != torch.bfloat16:
            raise TypeError(
                "Qwen3.6 packed expert weights must be BF16, "
                f"got {loaded_weight.dtype}."
            )
        expected_local = {
            "gate_up_proj": (
                self.num_local_experts,
                2 * self.global_intermediate_size,
                self.hidden_size,
            ),
            "down_proj": (
                self.num_local_experts,
                self.hidden_size,
                self.global_intermediate_size,
            ),
        }.get(projection)
        if expected_local is None:
            raise ValueError(
                f"Unsupported Qwen3.6 packed expert projection {projection!r}."
            )
        if tuple(loaded_weight.shape) != expected_local:
            raise ValueError(
                "Qwen3.6 rank-local packed expert shape mismatch: "
                f"projection={projection} expected={expected_local} "
                f"got={tuple(loaded_weight.shape)}."
            )

        for local_expert_id in range(self.num_local_experts):
            global_expert_id = self.local_expert_start + local_expert_id
            if projection == "gate_up_proj":
                gate, up = loaded_weight[local_expert_id].split(
                    self.global_intermediate_size,
                    dim=0,
                )
                self.load_expert_weight(
                    global_expert_id, "gate_proj", gate, None
                )
                self.load_expert_weight(
                    global_expert_id, "up_proj", up, None
                )
            else:
                self.load_expert_weight(
                    global_expert_id,
                    "down_proj",
                    loaded_weight[local_expert_id],
                    None,
                )
        self._loaded_packed_projections.add(projection)

    def validate_loaded_weights(self) -> None:
        missing = {"gate_up_proj", "down_proj"} - self._loaded_packed_projections
        if missing:
            raise ValueError(
                f"Missing Qwen3.6 packed expert projections: {sorted(missing)}."
            )
        super().validate_loaded_weights()


class Qwen35MoeSparseMoeBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.debug_enabled = os.getenv("SPARSEVLLM_DEBUG_MOE", "0") == "1"
        self.mlp_chunk_size = int(getattr(config, "mlp_chunk_size", 16384))
        if self.mlp_chunk_size <= 0:
            raise ValueError(
                f"mlp_chunk_size must be > 0, got {self.mlp_chunk_size}."
            )
        self.gate = Qwen35MoeRouter(config)
        self.experts = Qwen35MoePackedExperts(config)
        self.shared_expert = Qwen35MLP(
            config,
            intermediate_size=int(config.shared_expert_intermediate_size),
        )
        self.shared_expert_gate = nn.Linear(
            int(config.hidden_size), 1, bias=False
        )

    def _forward_chunk(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        shared_output = self.shared_expert(hidden_states)
        router_logits, topk_weights, topk_ids = self.gate(hidden_states)
        local_output = self.experts(hidden_states, topk_ids, topk_weights)
        routed_output = self.parallel_context.world_all_reduce(local_output)
        shared_gate = torch.sigmoid(self.shared_expert_gate(hidden_states))
        gated_shared_output = shared_gate * shared_output
        return (
            routed_output + gated_shared_output,
            router_logits,
            topk_weights,
            topk_ids,
            local_output,
            gated_shared_output,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.dim() != 2:
            raise ValueError(
                "Qwen35MoeSparseMoeBlock expects [tokens, hidden], "
                f"got {tuple(hidden_states.shape)}."
            )
        debug_enabled = self.debug_enabled
        if debug_enabled:
            self.debug_last_input = hidden_states.detach().clone()

        chunks = hidden_states.split(self.mlp_chunk_size, dim=0)
        outputs = []
        router_logits_chunks = []
        topk_weights_chunks = []
        topk_ids_chunks = []
        local_output_chunks = []
        shared_output_chunks = []
        for chunk in chunks:
            (
                output,
                router_logits,
                topk_weights,
                topk_ids,
                local_output,
                shared_output,
            ) = self._forward_chunk(chunk)
            outputs.append(output)
            if debug_enabled:
                router_logits_chunks.append(router_logits)
                topk_weights_chunks.append(topk_weights)
                topk_ids_chunks.append(topk_ids)
                local_output_chunks.append(local_output)
                shared_output_chunks.append(shared_output)
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)

        if debug_enabled:
            self.debug_last_router_logits = torch.cat(
                router_logits_chunks, dim=0
            ).detach().clone()
            self.debug_last_topk_weights = torch.cat(
                topk_weights_chunks, dim=0
            ).detach().clone()
            self.debug_last_topk_ids = torch.cat(
                topk_ids_chunks, dim=0
            ).detach().clone()
            self.debug_last_local_output = torch.cat(
                local_output_chunks, dim=0
            ).detach().clone()
            self.debug_last_shared_output = torch.cat(
                shared_output_chunks, dim=0
            ).detach().clone()
            local_mask = (
                self.debug_last_topk_ids >= self.experts.local_expert_start
            ) & (self.debug_last_topk_ids < self.experts.local_expert_end)
            local_hit_count = local_mask.sum()
            self.debug_last_local_hit_count = (
                local_hit_count
                if device_runtime.is_stream_capturing()
                else int(local_hit_count.item())
            )
            self.debug_last_output = output.detach().clone()
        return output


class Qwen35MoeDecoderLayer(Qwen35DecoderLayer):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__(config, layer_idx, mlp_cls=Qwen35MoeSparseMoeBlock)


class Qwen35MoeModel(Qwen35Model):
    def __init__(self, config) -> None:
        setattr(config, "runtime_recurrent_state_dtype", torch.float32)
        super().__init__(config, Qwen35MoeDecoderLayer)


class Qwen35MoeForCausalLM(Qwen35ForCausalLM):
    ignored_weight_prefixes = ("model.visual.", "visual.", "mtp.")
    special_weight_loaders = {
        **Qwen35ForCausalLM.special_weight_loaders,
        ".packed_expert_weight": "load_packed_expert_weight",
    }

    def __init__(self, config) -> None:
        nn.Module.__init__(self)
        self.config = config
        self.parallel_context = get_parallel_context()
        self.model = Qwen35MoeModel(config)
        self.lm_head = ParallelLMHead(
            int(config.vocab_size), int(config.hidden_size)
        )
        if bool(getattr(config, "tie_word_embeddings", False)):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self._loaded_linear_special_weights: set[str] = set()
        self._intentionally_skipped_weights: set[str] = set()

    @staticmethod
    def recurrent_state_spec(config, attention_tp_size: int) -> RecurrentStateSpec:
        attention_tp_size = int(attention_tp_size)
        num_k_heads = int(config.linear_num_key_heads) // attention_tp_size
        num_v_heads = int(config.linear_num_value_heads) // attention_tp_size
        key_head_dim = int(config.linear_key_head_dim)
        value_head_dim = int(config.linear_value_head_dim)
        conv_dim = 2 * num_k_heads * key_head_dim + num_v_heads * value_head_dim
        return RecurrentStateSpec(
            name="qwen3_5_moe gated delta net",
            tensor_specs=(
                RecurrentTensorSpec(
                    "conv_state",
                    (conv_dim, int(config.linear_conv_kernel_dim) - 1),
                    config.torch_dtype,
                ),
                RecurrentTensorSpec(
                    "recurrent_state",
                    (num_v_heads, key_head_dim, value_head_dim),
                    torch.float32,
                ),
            ),
        )

    def runtime_diagnostic_status(self) -> dict[str, object]:
        experts = self.model.layers[0].mlp.experts
        router = self.model.layers[0].mlp.gate
        return {
            "moe_expert_provider": experts.provider.name,
            "moe_router_provider": router.provider.name,
            "local_expert_start": int(experts.local_expert_start),
            "local_expert_end": int(experts.local_expert_end),
        }

    @torch.inference_mode()
    def warmup_moe(self, num_tokens: int = 1) -> None:
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be > 0, got {num_tokens}.")
        block = self.model.layers[0].mlp
        experts = block.experts
        device = experts.w13_weight.device
        dtype = block.gate.weight.dtype
        hidden_states = torch.zeros(
            (num_tokens, experts.hidden_size), dtype=dtype, device=device
        )
        top_k = int(self.config.num_experts_per_tok)
        topk_ids = (
            torch.arange(num_tokens * top_k, dtype=torch.int64, device=device)
            .remainder(experts.num_local_experts)
            .add(experts.local_expert_start)
            .view(num_tokens, top_k)
        )
        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            dtype=dtype,
            device=device,
        )
        experts(hidden_states, topk_ids, topk_weights)
        block(hidden_states)
        device_runtime.synchronize()

    def map_weight_name(self, source_weight_name: str) -> str:
        match = _PACKED_EXPERT_SOURCE_RE.match(source_weight_name)
        if match is not None:
            layer_idx, projection = match.groups()
            return (
                f"model.layers.{layer_idx}.mlp.experts.{projection}."
                "packed_expert_weight"
            )
        return super().map_weight_name(source_weight_name)

    def resolve_special_weight(
        self,
        target_weight_name: str,
    ) -> WeightTarget | None:
        match = _PACKED_EXPERT_TARGET_RE.match(target_weight_name)
        if match is not None:
            layer_idx, projection = match.groups()
            return WeightTarget(
                self.model.layers[int(layer_idx)].mlp.experts,
                projection,
            )
        return super().resolve_special_weight(target_weight_name)

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        match = _PACKED_EXPERT_TARGET_RE.match(target_weight_name)
        if match is not None:
            if loaded_scale is not None:
                raise ValueError(
                    "Qwen3.6 BF16 packed experts unexpectedly have weight scales."
                )
            layer_idx, projection = match.groups()
            self.model.layers[int(layer_idx)].mlp.experts.load_packed_expert_weight(
                projection,
                loaded_weight,
            )
            return 1
        loaded = super().load_special_weight(
            target_weight_name,
            loaded_weight,
            loaded_scale,
        )
        if loaded:
            self._loaded_linear_special_weights.add(target_weight_name)
        return loaded

    def record_skipped_weight(
        self,
        source_weight_name: str,
        loaded_weight_shape: tuple[int, ...] | None,
        loaded_weight_dtype: str | None,
        loaded_scale_shape: tuple[int, ...] | None,
        loaded_scale_dtype: str | None,
    ) -> None:
        del loaded_weight_shape, loaded_weight_dtype
        if not source_weight_name.startswith(self.ignored_weight_prefixes):
            raise ValueError(
                f"Qwen3.6 MoE loader unexpectedly skipped {source_weight_name!r}."
            )
        if loaded_scale_shape is not None or loaded_scale_dtype is not None:
            raise ValueError(
                "Qwen3.6 MoE visual/MTP intentional skips must not consume "
                f"quantization scales: {source_weight_name!r}."
            )
        self._intentionally_skipped_weights.add(source_weight_name)

    def validate_loaded_weights(self, loaded_parameter_names: set[str]) -> None:
        packed_parameters = {
            name
            for name, _ in self.named_parameters()
            if name.endswith(".mlp.experts.w13_weight")
            or name.endswith(".mlp.experts.w2_weight")
        }
        linear_special_parameters = {
            name
            for name, _ in self.named_parameters()
            if ".linear_attn.in_proj_" in name
            and name.endswith(".weight")
            and name.rsplit(".", 2)[-2] in {"in_proj_q", "in_proj_k", "in_proj_v"}
        }
        expected_dense = {
            name for name, _ in self.named_parameters()
        } - packed_parameters - linear_special_parameters
        missing_dense = sorted(expected_dense - loaded_parameter_names)
        if missing_dense:
            raise ValueError(
                f"Missing Qwen3.6 MoE replicated/sharded weights: {missing_dense[:8]}."
            )

        expected_linear_special = {
            f"model.layers.{layer_idx}.linear_attn.in_proj_qkv.weight"
            for layer_idx in self.config.runtime_layout.linear_attention_layer_indices
        }
        missing_linear = sorted(
            expected_linear_special - self._loaded_linear_special_weights
        )
        if missing_linear:
            raise ValueError(
                f"Missing Qwen3.6 packed GDN weights: {missing_linear[:8]}."
            )
        for layer in self.model.layers:
            layer.mlp.experts.validate_loaded_weights()

        skip_groups = {
            "visual": any(
                name.startswith(("model.visual.", "visual."))
                for name in self._intentionally_skipped_weights
            ),
            "mtp": any(
                name.startswith("mtp.")
                for name in self._intentionally_skipped_weights
            ),
        }
        missing_skip_groups = [name for name, seen in skip_groups.items() if not seen]
        if missing_skip_groups:
            raise ValueError(
                "Qwen3.6 MoE checkpoint is missing expected intentional-skip "
                f"groups: {missing_skip_groups}."
            )
        first_experts = self.model.layers[0].mlp.experts
        logger.info(
            "Loaded Qwen3.6 MoE rank {} expert_provider={} router_provider={} "
            "attention TP {}/{} "
            "MoE TP {}/{} EP {}/{} local experts [{}, {}) across {} layers; "
            "intentionally skipped {} visual/MTP tensors.",
            self.parallel_context.world_rank,
            first_experts.provider.name,
            self.model.layers[0].mlp.gate.provider.name,
            self.parallel_context.attention_tp_rank,
            self.parallel_context.attention_tp_size,
            self.parallel_context.moe_tp_rank,
            self.parallel_context.moe_tp_size,
            self.parallel_context.ep_rank,
            self.parallel_context.ep_size,
            first_experts.local_expert_start,
            first_experts.local_expert_end,
            len(self.model.layers),
            len(self._intentionally_skipped_weights),
        )
