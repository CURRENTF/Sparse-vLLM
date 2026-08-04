from __future__ import annotations

import os
import re

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Glm4MoeLiteConfig

from sparsevllm.distributed import get_parallel_context
from sparsevllm.engine.cache_manager import MlaLatentWrite
from sparsevllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.layers.linear import (
    ColumnParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sparsevllm.layers.mla_attention import MLAAttention
from sparsevllm.layers.packed_moe import PackedMoeExperts
from sparsevllm.layers.rotary_embedding import RotaryEmbedding, get_rope
from sparsevllm.models.qwen3 import Qwen3MLP
from sparsevllm.operators.mla_attention import MlaAttentionOpSpec
from sparsevllm.operators.moe import model_activation_dtype, resolve_moe_provider
from sparsevllm.platforms import device_runtime
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger
from sparsevllm.utils.weight_target import WeightTarget


_EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_EXPERT_TARGET_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(gate_proj|up_proj|down_proj)\.expert_weight$"
)
_MTP_LAYER_INDEX = 47
_MTP_PREFIX = f"model.layers.{_MTP_LAYER_INDEX}."


def build_glm4_moe_lite_mla_attention(
    config: Glm4MoeLiteConfig,
    *,
    device: torch.device | str,
    max_batch_size: int,
    prefill_workspace_bytes: int,
    decode_cuda_graph: bool,
) -> MLAAttention:
    """Bind the one process-local MLA operator from explicit runtime inputs."""

    parallel_context = get_parallel_context()
    activation_dtype = model_activation_dtype(config)
    spec = MlaAttentionOpSpec(
        num_q_heads=int(config.num_attention_heads),
        kv_lora_rank=int(config.kv_lora_rank),
        rope_dim=int(config.qk_rope_head_dim),
        qk_head_dim=int(config.qk_nope_head_dim + config.qk_rope_head_dim),
        value_head_dim=int(config.v_head_dim),
        activation_dtype=activation_dtype,
        cache_dtype=activation_dtype,
        tp_size=int(parallel_context.attention_tp_size),
        cuda_graph=bool(decode_cuda_graph),
    )
    return MLAAttention.bind(
        spec=spec,
        device=device,
        max_batch_size=max_batch_size,
        prefill_workspace_bytes=prefill_workspace_bytes,
    )


class Glm4MoeLiteAttention(nn.Module):
    """GLM projections around a shared latent-MLA execution object."""

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        mla_attention: MLAAttention,
        *,
        projection_chunk_size: int,
    ) -> None:
        super().__init__()
        self.mla_attention = mla_attention
        self.parallel_context = get_parallel_context()
        self.num_heads = int(config.num_attention_heads)
        self.local_heads = int(mla_attention.spec.local_q_heads)
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        self.v_head_dim = int(config.v_head_dim)
        self.proj_chunk_size = int(projection_chunk_size)
        if self.proj_chunk_size <= 0:
            raise ValueError(
                f"mlp_chunk_size must be positive, got {self.proj_chunk_size}."
            )
        quantization = getattr(config, "quantization_config", None)
        if bool(getattr(quantization, "enabled", False)):
            raise NotImplementedError("GLM MLA projections do not support quantization.")

        self.q_a_proj = ReplicatedLinear(
            int(config.hidden_size),
            self.q_lora_rank,
            bias=bool(config.attention_bias),
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
        )
        self.kv_a_proj_with_mqa = ReplicatedLinear(
            int(config.hidden_size),
            self.kv_lora_rank + self.qk_rope_head_dim,
            bias=bool(config.attention_bias),
        )
        self.kv_a_layernorm = RMSNorm(
            self.kv_lora_rank,
            eps=config.rms_norm_eps,
        )
        self.kv_b_proj = ColumnParallelLinear(
            self.kv_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            int(config.hidden_size),
            bias=bool(config.attention_bias),
        )

    def _project_kv_history(self, latent: torch.Tensor) -> torch.Tensor:
        if int(latent.shape[0]) <= self.proj_chunk_size:
            return self.kv_b_proj(latent)
        output = torch.empty(
            latent.shape[0],
            self.local_heads * (self.qk_nope_head_dim + self.v_head_dim),
            dtype=latent.dtype,
            device=latent.device,
        )
        for start in range(0, int(latent.shape[0]), self.proj_chunk_size):
            end = min(start + self.proj_chunk_size, int(latent.shape[0]))
            output[start:end].copy_(self.kv_b_proj(latent[start:end]))
        return output

    def _project_output(self, value_output: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        flattened = value_output.flatten(1, -1)
        if int(flattened.shape[0]) <= self.proj_chunk_size:
            out.copy_(self.o_proj(flattened))
            return out
        for start in range(0, int(flattened.shape[0]), self.proj_chunk_size):
            end = min(start + self.proj_chunk_size, int(flattened.shape[0]))
            out[start:end].copy_(self.o_proj(flattened[start:end]))
        return out

    def _decode_absorbed_query(self, q_nope: torch.Tensor) -> torch.Tensor:
        kv_b_weight = self.kv_b_proj.weight.view(
            self.local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        k_weight = kv_b_weight[:, : self.qk_nope_head_dim]
        return torch.bmm(
            q_nope.transpose(0, 1),
            k_weight,
        ).transpose(0, 1)

    def _reconstruct_decode_values(self, latent_output: torch.Tensor) -> torch.Tensor:
        kv_b_weight = self.kv_b_proj.weight.view(
            self.local_heads,
            self.qk_nope_head_dim + self.v_head_dim,
            self.kv_lora_rank,
        )
        v_weight = kv_b_weight[:, self.qk_nope_head_dim :]
        return torch.bmm(
            latent_output.transpose(0, 1),
            v_weight.transpose(1, 2),
        ).transpose(0, 1)

    def _run_attention(
        self,
        q: torch.Tensor,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        latent: torch.Tensor,
        rope: torch.Tensor,
    ) -> torch.Tensor:
        context = get_context()
        cache_manager = context.cache_manager
        sparse_controller = context.sparse_controller
        layer_idx = int(context.now_layer_idx)
        slot_mapping = cache_manager.store_attention_payload(
            layer_idx,
            MlaLatentWrite(
                latent=latent.unsqueeze(1),
                rope=rope.unsqueeze(1),
            ),
        )
        cache_manager.on_kv_stored(layer_idx, latent, slot_mapping)

        temp_slots = None
        try:
            if context.is_prefill:
                selection = sparse_controller.get_prefill_selection(layer_idx)
                cache_manager.before_prefill_layer_attention(layer_idx, selection)
                view = cache_manager.build_prefill_compute_view(
                    layer_idx,
                    latent,
                    rope,
                    selection,
                )
                temp_slots = view.meta.temp_slots
                if context.cu_seqlens_q is None or context.cu_seqlens_q.numel() <= 1:
                    return torch.empty_like(q)
                history = self.mla_attention.prepare_prefill_history(view)
                expanded = self._project_kv_history(history.gathered_latent).view(
                    history.visible_tokens,
                    self.local_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                )
                expanded_k_nope, expanded_v = expanded.split(
                    [self.qk_nope_head_dim, self.v_head_dim],
                    dim=-1,
                )
                expanded_rope = history.gathered_rope[:, None, :].expand(
                    -1,
                    self.local_heads,
                    -1,
                )
                workset = self.mla_attention.bind_prefill_kv(
                    history,
                    expanded_k=torch.cat((expanded_k_nope, expanded_rope), dim=-1),
                    expanded_v=expanded_v,
                )
                b_start_loc = context.cu_seqlens_q[:-1]
                chunk_lens = context.cu_seqlens_q[1:] - context.cu_seqlens_q[:-1]
                output = self.mla_attention.run_prefill(
                    q,
                    workset,
                    b_start_loc=b_start_loc,
                    chunk_lens=chunk_lens,
                )
                cache_manager.collect_prefill_attention_score(
                    layer_idx,
                    q,
                    view,
                    b_start_loc=b_start_loc,
                    chunk_lens=chunk_lens,
                )
                cache_manager.record_prefill_query(
                    layer_idx,
                    q,
                    view,
                    b_start_loc=b_start_loc,
                    chunk_lens=chunk_lens,
                )
            else:
                selection = sparse_controller.get_decode_selection(layer_idx, q)
                view = cache_manager.build_decode_compute_view(
                    layer_idx,
                    q,
                    selection,
                    num_heads=self.local_heads,
                    num_kv_heads=1,
                )
                latent_output = self.mla_attention.run_decode(
                    self._decode_absorbed_query(q_nope),
                    q_rope,
                    view,
                )
                output = self._reconstruct_decode_values(latent_output)
                cache_manager.record_decode_query(layer_idx, q)

            sparse_controller.on_layer_attention_end(layer_idx)
            cache_manager.on_layer_attention_end(layer_idx)
            return output
        finally:
            if temp_slots is not None and temp_slots.numel() > 0:
                cache_manager.release_layer_temp_slots(layer_idx, temp_slots)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        rotary_emb: RotaryEmbedding,
    ) -> torch.Tensor:
        q = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        q = q.view(-1, self.local_heads, self.qk_head_dim)
        q_nope, q_rope = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        latent, k_rope = compressed_kv.split(
            [self.kv_lora_rank, self.qk_rope_head_dim],
            dim=-1,
        )
        latent = self.kv_a_layernorm(latent)
        q_rope, k_rope = rotary_emb(
            positions,
            q_rope,
            k_rope.unsqueeze(1),
        )
        k_rope = k_rope.squeeze(1)
        q = torch.cat((q_nope, q_rope), dim=-1)
        value_output = self._run_attention(
            q,
            q_nope,
            q_rope,
            latent,
            k_rope,
        )
        return self._project_output(value_output, hidden_states)


class Glm4MoeLiteRouter(nn.Module):
    def __init__(self, config: Glm4MoeLiteConfig) -> None:
        super().__init__()
        self.hidden_size = int(config.hidden_size)
        self.num_experts = int(config.n_routed_experts)
        self.top_k = int(config.num_experts_per_tok)
        self.routed_scaling_factor = float(config.routed_scaling_factor)
        self.weight = nn.Parameter(
            torch.empty(
                self.num_experts,
                self.hidden_size,
                dtype=torch.float32,
            )
        )
        self.e_score_correction_bias = nn.Parameter(
            torch.empty(self.num_experts, dtype=torch.float32)
        )
        from sparsevllm.triton_kernel.moe_biased_sigmoid import (
            topk_biased_sigmoid,
        )

        self.topk_impl = topk_biased_sigmoid

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
        topk_weights = topk_weights * self.routed_scaling_factor
        return router_logits, topk_weights, topk_ids


class Glm4MoeLitePackedExperts(PackedMoeExperts):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        *,
        decode_cuda_graph: bool,
    ) -> None:
        super().__init__(
            num_experts=int(config.n_routed_experts),
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            top_k=int(config.num_experts_per_tok),
            activation_dtype=model_activation_dtype(config),
            fp8_enabled=False,
            cuda_graph=bool(decode_cuda_graph),
            routing_method="biased_sigmoid",
            model_label="GLM-4.7-Flash",
            provider_resolver=resolve_moe_provider,
            parallel_context=get_parallel_context(),
        )


class Glm4MoeLiteSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        *,
        mlp_chunk_size: int,
        decode_cuda_graph: bool,
    ) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.mlp_chunk_size = int(mlp_chunk_size)
        if self.mlp_chunk_size <= 0:
            raise ValueError(
                f"mlp_chunk_size must be positive, got {self.mlp_chunk_size}."
            )
        self.gate = Glm4MoeLiteRouter(config)
        self.experts = Glm4MoeLitePackedExperts(
            config,
            decode_cuda_graph=decode_cuda_graph,
        )
        self.shared_experts = Qwen3MLP(
            hidden_size=int(config.hidden_size),
            intermediate_size=(
                int(config.moe_intermediate_size) * int(config.n_shared_experts)
            ),
            hidden_act=str(config.hidden_act),
            mlp_chunk_size=self.mlp_chunk_size,
            quantization=None,
        )

    def _routed_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, topk_weights, topk_ids = self.gate(hidden_states)
        return self.experts(hidden_states, topk_ids, topk_weights)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError(
                "Glm4MoeLiteSparseMoeBlock expects [tokens, hidden], got "
                f"{tuple(hidden_states.shape)}."
            )
        if int(hidden_states.shape[0]) <= self.mlp_chunk_size:
            routed = self._routed_chunk(hidden_states)
        else:
            routed = torch.cat(
                [
                    self._routed_chunk(chunk)
                    for chunk in hidden_states.split(self.mlp_chunk_size, dim=0)
                ],
                dim=0,
            )
        self.parallel_context.world_all_reduce(routed)
        return routed + self.shared_experts(hidden_states)


class Glm4MoeLiteDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        layer_idx: int,
        mla_attention: MLAAttention,
        *,
        mlp_chunk_size: int,
        decode_cuda_graph: bool,
    ) -> None:
        super().__init__()
        self.self_attn = Glm4MoeLiteAttention(
            config,
            mla_attention,
            projection_chunk_size=mlp_chunk_size,
        )
        layer_types = list(config.mlp_layer_types)
        if len(layer_types) != int(config.num_hidden_layers):
            raise ValueError(
                "GLM mlp_layer_types length must match num_hidden_layers, got "
                f"{len(layer_types)} and {config.num_hidden_layers}."
            )
        layer_type = str(layer_types[int(layer_idx)])
        if layer_type == "dense":
            self.mlp = Qwen3MLP(
                hidden_size=int(config.hidden_size),
                intermediate_size=int(config.intermediate_size),
                hidden_act=str(config.hidden_act),
                mlp_chunk_size=int(mlp_chunk_size),
                quantization=None,
            )
        elif layer_type == "sparse":
            self.mlp = Glm4MoeLiteSparseMoeBlock(
                config,
                mlp_chunk_size=mlp_chunk_size,
                decode_cuda_graph=decode_cuda_graph,
            )
        else:
            raise ValueError(
                f"Unsupported GLM MLP layer type at layer {layer_idx}: {layer_type!r}."
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        rotary_emb: RotaryEmbedding,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states, rotary_emb)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Glm4MoeLiteModel(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        mla_attention: MLAAttention,
        *,
        mlp_chunk_size: int,
        decode_cuda_graph: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.mla_attention = mla_attention
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
        )
        rope_parameters = getattr(config, "rope_parameters", None) or {}
        self.rotary_emb = get_rope(
            int(config.qk_rope_head_dim),
            rotary_dim=int(config.qk_rope_head_dim),
            max_position=int(config.max_position_embeddings),
            base=float(rope_parameters.get("rope_theta", 1_000_000.0)),
            rope_scaling=None,
            backend="torch",
            interleaved=True,
        )
        self.layers = nn.ModuleList(
            [
                Glm4MoeLiteDecoderLayer(
                    config,
                    layer_idx,
                    mla_attention,
                    mlp_chunk_size=mlp_chunk_size,
                    decode_cuda_graph=decode_cuda_graph,
                )
                for layer_idx in range(int(config.num_hidden_layers))
            ]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.sparse_controller = None

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)
        residual = None
        context = get_context()
        debug_layers_env = os.getenv("SPARSEVLLM_DEBUG_HIDDEN_LAYERS")
        debug_layers = None
        if debug_layers_env:
            debug_layers = {
                int(part) for part in debug_layers_env.split(",") if part.strip()
            }
            self.debug_last_hidden_states = {
                -1: hidden_states[-1:].detach().clone(),
            }

        for layer_idx, layer in enumerate(self.layers):
            context.now_layer_idx = layer_idx
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                self.rotary_emb,
            )
            if self.sparse_controller is not None:
                hidden_states, residual = self.sparse_controller.apply_activation_hook(
                    layer_idx,
                    hidden_states,
                    residual,
                    context,
                )
            if debug_layers is not None and layer_idx in debug_layers:
                layer_output = (
                    hidden_states if residual is None else hidden_states + residual
                )
                self.debug_last_hidden_states[layer_idx] = (
                    layer_output[-1:].detach().clone()
                )
            if self.sparse_controller is not None:
                self.sparse_controller.on_layer_end(layer_idx, context)

        hidden_states, _ = self.norm(hidden_states, residual)
        if debug_layers is not None:
            self.debug_last_hidden_states[int(self.config.num_hidden_layers)] = (
                hidden_states[-1:].detach().clone()
            )
        return hidden_states


def _expected_mtp_weight_names(num_experts: int) -> set[str]:
    prefix = _MTP_PREFIX
    names = {
        prefix + suffix
        for suffix in (
            "eh_proj.weight",
            "embed_tokens.weight",
            "enorm.weight",
            "hnorm.weight",
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.kv_a_layernorm.weight",
            "self_attn.kv_a_proj_with_mqa.weight",
            "self_attn.kv_b_proj.weight",
            "self_attn.o_proj.weight",
            "self_attn.q_a_layernorm.weight",
            "self_attn.q_a_proj.weight",
            "self_attn.q_b_proj.weight",
            "mlp.gate.e_score_correction_bias",
            "mlp.gate.weight",
            "mlp.shared_experts.down_proj.weight",
            "mlp.shared_experts.gate_proj.weight",
            "mlp.shared_experts.up_proj.weight",
            "shared_head.head.weight",
            "shared_head.norm.weight",
        )
    }
    names.update(
        f"{prefix}mlp.experts.{expert_id}.{projection}.weight"
        for expert_id in range(int(num_experts))
        for projection in ("gate_proj", "up_proj", "down_proj")
    )
    return names


class Glm4MoeLiteForCausalLM(nn.Module):
    special_weight_loaders = (".expert_weight",)
    packed_modules_mapping = {
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        *,
        mla_attention: MLAAttention,
        mlp_chunk_size: int,
        decode_cuda_graph: bool,
        expect_mtp_weights: bool,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallel_context = get_parallel_context()
        self.model = Glm4MoeLiteModel(
            config,
            mla_attention,
            mlp_chunk_size=mlp_chunk_size,
            decode_cuda_graph=decode_cuda_graph,
        )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.expect_mtp_weights = bool(expect_mtp_weights)
        self._intentionally_skipped_expert_weights: set[str] = set()
        self._intentionally_skipped_mtp_weights: set[str] = set()

    def _sparse_block(self, layer_idx: int) -> Glm4MoeLiteSparseMoeBlock:
        if not 0 <= int(layer_idx) < len(self.model.layers):
            raise ValueError(
                f"GLM expert checkpoint layer {layer_idx} is outside the base model."
            )
        block = self.model.layers[int(layer_idx)].mlp
        if not isinstance(block, Glm4MoeLiteSparseMoeBlock):
            raise ValueError(
                f"GLM checkpoint contains experts for dense layer {layer_idx}."
            )
        return block

    def iter_tiny_reference_weights(
        self,
        state_dict: dict[str, torch.Tensor],
    ):
        """Expand Transformers' packed tiny experts into checkpoint-style names."""

        gate_up_suffix = ".mlp.experts.gate_up_proj"
        down_suffix = ".mlp.experts.down_proj"
        for source_name, weight in state_dict.items():
            if source_name.endswith(gate_up_suffix):
                prefix = source_name[: -len("gate_up_proj")]
                intermediate_size = int(weight.shape[1]) // 2
                for expert_id in range(int(weight.shape[0])):
                    yield (
                        f"{prefix}{expert_id}.gate_proj.weight",
                        weight[expert_id, :intermediate_size],
                    )
                    yield (
                        f"{prefix}{expert_id}.up_proj.weight",
                        weight[expert_id, intermediate_size:],
                    )
                continue
            if source_name.endswith(down_suffix):
                prefix = source_name[: -len("down_proj")]
                for expert_id in range(int(weight.shape[0])):
                    yield (
                        f"{prefix}{expert_id}.down_proj.weight",
                        weight[expert_id],
                    )
                continue
            yield source_name, weight

    @torch.inference_mode()
    def warmup_moe(self, num_tokens: int = 1) -> None:
        num_tokens = int(num_tokens)
        if num_tokens <= 0:
            raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
        block = next(
            (
                layer.mlp
                for layer in self.model.layers
                if isinstance(layer.mlp, Glm4MoeLiteSparseMoeBlock)
            ),
            None,
        )
        if block is None:
            raise RuntimeError("GLM model has no sparse MoE layer to warm up.")
        experts = block.experts
        hidden_states = torch.zeros(
            (num_tokens, experts.hidden_size),
            dtype=model_activation_dtype(self.config),
            device=experts.w13_weight.device,
        )
        block.gate(hidden_states)
        top_k = int(self.config.num_experts_per_tok)
        topk_ids = (
            torch.arange(
                num_tokens * top_k,
                dtype=torch.int64,
                device=hidden_states.device,
            )
            .remainder(experts.num_local_experts)
            .add(experts.local_expert_start)
            .view(num_tokens, top_k)
        )
        topk_weights = torch.full(
            (num_tokens, top_k),
            1.0 / top_k,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        experts(hidden_states, topk_ids, topk_weights)
        device_runtime.synchronize()

    def map_weight_name(self, source_weight_name: str) -> str | None:
        if source_weight_name.startswith(_MTP_PREFIX):
            return None
        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is None:
            return source_weight_name
        layer_idx, global_expert_id, projection = match.groups()
        block = self._sparse_block(int(layer_idx))
        global_expert_id = int(global_expert_id)
        if not block.experts.is_local_expert(global_expert_id):
            self._intentionally_skipped_expert_weights.add(source_weight_name)
            return None
        return (
            f"model.layers.{layer_idx}.mlp.experts.{global_expert_id}."
            f"{projection}.expert_weight"
        )

    def resolve_special_weight(
        self,
        target_weight_name: str,
    ) -> WeightTarget | None:
        match = _EXPERT_TARGET_RE.match(target_weight_name)
        if match is None:
            return None
        layer_idx, global_expert_id, projection = match.groups()
        experts = self._sparse_block(int(layer_idx)).experts
        return WeightTarget(experts, (int(global_expert_id), projection))

    def record_skipped_weight(
        self,
        source_weight_name: str,
        loaded_weight_shape: tuple[int, ...] | None,
        loaded_weight_dtype: str | None,
        loaded_scale_shape: tuple[int, ...] | None,
        loaded_scale_dtype: str | None,
    ) -> None:
        if source_weight_name.startswith(_MTP_PREFIX):
            if loaded_weight_shape is None or any(
                int(dim) <= 0 for dim in loaded_weight_shape
            ):
                raise ValueError(
                    f"Invalid GLM MTP tensor shape for {source_weight_name!r}: "
                    f"{loaded_weight_shape}."
                )
            if loaded_weight_dtype not in {"BF16", "F32"}:
                raise TypeError(
                    f"Unsupported GLM MTP dtype for {source_weight_name!r}: "
                    f"{loaded_weight_dtype}."
                )
            if loaded_scale_shape is not None or loaded_scale_dtype is not None:
                raise ValueError(
                    f"Unquantized GLM MTP tensor has an unexpected scale: "
                    f"{source_weight_name!r}."
                )
            self._intentionally_skipped_mtp_weights.add(source_weight_name)
            return

        match = _EXPERT_SOURCE_RE.match(source_weight_name)
        if match is None:
            raise ValueError(
                f"GLM loader unexpectedly skipped {source_weight_name!r}."
            )
        layer_idx, global_expert_id, projection = match.groups()
        experts = self._sparse_block(int(layer_idx)).experts
        global_expert_id = int(global_expert_id)
        if experts.is_local_expert(global_expert_id):
            raise ValueError(
                f"GLM loader skipped local expert weight {source_weight_name!r}."
            )
        expected_shape = (
            (experts.hidden_size, experts.global_intermediate_size)
            if projection == "down_proj"
            else (experts.global_intermediate_size, experts.hidden_size)
        )
        if loaded_weight_shape != expected_shape:
            raise ValueError(
                "Remote GLM expert weight shape mismatch for "
                f"{source_weight_name!r}: expected={expected_shape}, "
                f"got={loaded_weight_shape}."
            )
        if loaded_weight_dtype not in {"BF16", "F16", "F32"}:
            raise TypeError(
                f"Unsupported remote GLM expert dtype for "
                f"{source_weight_name!r}: {loaded_weight_dtype}."
            )
        if loaded_scale_shape is not None or loaded_scale_dtype is not None:
            raise ValueError(
                f"Unquantized remote GLM expert has an unexpected scale: "
                f"{source_weight_name!r}."
            )
        self._intentionally_skipped_expert_weights.add(source_weight_name)

    def load_special_weight(
        self,
        target_weight_name: str,
        loaded_weight: torch.Tensor,
        loaded_scale: torch.Tensor | None,
    ) -> int:
        target = self.resolve_special_weight(target_weight_name)
        if target is None:
            return 0
        global_expert_id, projection = target.shard_id
        target.module.load_expert_weight(
            global_expert_id,
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
                f"Missing replicated GLM base weights: {missing_dense[:8]}."
            )

        sparse_blocks = [
            layer.mlp
            for layer in self.model.layers
            if isinstance(layer.mlp, Glm4MoeLiteSparseMoeBlock)
        ]
        for block in sparse_blocks:
            block.experts.validate_loaded_weights()

        expected_remote = {
            f"model.layers.{layer_idx}.mlp.experts.{expert_id}."
            f"{projection}.weight"
            for layer_idx, layer in enumerate(self.model.layers)
            if isinstance(layer.mlp, Glm4MoeLiteSparseMoeBlock)
            for expert_id in range(int(self.config.n_routed_experts))
            if not layer.mlp.experts.is_local_expert(expert_id)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        missing_remote = sorted(
            expected_remote - self._intentionally_skipped_expert_weights
        )
        unexpected_remote = sorted(
            self._intentionally_skipped_expert_weights - expected_remote
        )
        if missing_remote or unexpected_remote:
            raise ValueError(
                "GLM remote expert skip set is inconsistent: "
                f"missing={missing_remote[:4]} unexpected={unexpected_remote[:4]}."
            )

        expected_mtp = _expected_mtp_weight_names(self.config.n_routed_experts)
        if self.expect_mtp_weights:
            missing_mtp = sorted(
                expected_mtp - self._intentionally_skipped_mtp_weights
            )
            unexpected_mtp = sorted(
                self._intentionally_skipped_mtp_weights - expected_mtp
            )
            if missing_mtp or unexpected_mtp:
                raise ValueError(
                    "GLM MTP skip set is inconsistent: "
                    f"missing={missing_mtp[:4]} unexpected={unexpected_mtp[:4]}."
                )
        elif self._intentionally_skipped_mtp_weights:
            raise ValueError(
                "GLM checkpoint supplied MTP weights when they were not expected: "
                f"{sorted(self._intentionally_skipped_mtp_weights)[:4]}."
            )

        first_sparse = sparse_blocks[0] if sparse_blocks else None
        logger.info(
            "Loaded GLM-4.7-Flash rank {} MLA provider={} MoE provider={} "
            "attention TP {}/{} across {} base layers; local expert shards={} "
            "remote expert skips={} MTP skips={}.",
            self.parallel_context.world_rank,
            self.model.mla_attention.provider.name,
            first_sparse.experts.provider.name if first_sparse is not None else "none",
            self.parallel_context.attention_tp_rank,
            self.parallel_context.attention_tp_size,
            len(self.model.layers),
            sum(len(block.experts._loaded_expert_shards) for block in sparse_blocks),
            len(self._intentionally_skipped_expert_weights),
            len(self._intentionally_skipped_mtp_weights),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


__all__ = [
    "Glm4MoeLiteAttention",
    "Glm4MoeLiteForCausalLM",
    "Glm4MoeLiteModel",
    "Glm4MoeLitePackedExperts",
    "Glm4MoeLiteRouter",
    "Glm4MoeLiteSparseMoeBlock",
    "build_glm4_moe_lite_mla_attention",
]
