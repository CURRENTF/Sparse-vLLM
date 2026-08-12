from __future__ import annotations

import os
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from transformers import Glm4MoeLiteConfig

from sparsevllm.distributed import ParallelContext, ParallelGroup, get_parallel_context
from sparsevllm.models.layout import resolve_attention_qk_head_dim
from sparsevllm.engine.cache_manager import (
    AttentionKeyComputeView,
    MlaLatentWrite,
)
from sparsevllm.layers.embed_head import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sparsevllm.layers.activation import SiluAndMul
from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.layers.linear import (
    ColumnParallelLinear,
    MergedReplicatedLinear,
    RowParallelLinear,
)
from sparsevllm.layers.mla_attention import MLAAttention
from sparsevllm.layers.packed_moe import PackedMoeExperts
from sparsevllm.layers.rotary_embedding import RotaryEmbedding, get_rope
from sparsevllm.models.qwen3 import Qwen3MLP
from sparsevllm.operators.mla_attention import MlaAttentionOpSpec
from sparsevllm.operators.all_reduce import (
    AllReduceOpSpec,
    PreparedAllReduceOp,
    prepare_all_reduce_op,
)
from sparsevllm.operators.activation import resolve_silu_and_mul_provider
from sparsevllm.operators.moe import (
    MoeOpSpec,
    model_activation_dtype,
    resolve_moe_provider,
)
from sparsevllm.operators.moe_router import (
    MoeRouterOpSpec,
    resolve_moe_router_provider,
)
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
_SHARED_EXPERT_SOURCE_RE = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.shared_experts\."
    r"(gate_proj|up_proj|down_proj)\.weight$"
)
_MTP_LAYER_INDEX = 47
_MTP_PREFIX = f"model.layers.{_MTP_LAYER_INDEX}."


@dataclass
class Glm4MoeLiteRuntimeConfig:
    attention_decode_all_reduce: PreparedAllReduceOp
    moe_decode_all_reduce: PreparedAllReduceOp
    _closed: bool = False

    def close(self) -> None:
        if self._closed:
            return
        ops = (self.attention_decode_all_reduce, self.moe_decode_all_reduce)
        for op in {id(op): op for op in ops}.values():
            op.close()
        self._closed = True


def _prepare_decode_all_reduce(
    group: ParallelGroup,
    *,
    max_rows: int,
    hidden_size: int,
    dtype: torch.dtype,
    cuda_graph: bool,
    device_index: int,
) -> PreparedAllReduceOp:
    return prepare_all_reduce_op(
        AllReduceOpSpec(
            world_size=group.size,
            ranks=group.ranks,
            max_rows=max_rows,
            hidden_size=hidden_size,
            dtype=dtype,
            cuda_graph=cuda_graph,
            backend=(
                "none"
                if group.process_group is None
                else str(torch.distributed.get_backend(group.process_group))
            ),
        ),
        group=group.process_group,
        rank=group.rank,
        device_index=device_index,
    )


def build_glm4_moe_lite_runtime_config(
    config: Glm4MoeLiteConfig,
    parallel_context: ParallelContext,
    *,
    max_decode_tokens: int,
    cuda_graph: bool,
    device_index: int,
) -> Glm4MoeLiteRuntimeConfig:
    max_decode_tokens = int(max_decode_tokens)
    moe_op = _prepare_decode_all_reduce(
        parallel_context.world,
        max_rows=2 * max_decode_tokens,
        hidden_size=int(config.hidden_size),
        dtype=model_activation_dtype(config),
        cuda_graph=cuda_graph,
        device_index=device_index,
    )
    attention_op = (
        moe_op
        if parallel_context.attention.ranks == parallel_context.world.ranks
        else _prepare_decode_all_reduce(
            parallel_context.attention,
            max_rows=max_decode_tokens,
            hidden_size=int(config.hidden_size),
            dtype=model_activation_dtype(config),
            cuda_graph=cuda_graph,
            device_index=device_index,
        )
    )
    return Glm4MoeLiteRuntimeConfig(attention_op, moe_op)


def build_glm4_moe_lite_mla_attention(
    config: Glm4MoeLiteConfig,
    *,
    device: torch.device | str,
    max_batch_size: int,
    prefill_workspace_bytes: int,
    decode_cuda_graph: bool,
    projection_chunk_size: int,
) -> MLAAttention:
    """Bind the one process-local MLA operator from explicit runtime inputs."""

    parallel_context = get_parallel_context()
    activation_dtype = model_activation_dtype(config)
    spec = MlaAttentionOpSpec(
        num_q_heads=int(config.num_attention_heads),
        kv_lora_rank=int(config.kv_lora_rank),
        rope_dim=int(config.qk_rope_head_dim),
        qk_head_dim=resolve_attention_qk_head_dim(config),
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
        hidden_size=int(config.hidden_size),
        projection_chunk_size=projection_chunk_size,
    )


class Glm4MoeLiteAttention(nn.Module):
    """GLM projections around a shared latent-MLA execution object."""

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        mla_attention: MLAAttention,
        *,
        projection_chunk_size: int,
        runtime_config: Glm4MoeLiteRuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.mla_attention = mla_attention
        self.parallel_context = get_parallel_context()
        self.runtime_config = runtime_config
        self.num_heads = int(config.num_attention_heads)
        self.local_heads = int(mla_attention.spec.local_q_heads)
        self.q_lora_rank = int(config.q_lora_rank)
        self.kv_lora_rank = int(config.kv_lora_rank)
        self.qk_nope_head_dim = int(config.qk_nope_head_dim)
        self.qk_rope_head_dim = int(config.qk_rope_head_dim)
        self.qk_head_dim = resolve_attention_qk_head_dim(config)
        self.v_head_dim = int(config.v_head_dim)
        self.proj_chunk_size = int(projection_chunk_size)
        if self.proj_chunk_size <= 0:
            raise ValueError(
                f"mlp_chunk_size must be positive, got {self.proj_chunk_size}."
            )
        if self.proj_chunk_size != int(mla_attention.projection_chunk_size):
            raise ValueError(
                "GLM projection chunk size must match the MLA workspace bound: "
                f"model={self.proj_chunk_size} "
                f"mla={mla_attention.projection_chunk_size}."
            )
        if int(config.hidden_size) != int(mla_attention.hidden_size):
            raise ValueError(
                "GLM hidden size must match the MLA workspace bound: "
                f"model={config.hidden_size} mla={mla_attention.hidden_size}."
            )
        quantization = getattr(config, "quantization_config", None)
        if bool(getattr(quantization, "enabled", False)):
            raise NotImplementedError("GLM MLA projections do not support quantization.")

        self.fused_qkv_a_proj = MergedReplicatedLinear(
            int(config.hidden_size),
            [
                self.q_lora_rank,
                self.kv_lora_rank + self.qk_rope_head_dim,
            ],
            bias=bool(config.attention_bias),
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * self.qk_head_dim,
            bias=False,
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
            reduce_results=runtime_config is None,
        )
        self._attention_key_materializer_binding: tuple[int, int] | None = None

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

    def _materialize_attention_keys(
        self,
        view: AttentionKeyComputeView,
    ) -> torch.Tensor:
        return self.mla_attention.materialize_expanded_keys(
            view,
            project_latent=self._project_kv_history,
        )

    def _ensure_attention_key_materializer(self, cache_manager, layer_idx: int) -> None:
        binding = (id(cache_manager), int(layer_idx))
        if self._attention_key_materializer_binding == binding:
            return
        cache_manager.register_attention_key_materializer(
            int(layer_idx),
            self._materialize_attention_keys,
        )
        self._attention_key_materializer_binding = binding

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
        self._ensure_attention_key_materializer(cache_manager, layer_idx)
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
                history = self.mla_attention.prepare_prefill_history(
                    view,
                    query_tokens=int(q.shape[0]),
                )
                expanded = self._project_kv_history(history.gathered_latent).view(
                    history.visible_tokens,
                    self.local_heads,
                    self.qk_nope_head_dim + self.v_head_dim,
                )
                expanded_k_nope, expanded_v = expanded.split(
                    [self.qk_nope_head_dim, self.v_head_dim],
                    dim=-1,
                )
                expanded_k = torch.empty(
                    (
                        history.visible_tokens,
                        self.local_heads,
                        self.qk_head_dim,
                    ),
                    dtype=expanded_k_nope.dtype,
                    device=expanded_k_nope.device,
                )
                expanded_k[..., : self.qk_nope_head_dim].copy_(expanded_k_nope)
                expanded_k[..., self.qk_nope_head_dim :].copy_(
                    history.gathered_rope[:, None, :]
                )
                workset = self.mla_attention.bind_prefill_kv(
                    history,
                    expanded_k=expanded_k,
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
                    self.mla_attention.build_prefill_explicit_view(workset),
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
        compressed_qkv = self.fused_qkv_a_proj(hidden_states)
        compressed_q, compressed_kv = compressed_qkv.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q = self.q_b_proj(self.q_a_layernorm(compressed_q))
        q = q.view(-1, self.local_heads, self.qk_head_dim)
        q_nope, q_rope = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim],
            dim=-1,
        )

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
        output = self._project_output(value_output, hidden_states)
        if self.runtime_config is None:
            return output
        if get_context().is_prefill:
            return self.parallel_context.attention_tp_all_reduce(output)
        return self.runtime_config.attention_decode_all_reduce.run(output)


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
        self.op_spec = MoeRouterOpSpec(
            num_experts=self.num_experts,
            top_k=self.top_k,
            activation_dtype=torch.float32,
            norm_topk_prob=True,
            cuda_graph=bool(getattr(config, "decode_cuda_graph", False)),
            routing_method="biased_sigmoid",
        )
        self.provider = resolve_moe_router_provider(self.op_spec)

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        router_logits = F.linear(hidden_states.float(), self.weight)
        topk_weights, topk_ids = self.provider.run(
            self.op_spec,
            router_logits,
            self.e_score_correction_bias,
            routed_scaling_factor=self.routed_scaling_factor,
        )
        return router_logits, topk_weights, topk_ids


class Glm4MoeLitePackedExperts(PackedMoeExperts):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        *,
        decode_cuda_graph: bool,
    ) -> None:
        parallel_context = get_parallel_context()
        self.routed_num_experts = int(config.n_routed_experts)
        self.routed_top_k = int(config.num_experts_per_tok)
        self.fuses_shared_decode = bool(
            decode_cuda_graph
            and (
                self.routed_num_experts,
                int(config.n_shared_experts),
                self.routed_top_k,
                int(config.hidden_size),
                int(config.moe_intermediate_size),
                int(parallel_context.moe_tp_size),
                int(parallel_context.ep_size),
            )
            == (64, 1, 4, 2048, 1536, 2, 1)
        )
        packed_num_experts = self.routed_num_experts + int(
            self.fuses_shared_decode
        )
        packed_top_k = self.routed_top_k + int(self.fuses_shared_decode)
        super().__init__(
            num_experts=packed_num_experts,
            hidden_size=int(config.hidden_size),
            intermediate_size=int(config.moe_intermediate_size),
            top_k=packed_top_k,
            activation_dtype=model_activation_dtype(config),
            fp8_enabled=False,
            cuda_graph=bool(decode_cuda_graph),
            routing_method="biased_sigmoid",
            model_label="GLM-4.7-Flash",
            provider_resolver=resolve_moe_provider,
            parallel_context=parallel_context,
        )
        self.shared_expert_id = (
            self.routed_num_experts if self.fuses_shared_decode else None
        )
        self.shared_act = SiluAndMul(
            provider=resolve_silu_and_mul_provider(
                activation_dtype=model_activation_dtype(config),
            )
        )
        if self.fuses_shared_decode:
            self.routed_op_spec = MoeOpSpec(
                num_experts=self.routed_num_experts,
                num_local_experts=self.routed_num_experts,
                hidden_size=self.hidden_size,
                intermediate_size=self.intermediate_size,
                top_k=self.routed_top_k,
                activation_dtype=self.op_spec.activation_dtype,
                weight_dtype=self.op_spec.weight_dtype,
                block_shape=self.op_spec.block_shape,
                ep_size=1,
                cuda_graph=self.op_spec.cuda_graph,
                tp_size=self.op_spec.tp_size,
                routing_method=self.op_spec.routing_method,
                scale_dtype=self.op_spec.scale_dtype,
            )
            self.routed_provider = resolve_moe_provider(self.routed_op_spec)
        else:
            self.routed_op_spec = self.op_spec
            self.routed_provider = self.provider

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if not self.fuses_shared_decode:
            return super().forward(hidden_states, topk_ids, topk_weights)
        return self.routed_provider.run(
            self.routed_op_spec,
            hidden_states,
            topk_ids,
            topk_weights,
            self.w13_weight[: self.routed_num_experts],
            self.w2_weight[: self.routed_num_experts],
            self.w13_scale_inv,
            self.w2_scale_inv,
            local_expert_start=0,
            ep_rank=int(self.ep_rank),
        )

    def forward_shared(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shared_expert_id is None:
            raise RuntimeError("Packed shared expert is not enabled.")
        gate_up = F.linear(
            hidden_states,
            self.w13_weight[self.shared_expert_id],
        )
        return F.linear(
            self.shared_act(gate_up),
            self.w2_weight[self.shared_expert_id],
        )

    def forward_routed_and_shared(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.shared_expert_id is None:
            raise RuntimeError("Packed shared expert is not enabled.")
        from sparsevllm.triton_kernel.moe import append_shared_expert_route

        fused_ids, fused_weights = append_shared_expert_route(
            topk_ids,
            topk_weights,
            shared_expert_id=self.shared_expert_id,
        )
        return super().forward(hidden_states, fused_ids, fused_weights)


class Glm4MoeLiteSparseMoeBlock(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        *,
        mlp_chunk_size: int,
        decode_cuda_graph: bool,
        runtime_config: Glm4MoeLiteRuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.runtime_config = runtime_config
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
        self.shared_experts = (
            None
            if self.experts.fuses_shared_decode
            else Qwen3MLP(
                hidden_size=int(config.hidden_size),
                intermediate_size=(
                    int(config.moe_intermediate_size)
                    * int(config.n_shared_experts)
                ),
                hidden_act=str(config.hidden_act),
                mlp_chunk_size=self.mlp_chunk_size,
                quantization=None,
                reduce_results=False,
                activation_provider=resolve_silu_and_mul_provider(
                    activation_dtype=model_activation_dtype(config),
                ),
            )
        )

    def _routed_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        _, topk_weights, topk_ids = self.gate(hidden_states)
        return self.experts(hidden_states, topk_ids, topk_weights)

    def _shared_chunk(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self.shared_experts is not None:
            return self.shared_experts(hidden_states)
        return self.experts.forward_shared(hidden_states)

    def _routed_and_shared_chunk(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        _, topk_weights, topk_ids = self.gate(hidden_states)
        return self.experts.forward_routed_and_shared(
            hidden_states,
            topk_ids,
            topk_weights,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.ndim != 2:
            raise ValueError(
                "Glm4MoeLiteSparseMoeBlock expects [tokens, hidden], got "
                f"{tuple(hidden_states.shape)}."
            )
        debug_enabled = os.getenv("SPARSEVLLM_DEBUG_MOE", "0") == "1"
        if debug_enabled:
            self.debug_last_input = hidden_states.detach().clone()
        context = get_context()
        if (
            not debug_enabled
            and getattr(
                getattr(self, "experts", None),
                "fuses_shared_decode",
                False,
            )
            and not context.is_prefill
        ):
            if int(hidden_states.shape[0]) <= self.mlp_chunk_size:
                local_output = self._routed_and_shared_chunk(hidden_states)
            else:
                local_output = torch.cat(
                    [
                        self._routed_and_shared_chunk(chunk)
                        for chunk in hidden_states.split(
                            self.mlp_chunk_size,
                            dim=0,
                        )
                    ],
                    dim=0,
                )
            if self.runtime_config is not None:
                return self.runtime_config.moe_decode_all_reduce.run(local_output)
            return self.parallel_context.world_all_reduce_out_of_place(local_output)
        if not debug_enabled:
            if int(hidden_states.shape[0]) <= self.mlp_chunk_size:
                routed = self._routed_chunk(hidden_states)
            else:
                routed = torch.cat(
                    [
                        self._routed_chunk(chunk)
                        for chunk in hidden_states.split(
                            self.mlp_chunk_size,
                            dim=0,
                        )
                    ],
                    dim=0,
                )
        elif int(hidden_states.shape[0]) <= self.mlp_chunk_size:
            router_logits, topk_weights, topk_ids = self.gate(hidden_states)
            routed = self.experts(hidden_states, topk_ids, topk_weights)
        else:
            router_logits_chunks = []
            topk_weights_chunks = []
            topk_ids_chunks = []
            routed_chunks = []
            for chunk in hidden_states.split(self.mlp_chunk_size, dim=0):
                router_logits, topk_weights, topk_ids = self.gate(chunk)
                routed_chunks.append(
                    self.experts(chunk, topk_ids, topk_weights)
                )
                router_logits_chunks.append(router_logits)
                topk_weights_chunks.append(topk_weights)
                topk_ids_chunks.append(topk_ids)
            routed = torch.cat(routed_chunks, dim=0)
            router_logits = torch.cat(router_logits_chunks, dim=0)
            topk_weights = torch.cat(topk_weights_chunks, dim=0)
            topk_ids = torch.cat(topk_ids_chunks, dim=0)
        if debug_enabled:
            self.debug_last_router_logits = router_logits.detach().clone()
            self.debug_last_topk_ids = topk_ids.detach().clone()
            self.debug_last_topk_weights = topk_weights.detach().clone()
            self.debug_last_local_output = routed.detach().clone()
            local_mask = (topk_ids >= self.experts.local_expert_start) & (
                topk_ids < self.experts.local_expert_end
            )
            local_hit_count = local_mask.sum()
            self.debug_last_local_hit_count = (
                local_hit_count
                if device_runtime.is_stream_capturing()
                else int(local_hit_count.item())
            )
        if debug_enabled:
            # Preserve the general EP composition and routed-only debug
            # evidence while keeping shared-expert reductions explicit.
            self.parallel_context.world_all_reduce(routed)
            self.debug_last_routed_output = routed.detach().clone()
            shared = self._shared_chunk(hidden_states)
            if self.parallel_context.tp_size > 1:
                shared = self.parallel_context.tp_all_reduce(shared)
            output = routed + shared
        elif self.parallel_context.ep_size > 1:
            shared_local = self._shared_chunk(hidden_states)
            if self.parallel_context.tp_size > 1:
                # Hybrid TP+EP makes both branches partial over the same
                # outer world. Sum them locally so one collective completes
                # routed experts and the TP-sharded shared expert together.
                local_output = routed + shared_local
                output = (
                    self.runtime_config.moe_decode_all_reduce.run(local_output)
                    if self.runtime_config is not None and not context.is_prefill
                    else self.parallel_context.world_all_reduce_out_of_place(
                        local_output
                    )
                )
            else:
                # Retain the pure-EP semantic path for direct module use: the
                # shared expert is replicated rather than TP-sharded.
                routed = (
                    self.runtime_config.moe_decode_all_reduce.run(routed)
                    if self.runtime_config is not None and not context.is_prefill
                    else self.parallel_context.world_all_reduce(routed)
                )
                output = routed + shared_local
        else:
            shared_local = self._shared_chunk(hidden_states)
            if context.is_prefill:
                # Both branches are TP partials. Compose them locally so the
                # full MoE block needs one collective, matching the fused-MoE
                # communication contract used by the reference runtime.
                output = self.parallel_context.world_all_reduce_out_of_place(
                    routed + shared_local
                )
            else:
                # Decode tensors are small. Pack both partials into one
                # collective, then add the independently reduced rows. This
                # preserves the original BF16 reduction/addition order.
                partials = torch.stack((routed, shared_local), dim=0)
                partials = (
                    self.runtime_config.moe_decode_all_reduce.run(partials)
                    if self.runtime_config is not None
                    else self.parallel_context.world_all_reduce_out_of_place(
                        partials
                    )
                )
                output = partials[0] + partials[1]
        if debug_enabled:
            # ModelRunner's cross-rank evidence contract consumes the final
            # MoE block output, including both the synced routed experts and
            # the synchronized shared-expert branch.
            self.debug_last_output = output.detach().clone()
        return output


class Glm4MoeLiteDecoderLayer(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        layer_idx: int,
        mla_attention: MLAAttention,
        *,
        mlp_chunk_size: int,
        decode_cuda_graph: bool,
        runtime_config: Glm4MoeLiteRuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.parallel_context = get_parallel_context()
        self.runtime_config = runtime_config
        self.self_attn = Glm4MoeLiteAttention(
            config,
            mla_attention,
            projection_chunk_size=mlp_chunk_size,
            runtime_config=runtime_config,
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
                reduce_results=runtime_config is None,
                activation_provider=resolve_silu_and_mul_provider(
                    activation_dtype=model_activation_dtype(config),
                ),
            )
        elif layer_type == "sparse":
            self.mlp = Glm4MoeLiteSparseMoeBlock(
                config,
                mlp_chunk_size=mlp_chunk_size,
                decode_cuda_graph=decode_cuda_graph,
                runtime_config=runtime_config,
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
        if self.parallel_context.tp_size == 1 and self.parallel_context.ep_size > 1:
            # Replicated MLA must enter the post-attention norm identically on
            # every expert rank before routed experts make their next decision.
            self.parallel_context.ep_broadcast(hidden_states, src_ep_rank=0)
        hidden_states, residual = self.post_attention_layernorm(
            hidden_states,
            residual,
        )
        hidden_states = self.mlp(hidden_states)
        if self.runtime_config is not None and isinstance(self.mlp, Qwen3MLP):
            hidden_states = (
                self.parallel_context.attention_tp_all_reduce(hidden_states)
                if get_context().is_prefill
                else self.runtime_config.attention_decode_all_reduce.run(hidden_states)
            )
        return hidden_states, residual


class Glm4MoeLiteModel(nn.Module):
    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        mla_attention: MLAAttention,
        *,
        mlp_chunk_size: int,
        decode_cuda_graph: bool,
        runtime_config: Glm4MoeLiteRuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.mla_attention = mla_attention
        self.parallel_context = get_parallel_context()
        self.runtime_config = runtime_config
        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            reduce_results=runtime_config is None,
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
                    runtime_config=runtime_config,
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
        context = get_context()
        hidden_states = self.embed_tokens(input_ids)
        if self.runtime_config is not None:
            hidden_states = (
                self.parallel_context.attention_tp_all_reduce(hidden_states)
                if context.is_prefill
                else self.runtime_config.attention_decode_all_reduce.run(hidden_states)
            )
        residual = None
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
        "self_attn.q_a_proj": ("self_attn.fused_qkv_a_proj", 0),
        "self_attn.kv_a_proj_with_mqa": ("self_attn.fused_qkv_a_proj", 1),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    @staticmethod
    def build_runtime_kwargs(
        config: Glm4MoeLiteConfig,
        *,
        engine_config,
        parallel_context: ParallelContext,
        device: torch.device,
        max_decode_tokens: int,
    ) -> dict:
        decode_cuda_graph = bool(engine_config.decode_cuda_graph)
        kwargs = {
            "mla_attention": build_glm4_moe_lite_mla_attention(
                config,
                device=device,
                max_batch_size=max(
                    engine_config.max_num_seqs_in_batch,
                    engine_config.max_decoding_seqs,
                ),
                prefill_workspace_bytes=engine_config.mla_prefill_workspace_bytes,
                decode_cuda_graph=decode_cuda_graph,
                projection_chunk_size=engine_config.mlp_chunk_size,
            ),
            "mlp_chunk_size": engine_config.mlp_chunk_size,
            "decode_cuda_graph": decode_cuda_graph,
            "expect_mtp_weights": not engine_config.tiny_random,
        }
        if parallel_context.world_size > 1:
            kwargs["runtime_config"] = build_glm4_moe_lite_runtime_config(
                config,
                parallel_context,
                max_decode_tokens=max_decode_tokens,
                cuda_graph=decode_cuda_graph,
                device_index=int(device.index or 0),
            )
        return kwargs

    def __init__(
        self,
        config: Glm4MoeLiteConfig,
        *,
        mla_attention: MLAAttention,
        mlp_chunk_size: int,
        decode_cuda_graph: bool,
        expect_mtp_weights: bool,
        runtime_config: Glm4MoeLiteRuntimeConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.parallel_context = get_parallel_context()
        self.runtime_config = runtime_config
        self.model = Glm4MoeLiteModel(
            config,
            mla_attention,
            mlp_chunk_size=mlp_chunk_size,
            decode_cuda_graph=decode_cuda_graph,
            runtime_config=runtime_config,
        )
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if config.tie_word_embeddings:
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self.expect_mtp_weights = bool(expect_mtp_weights)
        self._intentionally_skipped_expert_weights: set[str] = set()
        self._intentionally_skipped_mtp_weights: set[str] = set()

    def close_runtime_operators(self) -> None:
        if self.runtime_config is not None:
            self.runtime_config.close()

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
            .remainder(experts.routed_num_experts)
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
        shared_match = _SHARED_EXPERT_SOURCE_RE.match(source_weight_name)
        if shared_match is not None:
            layer_idx, projection = shared_match.groups()
            experts = self._sparse_block(int(layer_idx)).experts
            if experts.shared_expert_id is not None:
                return (
                    f"model.layers.{layer_idx}.mlp.experts."
                    f"{experts.shared_expert_id}.{projection}.expert_weight"
                )
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
