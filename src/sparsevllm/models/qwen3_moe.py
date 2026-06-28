import os
import re

import torch
import torch.distributed as dist
from torch import nn
import torch.nn.functional as F

from sparsevllm.layers.activation import SiluAndMul
from sparsevllm.layers.embed_head import ParallelLMHead, VocabParallelEmbedding
from sparsevllm.layers.layernorm import RMSNorm
from sparsevllm.layers.linear import MergedColumnParallelLinear, RowParallelLinear
from sparsevllm.models.qwen3 import Qwen3Attention, _get_rope_scaling, _get_rope_theta
from sparsevllm.utils.context import get_context
from sparsevllm.utils.log import logger
from sparsevllm.utils.profiler import profiler
from sparsevllm.utils.parallel_context import get_ep_group, get_ep_rank, get_ep_size


_NAMED_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer_idx>\d+)\.mlp\.experts\.(?P<expert_idx>\d+)\."
    r"(?P<part>gate_proj|up_proj|down_proj)\.weight$"
)
_FUSED_EXPERT_RE = re.compile(
    r"^model\.layers\.(?P<layer_idx>\d+)\.mlp\.experts\.(?P<part>gate_up_proj|down_proj)$"
)


def _get_num_experts(config) -> int:
    num_experts = getattr(config, "num_experts", None)
    if num_experts is None:
        num_experts = getattr(config, "num_local_experts", None)
    if num_experts is None:
        raise ValueError("Qwen3-MoE config must define num_experts or num_local_experts.")
    return int(num_experts)


class Qwen3MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        mlp_chunk_size: int = 16384,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
        )
        if hidden_act != "silu":
            raise NotImplementedError(f"Unsupported Qwen3-MoE hidden_act={hidden_act!r}.")
        self.act_fn = SiluAndMul()
        self.mlp_chunk_size = int(mlp_chunk_size)
        if self.mlp_chunk_size <= 0:
            raise ValueError(f"mlp_chunk_size must be > 0, got {mlp_chunk_size}.")

    def _forward_chunk(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        return self.down_proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunk_size = int(self.mlp_chunk_size)
        if int(x.shape[0]) <= chunk_size:
            return self._forward_chunk(x)

        out = torch.empty_like(x)
        for start in range(0, int(x.shape[0]), chunk_size):
            end = min(start + chunk_size, int(x.shape[0]))
            out[start:end].copy_(self._forward_chunk(x[start:end]))
        return out


class Qwen3MoeTopKRouter(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.top_k = int(config.num_experts_per_tok)
        self.num_experts = _get_num_experts(config)
        self.norm_topk_prob = bool(getattr(config, "norm_topk_prob", False))
        self.hidden_dim = int(config.hidden_size)
        if self.top_k <= 0 or self.top_k > self.num_experts:
            raise ValueError(
                "num_experts_per_tok must be in [1, num_experts], "
                f"got top_k={self.top_k} num_experts={self.num_experts}."
            )
        self.weight = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"Qwen3-MoE router weight shape mismatch: expected={tuple(param.shape)} "
                f"loaded={tuple(loaded_weight.shape)}."
            )
        param.data.copy_(loaded_weight)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float)
        routing_weights, selected_experts = torch.topk(router_probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            routing_weights = routing_weights / routing_weights.sum(dim=-1, keepdim=True)
        routing_weights = routing_weights.to(router_logits.dtype)
        return router_logits, routing_weights, selected_experts


class Qwen3MoeExperts(nn.Module):
    """Qwen3-MoE fused expert weights.

    This first-stage module supports the official fused expert layout:
    gate_up_proj[num_experts, 2 * moe_intermediate_size, hidden_size] and
    down_proj[num_experts, hidden_size, moe_intermediate_size].
    """

    def __init__(self, config) -> None:
        super().__init__()
        self.num_experts = _get_num_experts(config)
        self.hidden_dim = int(config.hidden_size)
        self.intermediate_dim = int(config.moe_intermediate_size)
        self.ep_size = get_ep_size()
        self.ep_rank = get_ep_rank()
        self.expert_parallel_backend = str(
            getattr(config, "expert_parallel_backend", "torch") or "torch"
        ).strip().lower()
        if self.num_experts % self.ep_size != 0:
            raise ValueError(
                "Qwen3-MoE EP v1 requires num_experts divisible by expert_parallel_size, "
                f"got num_experts={self.num_experts} expert_parallel_size={self.ep_size}."
            )
        self.num_local_experts = self.num_experts // self.ep_size
        self.local_expert_start = self.ep_rank * self.num_local_experts
        self.local_expert_end = self.local_expert_start + self.num_local_experts
        self.global_expert_ids = list(range(self.local_expert_start, self.local_expert_end))
        self._deepep_debug_call_id = 0
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_local_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_local_experts, self.hidden_dim, self.intermediate_dim)
        )
        self.gate_up_proj.weight_loader = self.weight_loader
        self.down_proj.weight_loader = self.weight_loader
        hidden_act = str(getattr(config, "hidden_act", "silu"))
        if hidden_act != "silu":
            raise NotImplementedError(f"Unsupported Qwen3-MoE hidden_act={hidden_act!r}.")

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor) -> None:
        if param.shape != loaded_weight.shape:
            raise ValueError(
                f"Qwen3-MoE fused expert weight shape mismatch: expected={tuple(param.shape)} "
                f"loaded={tuple(loaded_weight.shape)}."
            )
        param.data.copy_(loaded_weight)

    def _use_grouped_mm(self, hidden_states: torch.Tensor) -> bool:
        if not hasattr(torch, "_grouped_mm"):
            return False
        if not hidden_states.is_cuda:
            return False
        if hidden_states.dtype not in (torch.float16, torch.bfloat16):
            return False
        # torch._grouped_mm requires 16-byte-aligned matrix strides on CUDA.
        element_size = int(torch.tensor([], dtype=hidden_states.dtype).element_size())
        return (
            (self.hidden_dim * element_size) % 16 == 0
            and (self.intermediate_dim * element_size) % 16 == 0
        )

    def _apply_local_experts_reference(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            valid_expert_mask = (selected_experts >= 0) & (selected_experts < self.num_experts)
            expert_mask = F.one_hot(
                torch.clamp(selected_experts, min=0, max=self.num_experts - 1),
                num_classes=self.num_experts,
            )
            expert_mask = expert_mask * valid_expert_mask.unsqueeze(-1)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hits = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx_tensor in expert_hits:
            expert_idx = int(expert_idx_tensor.item())
            if expert_idx < self.local_expert_start or expert_idx >= self.local_expert_end:
                continue
            local_expert_idx = expert_idx - self.local_expert_start
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[local_expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[local_expert_idx])
            current_hidden_states = current_hidden_states * routing_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))
        return final_hidden_states

    def _apply_local_experts_grouped_mm(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        num_tokens = int(hidden_states.shape[0])
        if num_tokens == 0:
            return final_hidden_states

        top_k = int(selected_experts.shape[-1])
        flat_experts = selected_experts.reshape(-1).to(torch.long)
        flat_weights = routing_weights.reshape(-1)
        flat_token_idx = torch.arange(
            num_tokens,
            device=hidden_states.device,
            dtype=torch.long,
        ).repeat_interleave(top_k)

        valid = (flat_experts >= self.local_expert_start) & (flat_experts < self.local_expert_end)
        local_ids = flat_experts - int(self.local_expert_start)
        local_ids = torch.where(valid, local_ids, torch.zeros_like(local_ids))
        flat_weights = torch.where(valid, flat_weights, torch.zeros_like(flat_weights))

        order = torch.argsort(local_ids)
        sorted_local_ids = local_ids[order]
        sorted_token_idx = flat_token_idx[order]
        sorted_weights = flat_weights[order]
        sorted_hidden = hidden_states.index_select(0, sorted_token_idx)

        counts = torch.bincount(
            sorted_local_ids,
            minlength=int(self.num_local_experts),
        )[: int(self.num_local_experts)]
        offsets = torch.cumsum(counts.to(torch.int32), dim=0).to(torch.int32)

        gate_up = torch._grouped_mm(  # type: ignore[attr-defined]
            sorted_hidden,
            self.gate_up_proj.transpose(1, 2),
            offsets,
        )
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        current_hidden_states = torch._grouped_mm(  # type: ignore[attr-defined]
            activated,
            self.down_proj.transpose(1, 2),
            offsets,
        )
        current_hidden_states = current_hidden_states * sorted_weights[:, None]
        final_hidden_states.index_add_(
            0,
            sorted_token_idx,
            current_hidden_states.to(final_hidden_states.dtype),
        )
        return final_hidden_states

    def _apply_local_experts(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self._use_grouped_mm(hidden_states):
            return self._apply_local_experts_grouped_mm(
                hidden_states,
                selected_experts,
                routing_weights,
            )
        return self._apply_local_experts_reference(hidden_states, selected_experts, routing_weights)

    def _apply_deepep_v2(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.ep_size <= 1:
            return self._apply_local_experts(hidden_states, selected_experts, routing_weights)
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Qwen3-MoE DeepEP v2 backend requires initialized torch.distributed.")

        from sparsevllm.engine.expert_parallel import deepep_v2

        debug_deepep = str(os.environ.get("SPARSEVLLM_DEEPEP_DEBUG", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        context = get_context()
        layer_idx = int(getattr(context, "now_layer_idx", -1))
        previous_event = getattr(context, "deepep_v2_previous_event", None)
        allocate_on_comm_stream = bool(getattr(context, "deepep_v2_allocate_on_comm_stream", False))
        disable_cpu_sync = bool(getattr(context, "deepep_v2_disable_cpu_sync", False))
        pre_dispatch_barrier = bool(getattr(context, "deepep_v2_pre_dispatch_barrier", False))
        debug_call_id = -1
        if debug_deepep and layer_idx == 0:
            debug_call_id = self._deepep_debug_call_id
            self._deepep_debug_call_id += 1
            logger.info(
                "DeepEP debug layer={} call={} before dispatch hidden={} topk={} weights={} previous_event={} disable_cpu_sync={} barrier={}",
                layer_idx,
                debug_call_id,
                tuple(hidden_states.shape),
                tuple(selected_experts.shape),
                tuple(routing_weights.shape),
                previous_event is not None,
                disable_cpu_sync,
                pre_dispatch_barrier,
            )
        if pre_dispatch_barrier:
            ep_group = get_ep_group()
            if ep_group is None:
                raise RuntimeError("DeepEP v2 pre-dispatch barrier requires an EP process group.")
            dist.barrier(group=ep_group)
        dispatched = deepep_v2.dispatch(
            hidden_states=hidden_states,
            selected_experts=selected_experts,
            routing_weights=routing_weights,
            num_experts=self.num_experts,
            previous_event=previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
            disable_cpu_sync=disable_cpu_sync,
        )
        num_recv_tokens = int(dispatched.num_recv_tokens)
        if debug_deepep and layer_idx == 0:
            logger.info(
                "DeepEP debug layer={} call={} after dispatch recv_x={} num_recv={}",
                layer_idx,
                debug_call_id,
                tuple(dispatched.recv_x.shape),
                num_recv_tokens,
            )
        recv_selected_experts = dispatched.recv_topk_idx[:num_recv_tokens] + self.local_expert_start
        recv_selected_experts = torch.where(
            dispatched.recv_topk_idx[:num_recv_tokens] >= 0,
            recv_selected_experts,
            torch.full_like(recv_selected_experts, -1),
        )
        if dispatched.recv_topk_weights is None:
            raise RuntimeError("DeepEP v2 dispatch did not return routing weights.")
        local_output = torch.zeros_like(dispatched.recv_x)
        if num_recv_tokens > 0:
            if dispatched.valid_recv_tokens is not None:
                valid_rows = torch.arange(
                    num_recv_tokens,
                    device=dispatched.recv_topk_idx.device,
                    dtype=dispatched.valid_recv_tokens.dtype,
                ) < dispatched.valid_recv_tokens
                recv_selected_experts = torch.where(
                    valid_rows[:, None],
                    recv_selected_experts,
                    torch.full_like(recv_selected_experts, -1),
                )
            if debug_deepep and layer_idx == 0:
                logger.info(
                    "DeepEP debug layer={} call={} before local experts num_recv={}",
                    layer_idx,
                    debug_call_id,
                    num_recv_tokens,
                )
            local_output[:num_recv_tokens] = self._apply_local_experts(
                dispatched.recv_x[:num_recv_tokens],
                recv_selected_experts,
                dispatched.recv_topk_weights[:num_recv_tokens].to(dispatched.recv_x.dtype),
            )
        combine_previous_event = deepep_v2.capture_event() if allocate_on_comm_stream else None
        if debug_deepep and layer_idx == 0:
            logger.info(
                "DeepEP debug layer={} call={} before combine local_output={} previous_event={}",
                layer_idx,
                debug_call_id,
                tuple(local_output.shape),
                combine_previous_event is not None,
            )
        combined = deepep_v2.combine(
            local_output,
            dispatched.handle,
            previous_event=combine_previous_event,
            allocate_on_comm_stream=allocate_on_comm_stream,
        )
        if debug_deepep and layer_idx == 0:
            logger.info(
                "DeepEP debug layer={} call={} after combine output={}",
                layer_idx,
                debug_call_id,
                tuple(combined.shape),
            )
        return combined

    def forward(
        self,
        hidden_states: torch.Tensor,
        selected_experts: torch.Tensor,
        routing_weights: torch.Tensor,
    ) -> torch.Tensor:
        if self.expert_parallel_backend == "deepep_v2":
            return self._apply_deepep_v2(hidden_states, selected_experts, routing_weights)
        if self.ep_size > 1:
            raise RuntimeError(
                "Qwen3-MoE torch expert backend does not support expert_parallel_size > 1. "
                "Use expert_parallel_backend='deepep_v2' for EP."
            )
        return self._apply_local_experts(hidden_states, selected_experts, routing_weights)


class Qwen3MoeSparseMoeBlock(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.gate = Qwen3MoeTopKRouter(config)
        self.experts = Qwen3MoeExperts(config)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        original_shape = hidden_states.shape
        if hidden_states.dim() == 2:
            hidden_dim = hidden_states.shape[-1]
        elif hidden_states.dim() == 3:
            hidden_dim = hidden_states.shape[-1]
        else:
            raise ValueError(
                "Qwen3-MoE hidden_states must be rank 2 [tokens, hidden] or "
                f"rank 3 [batch, seq, hidden], got shape={tuple(hidden_states.shape)}."
            )
        hidden_states_flat = hidden_states.reshape(-1, hidden_dim)
        _, routing_weights, selected_experts = self.gate(hidden_states_flat)
        final_hidden_states = self.experts(hidden_states_flat, selected_experts, routing_weights)
        return final_hidden_states.reshape(original_shape)


def _is_moe_layer(config, layer_idx: int) -> bool:
    mlp_only_layers = set(int(layer) for layer in getattr(config, "mlp_only_layers", []) or [])
    decoder_sparse_step = int(getattr(config, "decoder_sparse_step", 1) or 1)
    if decoder_sparse_step <= 0:
        raise ValueError(f"decoder_sparse_step must be > 0, got {decoder_sparse_step}.")
    return (
        int(layer_idx) not in mlp_only_layers
        and _get_num_experts(config) > 0
        and (int(layer_idx) + 1) % decoder_sparse_step == 0
    )


class Qwen3MoeDecoderLayer(nn.Module):
    def __init__(self, config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", config.hidden_size // config.num_attention_heads),
            rope_theta=_get_rope_theta(config),
            rope_scaling=_get_rope_scaling(config),
            proj_chunk_size=getattr(config, "mlp_chunk_size", 16384),
        )
        if _is_moe_layer(config, layer_idx):
            self.mlp = Qwen3MoeSparseMoeBlock(config)
        else:
            self.mlp = Qwen3MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                mlp_chunk_size=getattr(config, "mlp_chunk_size", 16384),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

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
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen3MoeModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3MoeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
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
            debug_layers = {int(part) for part in debug_layers_env.split(",") if part.strip()}
            self.debug_last_hidden_states = {-1: hidden_states[-1:].detach().clone()}

        for i, layer in enumerate(self.layers):
            context.now_layer_idx = i
            hidden_states, residual = layer(positions, hidden_states, residual)
            if self.sparse_controller is not None:
                hidden_states, residual = self.sparse_controller.apply_activation_hook(
                    i,
                    hidden_states,
                    residual,
                    context,
                )
            if debug_layers is not None and i in debug_layers:
                layer_output = hidden_states if residual is None else hidden_states + residual
                self.debug_last_hidden_states[int(i)] = layer_output[-1:].detach().clone()

            if self.sparse_controller is not None:
                self.sparse_controller.on_layer_end(i, context)

        hidden_states, _ = self.norm(hidden_states, residual)
        if debug_layers is not None:
            self.debug_last_hidden_states[self.config.num_hidden_layers] = hidden_states[-1:].detach().clone()
        return hidden_states

    def decode_dense_before_moe(
        self,
        layer_idx: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.layers[int(layer_idx)]
        context = get_context()
        context.now_layer_idx = int(layer_idx)
        if residual is None:
            hidden_states, residual = layer.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = layer.input_layernorm(hidden_states, residual)
        hidden_states = layer.self_attn(positions, hidden_states)
        hidden_states, residual = layer.post_attention_layernorm(hidden_states, residual)
        return hidden_states, residual

    def decode_final_hidden(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen3MoePiecewiseDecodeCudaGraphState:
    """CUDA graph dense decode segments while keeping MoE execution eager."""

    def __init__(
        self,
        owner: "Qwen3MoeForCausalLM",
        graph_state,
        *,
        graph_pool,
    ) -> None:
        self.owner = owner
        self.graph_key = graph_state.key
        self.graph_pool = graph_pool
        self.captured = False
        self.layer_graphs: list[torch.cuda.CUDAGraph] = []
        self.final_graph: torch.cuda.CUDAGraph | None = None
        self.layer_input_hidden: list[torch.Tensor | None] = []
        self.moe_input_hidden: list[torch.Tensor] = []
        self.dense_hidden: list[torch.Tensor] = []
        self.dense_residual: list[torch.Tensor] = []
        self.final_hidden_input: torch.Tensor | None = None
        self.logits: torch.Tensor | None = None
        self.keepalive: list[object] = []
        self._logged_replay_moe = False

    def clear(self) -> None:
        self.layer_graphs.clear()
        self.final_graph = None
        self.layer_input_hidden.clear()
        self.moe_input_hidden.clear()
        self.dense_hidden.clear()
        self.dense_residual.clear()
        self.final_hidden_input = None
        self.logits = None
        self.keepalive.clear()
        self._logged_replay_moe = False
        self.captured = False

    @property
    def batch_size(self) -> int:
        return int(self.graph_key.batch_size)

    @property
    def num_layers(self) -> int:
        return len(self.owner.model.layers)

    def _allocate_inputs(self, input_ids: torch.Tensor) -> None:
        dtype = self.owner.model.embed_tokens.weight.dtype
        device = input_ids.device
        hidden_size = int(self.owner.config.hidden_size)
        self.layer_input_hidden = [None]
        self.moe_input_hidden = []
        self.dense_hidden = []
        self.dense_residual = []
        for _ in range(1, self.num_layers):
            self.layer_input_hidden.append(
                torch.empty((self.batch_size, hidden_size), dtype=dtype, device=device)
            )
        for _ in range(self.num_layers):
            self.moe_input_hidden.append(
                torch.empty((self.batch_size, hidden_size), dtype=dtype, device=device)
            )
            self.dense_hidden.append(
                torch.empty((self.batch_size, hidden_size), dtype=dtype, device=device)
            )
            self.dense_residual.append(
                torch.empty((self.batch_size, hidden_size), dtype=dtype, device=device)
            )
        self.final_hidden_input = torch.empty((self.batch_size, hidden_size), dtype=dtype, device=device)

    def _uses_deepep_v2(self) -> bool:
        if get_ep_size() <= 1 or not self.owner.model.layers:
            return False
        experts = getattr(getattr(self.owner.model.layers[0], "mlp", None), "experts", None)
        return str(getattr(experts, "expert_parallel_backend", "")).strip().lower() == "deepep_v2"

    def _barrier_ep_ranks(self) -> None:
        if get_ep_size() <= 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Qwen3-MoE piecewise decode graph alignment requires initialized distributed ranks.")
        if torch.cuda.is_available():
            dist.barrier(group=get_ep_group(), device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier(group=get_ep_group())

    def _run_mlp_and_hooks(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        sparse_controller,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        layer = self.owner.model.layers[int(layer_idx)]
        context = get_context()
        context.now_layer_idx = int(layer_idx)
        hidden_states = layer.mlp(hidden_states)
        if sparse_controller is not None:
            hidden_states, residual = sparse_controller.apply_activation_hook(
                int(layer_idx),
                hidden_states,
                residual,
                context,
            )
            sparse_controller.on_layer_end(int(layer_idx), context)
        return hidden_states, residual

    def _sync_ep_ranks_before_capture_moe(self, layer_idx: int) -> None:
        if get_ep_size() <= 1:
            return
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("Qwen3-MoE piecewise decode graph capture requires initialized distributed ranks.")
        if torch.cuda.is_available():
            dist.barrier(group=get_ep_group(), device_ids=[torch.cuda.current_device()])
        else:
            dist.barrier(group=get_ep_group())
        if int(layer_idx) == 0:
            logger.info("Synchronized EP ranks before piecewise decode graph MoE capture.")

    def _run_warmup(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        sparse_controller,
    ) -> None:
        context = get_context()
        context.sparse_controller = sparse_controller
        use_graph_compatible_deepep = self._uses_deepep_v2()
        previous_pre_dispatch_barrier = getattr(context, "deepep_v2_pre_dispatch_barrier", None)
        if use_graph_compatible_deepep:
            context.deepep_v2_pre_dispatch_barrier = True
        if sparse_controller is not None:
            sparse_controller.prepare_forward(context.seqs, is_prefill=False)
        try:
            hidden_states = self.owner.model.embed_tokens(input_ids)
            residual = None
            for layer_idx in range(self.num_layers):
                if layer_idx > 0:
                    layer_input = self.layer_input_hidden[layer_idx]
                    if layer_input is None:
                        raise RuntimeError(f"Missing Qwen3-MoE piecewise input buffer for layer={layer_idx}.")
                    layer_input.copy_(hidden_states)
                hidden_states, residual = self.owner.model.decode_dense_before_moe(
                    layer_idx,
                    positions,
                    hidden_states,
                    residual,
                )
                self.dense_hidden[layer_idx].copy_(hidden_states)
                self.dense_residual[layer_idx].copy_(residual)
                hidden_states, residual = self._run_mlp_and_hooks(
                    layer_idx,
                    hidden_states,
                    residual,
                    sparse_controller,
                )
            if self.final_hidden_input is None:
                raise RuntimeError("Missing Qwen3-MoE piecewise final input buffer during warmup.")
            self.final_hidden_input.copy_(hidden_states)
            hidden_states = self.owner.model.decode_final_hidden(hidden_states, residual)
            _ = self.owner.compute_logits(hidden_states)
        finally:
            if use_graph_compatible_deepep:
                if previous_pre_dispatch_barrier is None:
                    context.deepep_v2_pre_dispatch_barrier = False
                else:
                    context.deepep_v2_pre_dispatch_barrier = previous_pre_dispatch_barrier

    def _capture_layer(
        self,
        layer_idx: int,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.cuda.CUDAGraph:
        graph = torch.cuda.CUDAGraph()
        context = get_context()
        context.now_layer_idx = int(layer_idx)
        dense_hidden_buffer = self.dense_hidden[int(layer_idx)]
        dense_residual_buffer = self.dense_residual[int(layer_idx)]
        try:
            with torch.cuda.graph(graph, pool=self.graph_pool):
                if int(layer_idx) == 0:
                    hidden_states = self.owner.model.embed_tokens(input_ids)
                    residual = None
                else:
                    hidden_states = self.layer_input_hidden[int(layer_idx)]
                    residual = self.dense_residual[int(layer_idx) - 1]
                dense_hidden, dense_residual = self.owner.model.decode_dense_before_moe(
                    layer_idx,
                    positions,
                    hidden_states,
                    residual,
                )
                dense_hidden_buffer.copy_(dense_hidden)
                dense_residual_buffer.copy_(dense_residual)
        except Exception as exc:
            raise RuntimeError(
                f"qwen3_moe piecewise decode_cuda_graph capture failed at layer={layer_idx}: {exc!r}"
            ) from exc
        return graph

    def _capture_final(self) -> tuple[torch.cuda.CUDAGraph, torch.Tensor]:
        if self.final_hidden_input is None:
            raise RuntimeError("Qwen3-MoE piecewise graph final input buffer is not allocated.")
        if not self.dense_residual:
            raise RuntimeError("Qwen3-MoE piecewise graph final residual buffer is not allocated.")
        final_residual = self.dense_residual[-1]
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph, pool=self.graph_pool):
                hidden_states = self.owner.model.decode_final_hidden(
                    self.final_hidden_input,
                    final_residual,
                )
                logits = self.owner.compute_logits(hidden_states)
        except Exception as exc:
            raise RuntimeError(f"qwen3_moe piecewise final decode_cuda_graph capture failed: {exc!r}") from exc
        return graph, logits

    def capture(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        sparse_controller,
    ) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Qwen3-MoE piecewise decode_cuda_graph requires CUDA.")
        self._allocate_inputs(input_ids)
        with profiler.record("decode_cuda_graph_piecewise_warmup"):
            self._run_warmup(input_ids, positions, sparse_controller)
        torch.cuda.synchronize(input_ids.device)

        context = get_context()
        context.sparse_controller = sparse_controller
        if sparse_controller is not None:
            sparse_controller.prepare_forward(context.seqs, is_prefill=False)
        with profiler.record("decode_cuda_graph_piecewise_capture"):
            logger.info(
                "Capturing Qwen3-MoE piecewise decode CUDA graphs: layers={} batch={} context_cap={}",
                self.num_layers,
                self.batch_size,
                self.graph_key.context_capacity,
            )
            for layer_idx in range(self.num_layers):
                graph = self._capture_layer(layer_idx, input_ids, positions)
                self.layer_graphs.append(graph)
                if (layer_idx + 1) % 8 == 0 or layer_idx + 1 == self.num_layers:
                    logger.info(
                        "Captured Qwen3-MoE piecewise decode CUDA graph layer {}/{}.",
                        layer_idx + 1,
                        self.num_layers,
                    )
            self.final_graph, self.logits = self._capture_final()
            logger.info("Captured Qwen3-MoE piecewise decode CUDA graphs without eager MoE in capture.")

        self.keepalive = [
            self.layer_input_hidden,
            self.moe_input_hidden,
            self.dense_hidden,
            self.dense_residual,
            self.final_hidden_input,
            self.logits,
            *self.layer_graphs,
            self.final_graph,
        ]
        self.captured = True

    def replay(
        self,
        sparse_controller,
    ) -> torch.Tensor | None:
        if not self.captured or self.final_graph is None:
            raise RuntimeError("Qwen3-MoE piecewise decode_cuda_graph was not captured.")
        if self.final_hidden_input is None:
            raise RuntimeError("Qwen3-MoE piecewise decode_cuda_graph final buffer is missing.")

        context = get_context()
        context.sparse_controller = sparse_controller
        with profiler.record("decode_cuda_graph_piecewise_replay"):
            for layer_idx, graph in enumerate(self.layer_graphs):
                context.now_layer_idx = int(layer_idx)
                graph.replay()
                moe_input = self.moe_input_hidden[layer_idx]
                moe_input.copy_(self.dense_hidden[layer_idx])
                layer = self.owner.model.layers[int(layer_idx)]
                experts = getattr(getattr(layer, "mlp", None), "experts", None)
                use_deepep_event = (
                    experts is not None
                    and str(getattr(experts, "expert_parallel_backend", "")).strip().lower() == "deepep_v2"
                    and get_ep_size() > 1
                )
                if use_deepep_event:
                    torch.cuda.current_stream(moe_input.device).synchronize()
                    context.deepep_v2_previous_event = None
                    context.deepep_v2_allocate_on_comm_stream = False
                    context.deepep_v2_pre_dispatch_barrier = True
                if layer_idx == 0 and not self._logged_replay_moe:
                    logger.info("Running eager Qwen3-MoE block after first piecewise graph replay.")
                try:
                    hidden_states, _ = self._run_mlp_and_hooks(
                        layer_idx,
                        moe_input,
                        self.dense_residual[layer_idx],
                        sparse_controller,
                    )
                finally:
                    if use_deepep_event:
                        context.deepep_v2_previous_event = None
                        context.deepep_v2_allocate_on_comm_stream = False
                        context.deepep_v2_pre_dispatch_barrier = False
                if layer_idx == 0 and not self._logged_replay_moe:
                    logger.info("Finished eager Qwen3-MoE block after first piecewise graph replay.")
                    self._logged_replay_moe = True
                if layer_idx + 1 < self.num_layers:
                    next_hidden = self.layer_input_hidden[layer_idx + 1]
                    if next_hidden is None:
                        raise RuntimeError(f"Missing Qwen3-MoE piecewise input buffer for layer={layer_idx + 1}.")
                    next_hidden.copy_(hidden_states)
                else:
                    self.final_hidden_input.copy_(hidden_states)
            self.final_graph.replay()
        return self.logits

    def run(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        sparse_controller,
        force_capture_alignment: bool = False,
    ) -> torch.Tensor | None:
        needs_capture = not self.captured
        if needs_capture:
            self.capture(input_ids, positions, sparse_controller)
        elif bool(force_capture_alignment) and self._uses_deepep_v2():
            with profiler.record("decode_cuda_graph_piecewise_capture_align_warmup"):
                self._run_warmup(input_ids, positions, sparse_controller)

        if bool(force_capture_alignment) and self._uses_deepep_v2():
            with profiler.record("decode_cuda_graph_piecewise_capture_align_barrier"):
                self._barrier_ep_ranks()

        return self.replay(sparse_controller)


class Qwen3MoeForCausalLM(nn.Module):
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3MoeModel(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)
        if getattr(config, "tie_word_embeddings", False):
            self.lm_head.weight.data = self.model.embed_tokens.weight.data
        self._local_expert_load_state: set[tuple[int, int, str]] = set()

    def _local_expert_range(self) -> tuple[int, int]:
        ep_size = get_ep_size()
        ep_rank = get_ep_rank()
        num_experts = _get_num_experts(self.config)
        if num_experts % ep_size != 0:
            raise ValueError(
                "Qwen3-MoE expert tensor loading requires num_experts divisible by "
                f"expert_parallel_size, got {num_experts} and {ep_size}."
            )
        local_num_experts = num_experts // ep_size
        start = ep_rank * local_num_experts
        return start, start + local_num_experts

    def _mark_local_expert_loaded(self, layer_idx: int, local_expert_idx: int, part: str) -> None:
        self._local_expert_load_state.add((int(layer_idx), int(local_expert_idx), str(part)))

    def _mark_fused_expert_loaded(self, layer_idx: int, part: str) -> None:
        local_num_experts = _get_num_experts(self.config) // get_ep_size()
        parts = ("gate_proj", "up_proj") if part == "gate_up_proj" else ("down_proj",)
        for local_expert_idx in range(local_num_experts):
            for named_part in parts:
                self._mark_local_expert_loaded(layer_idx, local_expert_idx, named_part)

    def _load_fused_expert_weight(self, source_weight_name: str, safe_file) -> str | None:
        match = _FUSED_EXPERT_RE.match(source_weight_name)
        if match is None:
            return None

        param = self.get_parameter(source_weight_name)
        num_experts = _get_num_experts(self.config)
        start, end = self._local_expert_range()
        local_num_experts = end - start

        if get_ep_size() == 1:
            loaded_weight = safe_file.get_tensor(source_weight_name)
        else:
            weight_slice = safe_file.get_slice(source_weight_name)
            source_shape = tuple(weight_slice.get_shape())
            if source_shape[0] == num_experts:
                if len(source_shape) == 3:
                    loaded_weight = weight_slice[start:end, :, :]
                else:
                    raise ValueError(
                        "Qwen3-MoE fused expert tensor must be rank 3, "
                        f"got key={source_weight_name!r} shape={source_shape}."
                    )
            elif source_shape[0] == local_num_experts:
                loaded_weight = safe_file.get_tensor(source_weight_name)
            else:
                raise ValueError(
                    "Qwen3-MoE fused expert tensor first dimension does not match global or local "
                    f"expert count: key={source_weight_name!r} shape={source_shape} "
                    f"num_experts={num_experts} local_num_experts={local_num_experts}."
                )
        weight_loader = getattr(param, "weight_loader")
        weight_loader(param, loaded_weight)
        self._mark_fused_expert_loaded(int(match.group("layer_idx")), match.group("part"))
        return "loaded"

    def _load_named_expert_weight(self, source_weight_name: str, safe_file) -> str | None:
        match = _NAMED_EXPERT_RE.match(source_weight_name)
        if match is None:
            return None

        layer_idx = int(match.group("layer_idx"))
        expert_idx = int(match.group("expert_idx"))
        part = match.group("part")
        start, end = self._local_expert_range()
        if expert_idx < start or expert_idx >= end:
            return "skipped"
        local_expert_idx = expert_idx - start
        layer = self.model.layers[layer_idx]
        if not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
            raise ValueError(
                "Qwen3-MoE checkpoint contains an expert tensor for a dense MLP layer: "
                f"key={source_weight_name!r} layer_idx={layer_idx}."
            )

        loaded_weight = safe_file.get_tensor(source_weight_name)
        if part == "down_proj":
            param = layer.mlp.experts.down_proj
            target = param.data[local_expert_idx]
        else:
            param = layer.mlp.experts.gate_up_proj
            split = int(self.config.moe_intermediate_size)
            if part == "gate_proj":
                target = param.data[local_expert_idx, :split, :]
            elif part == "up_proj":
                target = param.data[local_expert_idx, split:, :]
            else:
                raise AssertionError(f"Unhandled Qwen3-MoE expert part: {part}")

        if target.shape != loaded_weight.shape:
            raise ValueError(
                "Qwen3-MoE named expert weight shape mismatch: "
                f"key={source_weight_name!r} expected={tuple(target.shape)} "
                f"loaded={tuple(loaded_weight.shape)}."
            )
        target.copy_(loaded_weight)
        self._mark_local_expert_loaded(layer_idx, local_expert_idx, part)
        return "loaded"

    def load_weight_from_safetensors(self, source_weight_name: str, safe_file) -> str | None:
        if ".mlp.experts." not in source_weight_name:
            return None
        load_result = self._load_fused_expert_weight(source_weight_name, safe_file)
        if load_result is not None:
            return load_result
        load_result = self._load_named_expert_weight(source_weight_name, safe_file)
        if load_result is not None:
            return load_result
        raise NotImplementedError(
            "Qwen3-MoE EP v1 supports fused expert tensors or named expert tensors only. "
            f"Unsupported expert checkpoint key: {source_weight_name!r}."
        )

    def finalize_weight_loading(self) -> None:
        local_num_experts = _get_num_experts(self.config) // get_ep_size()
        required_parts = ("gate_proj", "up_proj", "down_proj")
        missing: list[str] = []
        for layer_idx, layer in enumerate(self.model.layers):
            if not isinstance(layer.mlp, Qwen3MoeSparseMoeBlock):
                continue
            for local_expert_idx in range(local_num_experts):
                for part in required_parts:
                    if (layer_idx, local_expert_idx, part) not in self._local_expert_load_state:
                        missing.append(
                            f"layer={layer_idx} local_expert={local_expert_idx} part={part}"
                        )
        if missing:
            preview = "; ".join(missing[:8])
            suffix = "" if len(missing) <= 8 else f"; ... ({len(missing)} missing total)"
            raise RuntimeError(
                "Incomplete Qwen3-MoE expert checkpoint load for this EP rank: "
                f"{preview}{suffix}."
            )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def run_decode_piecewise_cuda_graph(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        graph_state,
        *,
        graph_pool,
        sparse_controller,
        needs_capture: bool = False,
        force_capture_alignment: bool = False,
    ) -> torch.Tensor | None:
        del needs_capture
        piecewise_state = graph_state.piecewise_state
        if piecewise_state is None:
            piecewise_state = Qwen3MoePiecewiseDecodeCudaGraphState(
                self,
                graph_state,
                graph_pool=graph_pool,
            )
            graph_state.piecewise_state = piecewise_state
        if not isinstance(piecewise_state, Qwen3MoePiecewiseDecodeCudaGraphState):
            raise TypeError(
                "Qwen3-MoE decode graph state was created by an incompatible piecewise runner: "
                f"{type(piecewise_state).__name__}."
            )
        return piecewise_state.run(
            input_ids,
            positions,
            sparse_controller,
            force_capture_alignment=bool(force_capture_alignment),
        )
