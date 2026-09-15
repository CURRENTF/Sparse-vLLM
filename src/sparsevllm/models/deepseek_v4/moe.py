"""Native hash/learned routing, MXFP4 experts and replicated FP8 shared experts."""

from dataclasses import fields

import torch
from torch import nn

from sparsevllm.distributed import get_parallel_context
from sparsevllm.distributed.moe_communication import prepare_moe_communication
from sparsevllm.layers.linear import MergedReplicatedLinear, ReplicatedLinear
from sparsevllm.operators.activation import resolve_silu_and_mul_provider
from sparsevllm.operators.float32_linear import Float32LinearSpec, prepare_float32_linear
from sparsevllm.operators.moe_router import MoeRouterOpSpec, resolve_moe_router_provider
from sparsevllm.operators.mxfp4_moe import (
    Mxfp4ExpertWeights, Mxfp4MoeSpec, resolve_mxfp4_moe_provider,
)


class DeepseekV4Router(nn.Module):
    def __init__(self, config, layer_id, *, max_num_tokens, cuda_graph):
        super().__init__()
        self.use_hash = layer_id < config.num_hash_layers
        self.scale = float(config.routed_scaling_factor)
        self.weight = nn.Parameter(torch.empty(
            config.n_routed_experts, config.hidden_size, dtype=torch.float32,
        ), requires_grad=False)
        if self.use_hash:
            self.register_buffer("tid2eid", torch.empty(
                config.vocab_size, config.num_experts_per_tok, dtype=torch.int32,
            ))
            self.register_parameter("bias", None)
        else:
            self.bias = nn.Parameter(torch.empty(config.n_routed_experts, dtype=torch.float32),
                                     requires_grad=False)
            self.register_buffer("tid2eid", None)
        self.spec = MoeRouterOpSpec(
            config.n_routed_experts, config.num_experts_per_tok, torch.float32,
            config.norm_topk_prob, cuda_graph,
            "hash_sqrt_softplus" if self.use_hash else "sqrt_softplus", max_num_tokens,
        )
        self.provider = resolve_moe_router_provider(self.spec, device_index=self.weight.device.index)
        self.projection = prepare_float32_linear(
            Float32LinearSpec(config.hidden_size, config.n_routed_experts, max_num_tokens,
                              cuda_graph=cuda_graph), device_index=self.weight.device.index,
        )

    def forward(self, x, metadata=None):
        logits = self.projection.run(x, self.weight)
        weights, ids = self.provider.run(
            self.spec, logits, self.bias, routed_scaling_factor=self.scale,
            hash_indices=self.tid2eid,
            input_ids=metadata.reshape(-1) if metadata is not None else None,
        )
        return ids, weights


class DeepseekV4Experts(nn.Module):
    def __init__(self, spec, *, device_index):
        super().__init__()
        self.spec = spec
        self.provider = resolve_mxfp4_moe_provider(spec, device_index=device_index)
        storage = self.provider.allocate_weights()
        for field in fields(storage):
            self.register_buffer(field.name, getattr(storage, field.name))
        self._loaded = set()

    @property
    def storage(self):
        return Mxfp4ExpertWeights(**{field.name: getattr(self, field.name) for field in fields(Mxfp4ExpertWeights)})

    def load_projection(self, expert_id, projection, weight, scale):
        local = expert_id - self.spec.local_expert_start
        if not 0 <= local < self.spec.num_local_experts:
            return False
        names = {"w1": "gate", "w3": "up", "w2": "down"}
        self.provider.load_projection(self.storage, local, names[projection], weight, scale)
        self._loaded.add((local, projection))
        return True

    def validate_loaded_weights(self):
        expected = {(expert, name) for expert in range(self.spec.num_local_experts)
                    for name in ("w1", "w3", "w2")}
        if missing := expected - self._loaded:
            raise RuntimeError(f"Missing local MXFP4 projections: {sorted(missing)}")

    def forward(self, x, ids, weights):
        return self.provider.run(x, ids, weights, self.storage)


class DeepseekV4SharedExperts(nn.Module):
    def __init__(self, config, quantization):
        super().__init__()
        intermediate = config.moe_intermediate_size * config.n_shared_experts
        self.gate_up = MergedReplicatedLinear(config.hidden_size, [intermediate, intermediate],
                                             quantization=quantization)
        self.down = ReplicatedLinear(intermediate, config.hidden_size, quantization=quantization)
        self.activation = resolve_silu_and_mul_provider(
            activation_dtype=torch.bfloat16, swiglu_limit=float(config.swiglu_limit),
        )

    def load_projection(self, projection, weight, scale):
        if projection == "w2":
            self.down.load_quantized_weight(weight, scale)
        else:
            self.gate_up.load_quantized_weight(weight, scale, {"w1": 0, "w3": 1}[projection])

    def forward(self, x):
        return self.down(self.activation(self.gate_up(x)))


class DeepseekV4Moe(nn.Module):
    def __init__(self, config, layer_id, *, quantization, mlp_chunk_size,
                 cuda_graph, parallel_collectives=None):
        super().__init__()
        parallel = get_parallel_context()
        if parallel.moe_tp_size != 1 or config.n_routed_experts % parallel.moe_ep_size:
            raise ValueError("Native MXFP4 experts require MoE TP=1 and evenly sharded EP.")
        if config.scoring_func != "sqrtsoftplus" or config.expert_dtype != "fp4":
            raise ValueError("DeepSeek V4 FFN requires sqrt-softplus routing and MXFP4 experts.")
        if (not quantization.enabled or quantization.scale_fmt != "ue8m0"
                or quantization.activation_dtype != "bfloat16"
                or quantization.max_num_tokens is None or not 0 < mlp_chunk_size <= quantization.max_num_tokens):
            raise ValueError("Native shared experts require BF16/UE8M0 FP8 and prepared chunk capacity.")
        self.chunk_size = mlp_chunk_size
        self.communication = prepare_moe_communication(parallel, parallel_collectives)
        self.gate = DeepseekV4Router(config, layer_id, max_num_tokens=mlp_chunk_size, cuda_graph=cuda_graph)
        local_experts = config.n_routed_experts // parallel.moe_ep_size
        self.experts = DeepseekV4Experts(Mxfp4MoeSpec(
            config.hidden_size, config.moe_intermediate_size, config.n_routed_experts,
            local_experts, parallel.moe_ep_rank * local_experts, config.num_experts_per_tok,
            mlp_chunk_size, float(config.swiglu_limit), cuda_graph,
        ), device_index=self.gate.weight.device.index)
        self.shared_experts = DeepseekV4SharedExperts(config, quantization)

    def forward(self, x, input_ids):
        if input_ids.shape != (len(x),):
            raise ValueError("FFN input IDs must align with local token rows.")
        routed = self.communication.run(
            x, route=self.gate, experts=self.experts, chunk_size=self.chunk_size,
            routing_metadata=input_ids[:, None] if self.gate.use_hash else None,
        )
        # Shared weights are replicated. Add once on each owner after all EP
        # and attention-TP reductions, retaining FP32 until the final cast.
        for offset in range(0, len(x), self.chunk_size):
            chunk = slice(offset, offset + self.chunk_size)
            routed[chunk].add_(self.shared_experts(x[chunk]))
        return routed.bfloat16()
