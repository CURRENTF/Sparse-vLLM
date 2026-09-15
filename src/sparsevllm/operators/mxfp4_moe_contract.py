"""Shared logical MXFP4 expert contract, independent of physical weight layouts."""

from dataclasses import dataclass
import math

import torch

@dataclass(frozen=True)
class Mxfp4MoeSpec:
    hidden_size: int
    intermediate_size: int
    num_experts: int
    num_local_experts: int
    local_expert_start: int
    top_k: int
    max_num_tokens: int
    swiglu_limit: float
    cuda_graph: bool = True

    def __post_init__(self):
        if min(self.hidden_size, self.intermediate_size, self.num_local_experts, self.max_num_tokens) <= 0:
            raise ValueError("MXFP4 MoE dimensions and capacity must be positive.")
        if self.hidden_size % 128 or self.intermediate_size % 128:
            raise ValueError("MXFP4 MoE feature dimensions must be aligned to 128.")
        if not 0 <= self.local_expert_start < self.local_expert_start + self.num_local_experts <= self.num_experts:
            raise ValueError("MXFP4 MoE local expert interval must lie within the global expert set.")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError("MXFP4 MoE top_k must lie within the global expert count.")
        if not math.isfinite(self.swiglu_limit) or self.swiglu_limit <= 0:
            raise ValueError("MXFP4 MoE requires a finite positive SwiGLU limit.")


@dataclass(frozen=True)
class Mxfp4ExpertWeights:
    gate_up: torch.Tensor
    down: torch.Tensor
    gate_up_scale: torch.Tensor
    down_scale: torch.Tensor


def _validate_inputs(spec, device, x, ids, route_weights):
    if x.ndim != 2 or x.shape[1] != spec.hidden_size or x.dtype != torch.bfloat16:
        raise ValueError("MXFP4 MoE requires [tokens, hidden_size] BF16 input.")
    m = x.shape[0]
    if m > spec.max_num_tokens:
        raise ValueError("MXFP4 MoE token count exceeds prepared workspace capacity.")
    if ids.shape != (m, spec.top_k) or route_weights.shape != ids.shape:
        raise ValueError("MXFP4 MoE routing must match [tokens, top_k].")
    if ids.dtype not in (torch.int32, torch.int64) or route_weights.dtype != torch.float32:
        raise TypeError("MXFP4 MoE requires integer expert IDs and FP32 routing weights.")
    if any(t.device != device or not t.is_contiguous() for t in (x, ids, route_weights)):
        raise ValueError("MXFP4 MoE inputs must be contiguous on the prepared device.")
    return m
