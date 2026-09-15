"""Routed MXFP4 experts with UE8M0 activations and pre-down routing weights."""

from dataclasses import dataclass
import math

import torch

from sparsevllm.kernels.moe import MoeAlignment
from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.operators.workspace import get_workspace_manager
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


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


MXFP4_MOE_REGISTRY = OpRegistry(
    "MXFP4 experts with UE8M0 activations and weighted clipped SwiGLU",
    portfolio=PortfolioPolicy(repo_nonstandard=("triton_mxfp4_ue8m0",)),
)


@MXFP4_MOE_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonMxfp4MoeProvider:
    name = "triton_mxfp4_ue8m0"
    weight_layout_id = "packed_mxfp4_gate_up_e8m0_k32"
    block_size = 16

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton or not caps.supports_bfloat16:
            return SupportResult.unsupported("requires CUDA, Triton and BF16 tensor cores")
        if not caps.supports_native_fp8:
            return SupportResult.unsupported("requires E4M3 activation conversion support")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        from sparsevllm.kernels.external.sgl.moe import sgl_moe_alignment_support
        supported, reason = sgl_moe_alignment_support()
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        assignments = spec.max_num_tokens * spec.top_k
        width = max(spec.hidden_size, spec.intermediate_size)
        padded = self._alignment_capacity(assignments)
        self._regions, offset = {}, 0
        for name, count, dtype in (
            ("ids", assignments, torch.int32),
            ("sorted", padded, torch.int32),
            ("experts", (padded + self.block_size - 1) // self.block_size, torch.int32),
            ("count", 1, torch.int32),
            ("cumsum", spec.num_local_experts + 2, torch.int32),
            ("quantized", assignments * width, torch.float8_e4m3fn),
            ("scales", assignments * (width // 128), torch.float32),
            ("projection", assignments * max(2 * spec.intermediate_size, spec.hidden_size), torch.bfloat16),
            ("activated", assignments * spec.intermediate_size, torch.bfloat16),
        ):
            offset = (offset + 255) // 256 * 256
            size = count * dtype.itemsize
            self._regions[name] = offset, size, dtype
            offset += size
        self._lease = get_workspace_manager(device, create=True).reserve_bytes(
            offset, label="mxfp4_moe", lane="mxfp4_moe",
        )

    def _alignment_capacity(self, assignments):
        return min(assignments * self.block_size,
                   assignments + (self.spec.num_local_experts + 1) * (self.block_size - 1))

    def _view(self, name, *shape):
        offset, size, dtype = self._regions[name]
        return self._lease.buffer[offset:offset + size].view(dtype)[:math.prod(shape)].view(shape)

    def allocate_weights(self):
        s = self.spec
        def allocate(n, k):
            return torch.empty((s.num_local_experts, n, k), device=self.device, dtype=torch.uint8)
        return Mxfp4ExpertWeights(
            allocate(2 * s.intermediate_size, s.hidden_size // 2),
            allocate(s.hidden_size, s.intermediate_size // 2),
            allocate(2 * s.intermediate_size, s.hidden_size // 32),
            allocate(s.hidden_size, s.intermediate_size // 32),
        )

    def load_projection(self, storage, local_expert, projection, weight, scale):
        if not 0 <= local_expert < self.spec.num_local_experts:
            raise ValueError("MXFP4 checkpoint expert index lies outside local storage.")
        if projection == "down":
            target, scale_target = storage.down[local_expert], storage.down_scale[local_expert]
        elif projection in ("gate", "up"):
            i = self.spec.intermediate_size
            offset = 0 if projection == "gate" else i
            target = storage.gate_up[local_expert, offset:offset+i]
            scale_target = storage.gate_up_scale[local_expert, offset:offset+i]
        else:
            raise ValueError(f"Unknown MXFP4 logical projection {projection!r}.")
        if weight.dtype not in (torch.int8, torch.uint8, torch.float4_e2m1fn_x2) or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise TypeError("MXFP4 checkpoint loading requires packed FP4 and E8M0 bytes.")
        if weight.shape != target.shape or scale.shape != scale_target.shape:
            raise ValueError("MXFP4 checkpoint projection does not match prepared storage.")
        target.copy_(weight.view(torch.uint8))
        scale_target.copy_(scale.view(torch.uint8))

    def run(self, x, ids, route_weights, storage):
        """Return FP32 local expert sums for EP reduction and shared-expert addition."""
        s = self.spec
        if x.ndim != 2 or x.shape[1] != s.hidden_size or x.dtype != torch.bfloat16:
            raise ValueError("MXFP4 MoE requires [tokens, hidden_size] BF16 input.")
        m = x.shape[0]
        if m > s.max_num_tokens:
            raise ValueError("MXFP4 MoE token count exceeds prepared workspace capacity.")
        if ids.shape != (m, s.top_k) or route_weights.shape != ids.shape:
            raise ValueError("MXFP4 MoE routing must match [tokens, top_k].")
        if ids.dtype not in (torch.int32, torch.int64) or route_weights.dtype != torch.float32:
            raise TypeError("MXFP4 MoE requires integer expert IDs and FP32 routing weights.")
        if any(t.device != self.device or not t.is_contiguous() for t in (x, ids, route_weights)):
            raise ValueError("MXFP4 MoE inputs must be contiguous on the prepared device.")
        if not m:
            return x.new_empty((0, s.hidden_size), dtype=torch.float32)
        from sparsevllm.kernels.external.sgl.moe import sgl_moe_align_block_size
        from sparsevllm.kernels.triton.fp8_ue8m0 import quantize_fp8_ue8m0
        from sparsevllm.kernels.triton.moe import localize_expert_ids, moe_sum
        from sparsevllm.kernels.triton.mxfp4 import mxfp4_gemm
        from sparsevllm.kernels.triton.silu_and_mul import weighted_clipped_swiglu
        assignments, local_end = m * s.top_k, s.local_expert_start + s.num_local_experts
        local_ids = localize_expert_ids(
            ids, local_expert_start=s.local_expert_start, local_expert_end=local_end,
            out=self._view("ids", m, s.top_k),
        )
        padded = self._alignment_capacity(assignments)
        alignment = MoeAlignment(
            self._view("sorted", padded),
            self._view("experts", (padded + self.block_size - 1) // self.block_size),
            self._view("count", 1), self.block_size, False,
        )
        sgl_moe_align_block_size(local_ids, block_size=self.block_size, num_experts=s.num_local_experts,
                                alignment=alignment, cumsum_buffer=self._view("cumsum", s.num_local_experts + 2))
        q, scales = self._view("quantized", m, s.hidden_size), self._view("scales", m, s.hidden_size // 128)
        quantize_fp8_ue8m0(x, q, scales)
        projected = self._view("projection", assignments, 2 * s.intermediate_size)
        mxfp4_gemm(q, scales, storage.gate_up, storage.gate_up_scale, projected,
                    alignment=alignment, input_top_k=s.top_k)
        activated = self._view("activated", assignments, s.intermediate_size)
        weighted_clipped_swiglu(projected, ids, route_weights, activated,
                                local_start=s.local_expert_start, local_end=local_end, limit=s.swiglu_limit)
        q = self._view("quantized", assignments, s.intermediate_size)
        scales = self._view("scales", assignments, s.intermediate_size // 128)
        quantize_fp8_ue8m0(activated, q, scales)
        down = self._view("projection", assignments, s.hidden_size)
        mxfp4_gemm(q, scales, storage.down, storage.down_scale, down, alignment=alignment)
        return moe_sum(down.view(m, s.top_k, s.hidden_size), ids, num_experts=s.num_experts,
                       local_expert_start=s.local_expert_start, local_expert_end=local_end,
                       output_dtype=torch.float32, filter_invalid_routes=True)


def resolve_mxfp4_moe_provider(spec, *, device_index):
    return OpResolver(MXFP4_MOE_REGISTRY).resolve(spec, current_platform.get_device_caps(device_index)).provider
