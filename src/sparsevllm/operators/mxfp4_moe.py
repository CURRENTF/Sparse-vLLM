"""Routed MXFP4 experts with clipped SwiGLU and FP32 local expert sums."""

import math

import torch

from sparsevllm.kernels.moe import MoeAlignment
from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.operators.workspace import get_workspace_manager
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


from sparsevllm.operators.mxfp4_moe_contract import Mxfp4MoeSpec, Mxfp4ExpertWeights, _validate_inputs
from sparsevllm.operators.mxfp4_marlin import VllmMarlinMxfp4MoeProvider


class FlashInferMxfp4MoeProvider:
    name = "flashinfer_cutlass_w4a16"
    weight_layout_id = "flashinfer_sm90_mxfp4_up_gate"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported("FlashInfer W4A16 MoE requires SM90")
        if not caps.supports_bfloat16 or (spec.cuda_graph and not caps.supports_graph_capture):
            return SupportResult.unsupported("requires BF16 and requested CUDA Graph support")
        if spec.num_experts % spec.num_local_experts or spec.local_expert_start % spec.num_local_experts:
            return SupportResult.unsupported("requires equal contiguous EP partitions")
        from sparsevllm.kernels.external.flashinfer.mxfp4_moe import mxfp4_moe_ops
        mxfp4_moe_ops()
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        from sparsevllm.kernels.external.flashinfer.mxfp4_moe import mxfp4_moe_ops
        self._run, workspace_size, self._pack, self._pack_scale = mxfp4_moe_ops()
        self.spec, self.device = spec, device
        self._options = dict(
            ep_size=spec.num_experts // spec.num_local_experts,
            ep_rank=spec.local_expert_start // spec.num_local_experts,
            use_w4_group_scaling=True, use_fused_finalize=False,
        )
        size = workspace_size(
            spec.max_num_tokens, spec.hidden_size, spec.intermediate_size,
            spec.num_experts, spec.top_k, x_dtype=torch.bfloat16,
            weight_dtype=torch.uint8, output_dtype=torch.bfloat16,
            device=device, **self._options,
        )
        self._lease = get_workspace_manager(device, create=True).reserve_bytes(
            size, label="flashinfer_mxfp4_moe", lane="mxfp4_moe",
        )
        self._alpha = torch.ones(spec.num_local_experts, device=device, dtype=torch.float32)
        self._beta = torch.zeros_like(self._alpha)
        self._limit = torch.full_like(self._alpha, spec.swiglu_limit)

    def allocate_weights(self):
        s = self.spec
        def weight(n, k):
            return torch.empty((s.num_local_experts, n, k // 2), device=self.device, dtype=torch.uint8)
        def scale(n, k):
            return torch.empty((s.num_local_experts, n // 64, k // 128, 16, 16),
                               device=self.device, dtype=torch.uint8)
        return Mxfp4ExpertWeights(
            weight(2 * s.intermediate_size, s.hidden_size),
            weight(s.hidden_size, s.intermediate_size),
            scale(2 * s.intermediate_size, s.hidden_size),
            scale(s.hidden_size, s.intermediate_size),
        )

    def load_projection(self, storage, local_expert, projection, weight, scale):
        s = self.spec
        if not 0 <= local_expert < s.num_local_experts:
            raise ValueError("MXFP4 checkpoint expert index lies outside local storage.")
        if projection == "down":
            n, k = s.hidden_size, s.intermediate_size
            target, scale_target = storage.down[local_expert], storage.down_scale[local_expert]
        elif projection in ("gate", "up"):
            n, k = s.intermediate_size, s.hidden_size
            # CUTLASS consumes [up, gate]; checkpoint projections remain logical.
            offset = n if projection == "gate" else 0
            target = storage.gate_up[local_expert, offset:offset + n]
            scale_target = storage.gate_up_scale[local_expert, offset // 64:(offset + n) // 64]
        else:
            raise ValueError(f"Unknown MXFP4 logical projection {projection!r}.")
        if weight.dtype not in (torch.int8, torch.uint8, torch.float4_e2m1fn_x2) or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise TypeError("MXFP4 checkpoint loading requires packed FP4 and E8M0 bytes.")
        if weight.shape != (n, k // 2) or scale.shape != (n, k // 32):
            raise ValueError("MXFP4 checkpoint projection does not match prepared storage.")
        target.copy_(self._pack(weight.view(torch.uint8).to(self.device).contiguous()[None], "fp4")[0])
        scale_target.copy_(self._pack_scale(scale.view(torch.uint8).to(self.device).contiguous()[None])[0])

    def run(self, x, ids, route_weights, storage):
        if not _validate_inputs(self.spec, self.device, x, ids, route_weights):
            return x.new_empty((0, self.spec.hidden_size), dtype=torch.float32)
        output = self._run(
            x, ids.to(torch.int32), route_weights, storage.gate_up, storage.down,
            torch.bfloat16, [storage.gate_up_scale.view(torch.int32), storage.down_scale.view(torch.int32)],
            swiglu_alpha=self._alpha, swiglu_beta=self._beta, swiglu_limit=self._limit,
            workspace_buffer=self._lease.buffer, **self._options,
        )
        # The public API returns its output in a one-element list.
        return output[0].float()


MXFP4_MOE_REGISTRY = OpRegistry(
    "MXFP4 experts with clipped SwiGLU",
    portfolio=PortfolioPolicy(upstream_standard=("flashinfer_cutlass_w4a16", "vllm_marlin_w4a16"),
                              repo_nonstandard=("triton_mxfp4_ue8m0",)),
)
MXFP4_MOE_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)(FlashInferMxfp4MoeProvider)
MXFP4_MOE_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)(VllmMarlinMxfp4MoeProvider)


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
        m = _validate_inputs(s, self.device, x, ids, route_weights)
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
