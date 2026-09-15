"""Prepared MXFP4 experts backed by vLLM's stable-ABI Marlin kernels."""

import math

import torch

from sparsevllm.kernels.external.vllm_marlin import MXFP4_TYPE_ID
from sparsevllm.operators.mxfp4_moe_contract import Mxfp4ExpertWeights, _validate_inputs
from sparsevllm.operators.registry import SupportResult
from sparsevllm.operators.workspace import get_workspace_manager
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


class VllmMarlinMxfp4MoeProvider:
    name = "vllm_marlin_w4a16"
    weight_layout_id = "vllm_marlin_mxfp4_gate_up_e8m0"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability is None or caps.compute_capability < (8, 0):
            return SupportResult.unsupported("BF16 Marlin requires CUDA SM80 or newer")
        if not caps.supports_bfloat16 or (spec.cuda_graph and not caps.supports_graph_capture):
            return SupportResult.unsupported("requires BF16 and requested CUDA Graph support")
        if not caps.multi_processor_count:
            return SupportResult.unsupported("requires device multiprocessor count for Marlin locks")
        from sparsevllm.kernels.external.vllm_marlin import marlin_ops
        if marlin_ops() is None:
            return SupportResult.dependency_absent("optional vLLM stable-ABI library is unavailable")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index), caps.multi_processor_count)

    def __init__(self, spec, device, sm_count):
        from sparsevllm.kernels.external.vllm_marlin import marlin_ops
        self._repack, self._align, self._gemm, self._activation = marlin_ops()
        self.spec, self.device = spec, device
        self._permutation = torch.empty(0, device=device, dtype=torch.int32)
        self._expert_map = torch.full((spec.num_experts,), -1, device=device, dtype=torch.int32)
        self._expert_map[spec.local_expert_start:spec.local_expert_start + spec.num_local_experts] = torch.arange(
            spec.num_local_experts, device=device, dtype=torch.int32)
        # Marlin resets these locks after each GEMM. They cannot alias scratch
        # used by other operators or uninitialized workspace storage.
        self._locks = torch.zeros(sm_count * 4, device=device, dtype=torch.int32)
        assignments = spec.max_num_tokens * spec.top_k
        max_padded = assignments + spec.num_experts * 63
        self._regions, offset = {}, 0
        for name, count, dtype in (
            ("sorted", max_padded, torch.int32),
            ("experts", (max_padded + 7) // 8, torch.int32),
            ("count", 1, torch.int32),
            ("projection", assignments * max(2 * spec.intermediate_size, spec.hidden_size), torch.bfloat16),
            ("activation", assignments * spec.intermediate_size, torch.bfloat16),
        ):
            offset = (offset + 255) // 256 * 256
            size = count * dtype.itemsize
            self._regions[name] = offset, size, dtype
            offset += size
        self._lease = get_workspace_manager(device, create=True).reserve_bytes(
            offset, label="vllm_marlin_mxfp4", lane="mxfp4_moe")

    def binding_metadata(self):
        return {"kernel_path": "vllm_stable_abi._moe_C.moe_wna16_marlin_gemm",
                "interface": "raw stable-ABI kernels; no vLLM engine dispatcher"}

    def _view(self, name, *shape):
        offset, size, dtype = self._regions[name]
        return self._lease.buffer[offset:offset + size].view(dtype)[:math.prod(shape)].view(shape)

    def allocate_weights(self):
        s = self.spec
        def weight(n, k):
            return torch.empty((s.num_local_experts, k // 16, n * 2), device=self.device, dtype=torch.int32)
        def scale(n, k):
            return torch.empty((s.num_local_experts, k // 32, n), device=self.device, dtype=torch.float8_e8m0fnu)
        return Mxfp4ExpertWeights(weight(2 * s.intermediate_size, s.hidden_size),
                                 weight(s.hidden_size, s.intermediate_size),
                                 scale(2 * s.intermediate_size, s.hidden_size),
                                 scale(s.hidden_size, s.intermediate_size))

    def load_projection(self, storage, local_expert, projection, weight, scale):
        from sparsevllm.kernels.external.vllm_marlin import pack_mxfp4_projection
        s = self.spec
        if not 0 <= local_expert < s.num_local_experts:
            raise ValueError("MXFP4 checkpoint expert index lies outside local storage.")
        if projection == "down":
            n, k = s.hidden_size, s.intermediate_size
            target, scale_target = storage.down[local_expert], storage.down_scale[local_expert]
        elif projection in ("gate", "up"):
            n, k = s.intermediate_size, s.hidden_size
            offset = 0 if projection == "gate" else n
            target = storage.gate_up[local_expert, :, 2 * offset:2 * (offset + n)]
            scale_target = storage.gate_up_scale[local_expert, :, offset:offset + n]
        else:
            raise ValueError(f"Unknown MXFP4 logical projection {projection!r}.")
        if weight.dtype not in (torch.int8, torch.uint8, torch.float4_e2m1fn_x2) or scale.dtype not in (torch.uint8, torch.float8_e8m0fnu):
            raise TypeError("MXFP4 checkpoint loading requires packed FP4 and E8M0 bytes.")
        if weight.shape != (n, k // 2) or scale.shape != (n, k // 32):
            raise ValueError("MXFP4 checkpoint projection does not match prepared storage.")
        packed, scales = pack_mxfp4_projection(
            self._repack, weight.view(torch.uint8).to(self.device).contiguous(),
            scale.view(torch.uint8).to(self.device).contiguous(), self._permutation)
        target.copy_(packed)
        scale_target.copy_(scales)

    def run(self, x, ids, route_weights, storage):
        s = self.spec
        rows = _validate_inputs(s, self.device, x, ids, route_weights)
        if not rows:
            return x.new_empty((0, s.hidden_size), dtype=torch.float32)
        # vLLM 0.29's BF16 Marlin launch heuristic; shape-only, no autotuning.
        local_rows = (rows * s.num_local_experts + s.num_experts - 1) // s.num_experts
        for block in (8, 16, 32, 48, 64):
            if local_rows * s.top_k / s.num_local_experts / block < .9:
                break
        assignments = rows * s.top_k
        padded = assignments + s.num_experts * (block - 1)
        if assignments < s.num_experts:
            padded = min(assignments * block, padded)
        sorted_ids = self._view("sorted", padded)
        experts = self._view("experts", (padded + block - 1) // block)
        count = self._view("count", 1)
        first = self._view("projection", assignments, 2 * s.intermediate_size)
        activated = self._view("activation", assignments, s.intermediate_size)
        first.zero_()
        self._align(ids, s.num_experts, block, sorted_ids, experts, count, self._expert_map)
        self._gemm(x, first, storage.gate_up, None, storage.gate_up_scale, None, None, None, None, None,
                   self._locks, sorted_ids, experts, count, route_weights, block, s.top_k, False,
                   MXFP4_TYPE_ID, rows, 2 * s.intermediate_size, s.hidden_size,
                   True, False, True, False, -1, -1, -1)
        self._activation(activated, first, s.swiglu_limit, 1., 0.)
        final = self._view("projection", assignments, s.hidden_size)
        final.zero_()
        self._gemm(activated, final, storage.down, None, storage.down_scale, None, None, None, None, None,
                   self._locks, sorted_ids, experts, count, route_weights, block, 1, True,
                   MXFP4_TYPE_ID, assignments, s.hidden_size, s.intermediate_size,
                   True, False, True, False, -1, -1, -1)
        return final.view(rows, s.top_k, s.hidden_size).sum(1, dtype=torch.float32)
