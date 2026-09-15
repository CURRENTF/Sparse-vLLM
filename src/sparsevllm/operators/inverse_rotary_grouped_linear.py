"""Inverse adjacent-pair rotary followed by a grouped checkpoint projection."""

from dataclasses import dataclass
import math
from weakref import WeakValueDictionary

import torch

from sparsevllm.operators.grouped_linear import GroupedLinearSpec, prepare_grouped_linear
from sparsevllm.operators.native_rope import NativeRotarySpec, prepare_native_rotary
from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class InverseRotaryGroupedLinearSpec:
    groups: int
    heads: int
    head_dim: int
    output_features: int
    max_num_tokens: int
    max_positions: int
    frequencies: tuple[float, ...]

    def __post_init__(self):
        if min(self.groups, self.heads, self.head_dim, self.output_features,
               self.max_num_tokens, self.max_positions) <= 0 or self.heads % self.groups:
            raise ValueError("Inverse rotary projection requires positive dimensions and heads divisible by groups.")
        if (not self.frequencies or 2 * len(self.frequencies) > self.head_dim or self.head_dim % 2
                or any(not math.isfinite(f) for f in self.frequencies)):
            raise ValueError("Inverse rotary projection requires finite frequencies within an even head dimension.")

    @property
    def input_features(self):
        return self.heads // self.groups * self.head_dim


INVERSE_ROTARY_GROUPED_LINEAR_REGISTRY = OpRegistry(
    "inverse rotary grouped linear",
    portfolio=PortfolioPolicy(upstream_standard=("vllm_deepgemm",), repo_nonstandard=("rotary_torch_bmm",)),
)


class InverseRotaryGroupedLinearProvider:
    def __init__(self, spec, device):
        self.spec, self.device = spec, device

    def validate_input(self, x, positions):
        s = self.spec
        if x.ndim != 3 or x.shape[1:] != (s.heads, s.head_dim) or len(x) > s.max_num_tokens:
            raise ValueError("Inverse rotary projection input exceeds prepared dimensions/capacity.")
        if x.dtype != torch.bfloat16 or positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("Inverse rotary projection requires BF16 activations and integer positions.")
        if positions.shape != (len(x),) or any(t.device != self.device or not t.is_contiguous() for t in (x, positions)):
            raise ValueError("Inverse rotary projection requires contiguous inputs on the prepared device.")

    def validate_checkpoint(self, weight, scale):
        s = self.spec
        rows, cols = s.groups * s.output_features, s.input_features
        if (rows % 128 or cols % 128 or weight.shape != (rows, cols)
                or scale.shape != (rows // 128, cols // 128)):
            raise ValueError("Grouped checkpoint requires aligned 128x128 weight and scale blocks.")
        if weight.dtype != torch.float8_e4m3fn or scale.dtype not in (torch.float32, torch.float8_e8m0fnu):
            raise TypeError("Grouped checkpoint requires E4M3 weights and FP32/E8M0 block scales.")


# Providers retain their read-only table; destroying the last owner releases it.
# Layers sharing frequencies do not each allocate a context-length table.
_COS_SIN_TABLES = WeakValueDictionary()


def _cos_sin_table(spec, device):
    key = (device, spec.max_positions, spec.frequencies)
    table = _COS_SIN_TABLES.get(key)
    if table is None:
        frequencies = torch.tensor(spec.frequencies, device=device, dtype=torch.float32)
        angles = torch.arange(spec.max_positions, device=device, dtype=torch.float32)[:, None] * frequencies
        table = torch.cat((angles.cos(), angles.sin()), dim=-1)
        _COS_SIN_TABLES[key] = table
    return table


@torch.compile(fullgraph=True, dynamic=True)
def _safe_positions(positions, max_positions):
    # Negative positions are graph padding. The runtime owns the upper context
    # bound; assert on device before dereferencing the cached rotary table.
    torch._assert_async(torch.all(positions < max_positions), "rotary position exceeds prepared context capacity")
    return positions.clamp_min(0).to(torch.int64)


@torch.compile(fullgraph=True, dynamic=True)
def _mask_padding(output, positions):
    output.masked_fill_((positions < 0)[:, None, None], 0)


@INVERSE_ROTARY_GROUPED_LINEAR_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class VllmDeepGemmInverseProjectionProvider(InverseRotaryGroupedLinearProvider):
    name = "vllm_deepgemm"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.unsupported("adapter uses the SM90 FP32 scale contract")
        if not (caps.supports_triton and caps.supports_native_fp8 and caps.supports_torch_compile):
            return SupportResult.unsupported("requires Triton, native FP8 and torch.compile")
        if spec.head_dim != 512 or len(spec.frequencies) != 32 or spec.output_features % 128:
            return SupportResult.unsupported("requires D512, rotary D64 and block-aligned output features")
        from sparsevllm.kernels.external.vllm_projection import inverse_projection_ops
        if inverse_projection_ops() is None:
            return SupportResult.unsupported("requires optional native-Torch DeepGEMM and configured vLLM source")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        super().__init__(spec, device)
        from sparsevllm.kernels.external.vllm_projection import inverse_projection_ops
        self._quantize, self._einsum = inverse_projection_ops()
        self.cos_sin_cache = _cos_sin_table(spec, device)

    def allocate_weights(self):
        s = self.spec
        return (
            torch.empty((s.groups, s.output_features, s.input_features), device=self.device, dtype=torch.float8_e4m3fn),
            torch.empty((s.groups, s.output_features // 128, s.input_features // 128), device=self.device, dtype=torch.float32),
        )

    def load_fp8_weights(self, target, target_scale, weight, scale):
        self.validate_checkpoint(weight, scale)
        target.copy_(weight.reshape(target.shape))
        target_scale.copy_(scale.reshape(target_scale.shape))

    def run(self, x, positions, weight, weight_scale):
        self.validate_input(x, positions)
        s = self.spec
        out = torch.empty((len(x), s.groups, s.output_features), device=self.device, dtype=torch.bfloat16)
        if len(x):
            from sparsevllm.kernels.external.vllm_projection import quantize_inverse_rotary
            safe_positions = _safe_positions(positions, s.max_positions)
            activation = quantize_inverse_rotary(self._quantize, x, safe_positions, self.cos_sin_cache, s.groups)
            self._einsum("bhr,hdr->bhd", activation, (weight, weight_scale), out, recipe=(1, 128, 128))
            _mask_padding(out, positions)
        return out


@INVERSE_ROTARY_GROUPED_LINEAR_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class RotaryTorchGroupedProjectionProvider(InverseRotaryGroupedLinearProvider):
    name = "rotary_torch_bmm"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton or not caps.supports_bfloat16:
            return SupportResult.unsupported("requires CUDA BF16 and Triton")
        if spec.groups * spec.output_features % 128 or spec.input_features % 128:
            return SupportResult.unsupported("requires block-aligned grouped FP8 checkpoint")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        super().__init__(spec, device)
        self._rotary = prepare_native_rotary(
            NativeRotarySpec(spec.head_dim, 2 * len(spec.frequencies), inverse=True), device_index=device.index,
        )
        self._linear = prepare_grouped_linear(
            GroupedLinearSpec(spec.groups, spec.input_features, spec.output_features, spec.max_num_tokens),
            device_index=device.index,
        )
        self._frequencies = torch.tensor(spec.frequencies, device=device, dtype=torch.float32)

    def allocate_weights(self):
        return self._linear.allocate_weights(), None

    def load_fp8_weights(self, target, target_scale, weight, scale):
        self.validate_checkpoint(weight, scale)
        self._linear.load_fp8_weights(target, weight, scale)

    def run(self, x, positions, weight, weight_scale):
        self.validate_input(x, positions)
        if len(x):
            self._rotary.run(x, positions, self._frequencies, out=x)
        return self._linear.run(x.view(len(x), self.spec.groups, self.spec.input_features), weight)


def prepare_inverse_rotary_grouped_linear(spec, *, device_index):
    return OpResolver(INVERSE_ROTARY_GROUPED_LINEAR_REGISTRY).resolve(
        spec, current_platform.get_device_caps(device_index),
    ).provider
