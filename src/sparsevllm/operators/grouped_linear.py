"""Grouped linear semantics with provider-owned checkpoint dequantization."""

from dataclasses import dataclass

import torch

from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class GroupedLinearSpec:
    groups: int
    input_features: int
    output_features: int
    max_num_tokens: int

    def __post_init__(self):
        if min(self.groups, self.input_features, self.output_features, self.max_num_tokens) <= 0:
            raise ValueError("Grouped linear dimensions and token capacity must be positive.")


GROUPED_LINEAR_REGISTRY = OpRegistry(
    "grouped linear", portfolio=PortfolioPolicy(upstream_standard=("torch_bmm",)),
)


@GROUPED_LINEAR_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class TorchGroupedLinearProvider:
    name = "torch_bmm"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA or not caps.supports_bfloat16:
            return SupportResult.unsupported("requires CUDA BF16")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        self.spec, self.device = spec, device

    def allocate_weights(self):
        s = self.spec
        return torch.empty((s.groups, s.output_features, s.input_features), dtype=torch.bfloat16, device=self.device)

    def load_fp8_weights(self, target, weight, scale):
        s = self.spec
        rows, cols = s.groups * s.output_features, s.input_features
        if rows % 128 or cols % 128 or weight.shape != (rows, cols) or scale.shape != (rows // 128, cols // 128):
            raise ValueError("Grouped FP8 checkpoint loading requires aligned 128x128 weight/scale blocks.")
        if weight.dtype != torch.float8_e4m3fn or scale.dtype not in (torch.float32, torch.float8_e8m0fnu):
            raise ValueError("Grouped FP8 loading requires E4M3 weights and FP32/E8M0 block scales.")
        if target.shape != (s.groups, s.output_features, s.input_features) or target.dtype != torch.bfloat16 or target.device != self.device:
            raise ValueError("Grouped linear target does not match prepared weight storage.")
        flat = target.view(rows, cols)
        # Match the checkpoint conversion's BF16 weight semantics with bounded
        # temporary storage; dequantization occurs once, before forward.
        for start in range(0, rows, 128):
            values = weight[start:start + 128].to(self.device).float().view(128, cols // 128, 128)
            scales = scale[start // 128].to(self.device).float()
            flat[start:start + 128].copy_((values * scales[None, :, None]).flatten(1))

    def run(self, x, weight):
        s = self.spec
        if x.ndim != 3 or x.shape[1:] != (s.groups, s.input_features) or len(x) > s.max_num_tokens:
            raise ValueError("Grouped linear input differs from prepared token/group/feature dimensions.")
        if weight.shape != (s.groups, s.output_features, s.input_features):
            raise ValueError("Grouped linear weights differ from the prepared dimensions.")
        if any(t.dtype != torch.bfloat16 or t.device != self.device or not t.is_contiguous() for t in (x, weight)):
            raise ValueError("Grouped linear inputs require contiguous BF16 tensors on the prepared device.")
        out = torch.empty((len(x), s.groups, s.output_features), device=self.device, dtype=torch.bfloat16)
        torch.bmm(x.transpose(0, 1), weight.transpose(1, 2), out=out.transpose(0, 1))
        return out


def prepare_grouped_linear(spec, *, device_index):
    return OpResolver(GROUPED_LINEAR_REGISTRY).resolve(spec, current_platform.get_device_caps(device_index)).provider
