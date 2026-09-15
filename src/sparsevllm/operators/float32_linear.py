"""FP32 projection over active rows with reusable scratch."""

from dataclasses import dataclass

import torch

from sparsevllm.operators.registry import OpRegistry, OpResolver, PortfolioPolicy, ProviderRole, SupportResult
from sparsevllm.operators.workspace import get_workspace_manager
from sparsevllm.platforms import current_platform
from sparsevllm.platforms.interface import PlatformEnum


@dataclass(frozen=True)
class Float32LinearSpec:
    input_features: int
    output_features: int
    max_num_tokens: int
    activation_dtype: torch.dtype = torch.bfloat16
    cuda_graph: bool = True
    num_projections: int = 1

    def __post_init__(self):
        if min(self.input_features, self.output_features, self.max_num_tokens, self.num_projections) <= 0:
            raise ValueError("FP32 linear dimensions and token capacity must be positive.")


FLOAT32_LINEAR_REGISTRY = OpRegistry(
    "FP32 linear", portfolio=PortfolioPolicy(upstream_standard=("torch_mm",)),
)


@FLOAT32_LINEAR_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class TorchFloat32LinearProvider:
    name = "torch_mm"

    @classmethod
    def supports(cls, spec, caps):
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported("requires CUDA")
        if spec.activation_dtype not in (torch.bfloat16, torch.float32):
            return SupportResult.unsupported("requires BF16 or FP32 activations")
        if spec.activation_dtype == torch.bfloat16 and not caps.supports_bfloat16:
            return SupportResult.unsupported("requires BF16 conversion support")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        self.spec, self.device = spec, device
        self._input_elements = spec.max_num_tokens * spec.input_features
        self._lease = get_workspace_manager(device, create=True).reserve_bytes(
            4 * (self._input_elements + spec.num_projections * spec.max_num_tokens * spec.output_features),
            label="float32_linear", lane="float32_linear",
        )

    def run(self, x, weight):
        """Return scratch logits, valid until the next projection on this lane."""
        return self.run_many(x, (weight,))[0]

    def run_many(self, x, weights):
        """Project one input to disjoint scratch outputs with a shared conversion.

        All outputs remain valid until the next projection on this workspace
        lane. This lets gated pooling consume KV and gate values together.
        """
        s = self.spec
        if x.ndim != 2 or x.shape[1] != s.input_features or len(x) > s.max_num_tokens:
            raise ValueError("FP32 linear input differs from prepared dimensions or token capacity.")
        if len(weights) != s.num_projections or any(w.shape != (s.output_features, s.input_features) for w in weights):
            raise ValueError("FP32 linear weights differ from the prepared dimensions.")
        if x.dtype != s.activation_dtype or any(w.dtype != torch.float32 for w in weights):
            raise TypeError("FP32 linear requires the prepared activation dtype and FP32 weights.")
        if any(t.device != self.device or not t.is_contiguous() for t in (x, *weights)):
            raise ValueError("FP32 linear inputs must be contiguous on the prepared device.")
        scratch = self._lease.buffer.view(torch.float32)
        active = scratch[:self._input_elements].view(s.max_num_tokens, s.input_features)[:len(x)]
        outputs = scratch[self._input_elements:].view(s.num_projections, s.max_num_tokens, s.output_features).unbind(0)
        if len(x):
            active.copy_(x)
            for weight, output in zip(weights, outputs):
                torch.mm(active, weight.t(), out=output[:len(x)])
        return tuple(output[:len(x)] for output in outputs)


def prepare_float32_linear(spec, *, device_index):
    return OpResolver(FLOAT32_LINEAR_REGISTRY).resolve(spec, current_platform.get_device_caps(device_index)).provider
