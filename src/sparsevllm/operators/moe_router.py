from __future__ import annotations

from dataclasses import dataclass

import torch

import sparsevllm.platforms as platforms
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProviderRole,
    SupportResult,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class MoeRouterOpSpec:
    num_experts: int
    top_k: int
    activation_dtype: torch.dtype
    norm_topk_prob: bool
    cuda_graph: bool
    routing_method: str = "softmax"
    max_num_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.num_experts <= 0:
            raise ValueError("MoE router num_experts must be positive.")
        if not 1 <= self.top_k <= self.num_experts:
            raise ValueError(
                f"MoE router top_k must be in [1, {self.num_experts}], "
                f"got {self.top_k}."
            )
        if not self.activation_dtype.is_floating_point:
            raise TypeError(
                "MoE router activations must be floating point, "
                f"got {self.activation_dtype}."
            )
        if self.routing_method not in {"softmax", "biased_sigmoid", "sqrt_softplus", "hash_sqrt_softplus"}:
            raise ValueError(f"Unsupported MoE routing method {self.routing_method!r}.")


class MoeRouterProvider:
    name = ""

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "repo_triton",
            "kernel_path": self.name,
        }

    def run(
        self,
        spec: MoeRouterOpSpec,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError


MOE_ROUTER_REGISTRY: OpRegistry[MoeRouterOpSpec, MoeRouterProvider] = OpRegistry(
    "MoE router",
    portfolio=PortfolioPolicy(
        upstream_standard=("vllm_sqrt_softplus",),
        repo_nonstandard=(
            "triton_glm_biased_sigmoid",
            "triton_minimax_biased_sigmoid",
            "triton",
            "sqrt_softplus",
        )
    ),
)


@MOE_ROUTER_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class TritonMoeRouterProvider(MoeRouterProvider):
    name = "triton"

    def binding_metadata(self) -> dict[str, object]:
        return {
            **super().binding_metadata(),
            "kernel_path": "triton.moe_topk.topk_softmax",
        }

    @classmethod
    def supports(
        cls,
        spec: MoeRouterOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if spec.routing_method != "softmax":
            return SupportResult.unsupported("requires softmax routing")
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported(f"requires CUDA, got {caps.platform.name}")
        if not caps.supports_triton:
            return SupportResult.unsupported("platform does not support Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype not in {torch.bfloat16, torch.float16}:
            return SupportResult.unsupported(
                f"requires BF16 or FP16 logits, got {spec.activation_dtype}"
            )
        if spec.num_experts not in {128, 256} or spec.top_k != 8:
            return SupportResult.unsupported(
                "requires num_experts in {128, 256} and top_k=8"
            )
        return SupportResult.yes()

    def run(
        self,
        spec: MoeRouterOpSpec,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if correction_bias is not None or routed_scaling_factor != 1.0:
            raise ValueError("Softmax routing does not accept bias or route scaling.")
        from sparsevllm.kernels.triton.moe_topk import topk_softmax

        return topk_softmax(
            router_logits,
            top_k=spec.top_k,
            norm_topk_prob=spec.norm_topk_prob,
        )


@MOE_ROUTER_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class GlmBiasedSigmoidRouterProvider(MoeRouterProvider):
    name = "triton_glm_biased_sigmoid"

    def binding_metadata(self) -> dict[str, object]:
        return {
            **super().binding_metadata(),
            "kernel_path": (
                "triton.moe_biased_sigmoid.fused_topk_biased_sigmoid"
            ),
            "routing_contract": "glm_group_limited_biased_sigmoid",
        }

    @classmethod
    def supports(cls, spec: MoeRouterOpSpec, caps: DeviceCaps) -> SupportResult:
        if spec.routing_method != "biased_sigmoid":
            return SupportResult.unsupported("requires biased-sigmoid routing")
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA with Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if (spec.num_experts, spec.top_k) != (64, 4):
            return SupportResult.unsupported("requires 64 experts and top-k 4")
        return SupportResult.yes()

    def run(
        self,
        spec: MoeRouterOpSpec,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if correction_bias is None:
            raise ValueError("Biased-sigmoid routing requires correction_bias.")
        from sparsevllm.kernels.triton.moe_biased_sigmoid import (
            fused_topk_biased_sigmoid,
        )

        return fused_topk_biased_sigmoid(
            router_logits,
            correction_bias,
            top_k=spec.top_k,
            routed_scaling_factor=routed_scaling_factor,
        )


@MOE_ROUTER_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class MiniMaxBiasedSigmoidRouterProvider(MoeRouterProvider):
    """MiniMax M2's exact FP32 biased-sigmoid routing contract."""

    name = "triton_minimax_biased_sigmoid"

    @classmethod
    def supports(cls, spec: MoeRouterOpSpec, caps: DeviceCaps) -> SupportResult:
        if spec.routing_method != "biased_sigmoid":
            return SupportResult.unsupported("requires biased-sigmoid routing")
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA with Triton")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if (spec.num_experts, spec.top_k) != (256, 8):
            return SupportResult.unsupported("requires 256 experts and top-k 8")
        if spec.activation_dtype != torch.float32:
            return SupportResult.unsupported(
                f"requires FP32 logits, got {spec.activation_dtype}"
            )
        if not spec.norm_topk_prob:
            return SupportResult.unsupported("requires normalized top-k probabilities")
        return SupportResult.yes("MiniMax M2 FP32 biased-sigmoid router")

    def binding_metadata(self) -> dict[str, object]:
        return {
            **super().binding_metadata(),
            "kernel_path": "triton.minimax_m2_router.topk_biased_sigmoid",
            "routing_contract": "minimax_m2_biased_sigmoid",
        }

    def run(
        self,
        spec: MoeRouterOpSpec,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor | None = None,
        *,
        routed_scaling_factor: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if correction_bias is None:
            raise ValueError("MiniMax biased-sigmoid routing requires correction_bias.")
        if routed_scaling_factor != 1.0:
            raise ValueError("MiniMax biased-sigmoid routing does not accept route scaling.")
        from sparsevllm.kernels.triton.minimax_m2_router import (
            topk_biased_sigmoid,
        )

        return topk_biased_sigmoid(
            router_logits,
            correction_bias,
            top_k=spec.top_k,
        )


@MOE_ROUTER_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class SqrtSoftplusRouterProvider(MoeRouterProvider):
    name = "sqrt_softplus"

    @classmethod
    def supports(cls, spec, caps):
        if spec.routing_method not in ("sqrt_softplus", "hash_sqrt_softplus"):
            return SupportResult.unsupported("requires sqrt-softplus or hash sqrt-softplus routing")
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA and Triton")
        if spec.activation_dtype != torch.float32 or not spec.norm_topk_prob:
            return SupportResult.unsupported("requires FP32 logits and normalized route weights")
        if spec.max_num_tokens is None or spec.max_num_tokens <= 0:
            return SupportResult.unsupported("requires a positive prepared token capacity")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        return SupportResult.yes("native sqrt-softplus scores with Torch top-k or checkpoint hash indices")

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, platforms.current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        from sparsevllm.operators.workspace import get_workspace_manager
        self.spec, self.device = spec, device
        self._regions, offset = {}, 0
        rows, k = spec.max_num_tokens, spec.top_k
        regions = [("weights", rows * k, torch.float32), ("ids", rows * k, torch.int64),
                   ("norms", rows, torch.float32)]
        if spec.routing_method == "sqrt_softplus":
            regions.append(("scores", rows * spec.num_experts, torch.float32))
        for name, count, dtype in regions:
            offset = (offset + 255) // 256 * 256
            size = count * dtype.itemsize
            self._regions[name] = offset, size, dtype
            offset += size
        self._lease = get_workspace_manager(device, create=True).reserve_bytes(
            offset, label="sqrt_softplus_router", lane="sqrt_softplus_router",
        )

    def _view(self, name, rows, columns):
        offset, size, dtype = self._regions[name]
        return self._lease.buffer[offset:offset+size].view(dtype)[:rows * columns].view(rows, columns)

    def run(self, spec, router_logits, correction_bias=None, *, routed_scaling_factor=1.0,
            hash_indices=None, input_ids=None):
        import math
        if spec != self.spec:
            raise ValueError("Router spec changed after preparation.")
        if (router_logits.ndim != 2 or router_logits.shape[1] != spec.num_experts
                or router_logits.shape[0] > spec.max_num_tokens or router_logits.dtype != torch.float32
                or router_logits.device != self.device or not router_logits.is_contiguous()):
            raise ValueError("Router logits do not match prepared FP32 capacity/layout.")
        if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
            raise ValueError("Routing scale must be finite and positive.")
        rows = router_logits.shape[0]
        weights, ids = self._view("weights", rows, spec.top_k), self._view("ids", rows, spec.top_k)
        if rows == 0:
            return weights, ids
        from sparsevllm.kernels.triton.moe_sqrt_softplus import (
            biased_sqrt_softplus_scores, gather_sqrt_softplus_scores, normalize_sqrt_softplus_weights,
        )
        if spec.routing_method == "hash_sqrt_softplus":
            if (correction_bias is not None or hash_indices is None or input_ids is None
                    or hash_indices.ndim != 2 or hash_indices.shape[1] != spec.top_k
                    or input_ids.shape != (rows,)):
                raise ValueError("Hash routing requires a token-to-expert table and original token IDs.")
            if any(t.dtype not in (torch.int32, torch.int64) or t.device != self.device
                   or not t.is_contiguous() for t in (hash_indices, input_ids)):
                raise ValueError("Hash routing metadata must be contiguous integer tensors on the logits device.")
        else:
            if (hash_indices is not None or input_ids is not None or correction_bias is None
                    or correction_bias.shape != (spec.num_experts,) or correction_bias.dtype != torch.float32
                    or correction_bias.device != self.device or not correction_bias.is_contiguous()):
                raise ValueError("Learned sqrt-softplus routing requires an FP32 correction bias.")
            scores = self._view("scores", rows, spec.num_experts)
            biased_sqrt_softplus_scores(router_logits, correction_bias, scores)
            # Public Torch top-k preserves the reference's selection and tie behavior.
            torch.topk(scores, spec.top_k, dim=-1, out=(weights, ids))
        gather_sqrt_softplus_scores(router_logits, ids, weights, table=hash_indices, token_ids=input_ids)
        norms = self._view("norms", rows, 1)
        torch.sum(weights, dim=-1, keepdim=True, out=norms)
        normalize_sqrt_softplus_weights(weights, norms, ids, routed_scaling_factor)
        return weights, ids


@MOE_ROUTER_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class VllmSqrtSoftplusRouterProvider(MoeRouterProvider):
    name = "vllm_sqrt_softplus"

    @classmethod
    def supports(cls, spec, caps):
        if spec.routing_method not in ("sqrt_softplus", "hash_sqrt_softplus"):
            return SupportResult.unsupported("requires learned or hash sqrt-softplus routing")
        if caps.platform != PlatformEnum.CUDA or spec.activation_dtype != torch.float32:
            return SupportResult.unsupported("requires CUDA FP32 logits")
        if not spec.norm_topk_prob or spec.max_num_tokens is None or spec.max_num_tokens <= 0:
            return SupportResult.unsupported("requires normalized weights and prepared token capacity")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("requires CUDA Graph support")
        if spec.routing_method == "hash_sqrt_softplus" and not caps.supports_torch_compile:
            return SupportResult.unsupported("hash input preparation requires torch.compile")
        from sparsevllm.kernels.external.vllm_moe import sqrt_softplus_op
        if sqrt_softplus_op() is None:
            return SupportResult.unsupported("optional vLLM MoE library is not installed or configured")
        return SupportResult.yes()

    @classmethod
    def bind(cls, spec, caps):
        return cls(spec, platforms.current_platform.get_device(caps.device_index))

    def __init__(self, spec, device):
        from sparsevllm.kernels.external.vllm_moe import sqrt_softplus_op
        from sparsevllm.operators.workspace import get_workspace_manager
        self.spec, self.device = spec, device
        self._op = sqrt_softplus_op()
        self._hash_inputs = None
        if spec.routing_method == "hash_sqrt_softplus":
            from sparsevllm.kernels.external.vllm_moe import hash_inputs_op
            self._hash_inputs = hash_inputs_op()
        self._elements = spec.max_num_tokens * spec.top_k
        self._lease = get_workspace_manager(device, create=True).reserve_bytes(
            12 * self._elements, label="vllm_sqrt_softplus", lane="sqrt_softplus_router",
        )

    def binding_metadata(self):
        return dict(implementation_kind="atomic_provider", implementation_source="vllm_stable_abi",
                    kernel_path="_moe_C.topk_softplus_sqrt")

    def run(self, spec, router_logits, correction_bias=None, *, routed_scaling_factor=1.0,
            hash_indices=None, input_ids=None):
        import math
        if spec != self.spec:
            raise ValueError("Prepared vLLM router requires an unchanged spec.")
        rows = len(router_logits)
        if router_logits.shape != (rows, spec.num_experts) or rows > spec.max_num_tokens:
            raise ValueError("Router logits do not match prepared capacity/layout.")
        use_hash = spec.routing_method == "hash_sqrt_softplus"
        if use_hash:
            if (correction_bias is not None or hash_indices is None or input_ids is None
                    or hash_indices.ndim != 2 or hash_indices.shape[1] != spec.top_k
                    or len(hash_indices) == 0 or input_ids.shape != (rows,)):
                raise ValueError("Hash routing requires a token-to-expert table and original token IDs.")
            if any(t.dtype not in (torch.int32, torch.int64) or t.device != self.device
                   or not t.is_contiguous() for t in (hash_indices, input_ids)):
                raise ValueError("Hash routing metadata must be contiguous integer tensors on the logits device.")
        elif (hash_indices is not None or input_ids is not None or correction_bias is None
              or correction_bias.shape != (spec.num_experts,)):
            raise ValueError("Learned sqrt-softplus routing requires an expert correction bias.")
        floating_inputs = (router_logits,) if use_hash else (router_logits, correction_bias)
        if any(t.dtype != torch.float32 or t.device != self.device or not t.is_contiguous()
               for t in floating_inputs):
            raise ValueError("Router logits and bias must be contiguous FP32 on the prepared device.")
        if not math.isfinite(routed_scaling_factor) or routed_scaling_factor <= 0:
            raise ValueError("Routing scale must be finite and positive.")
        parts = self._lease.buffer.view(torch.int32).split(self._elements)
        weights = parts[0].view(torch.float32)[:rows * spec.top_k].view(rows, spec.top_k)
        ids, token_experts = (part[:rows * spec.top_k].view(rows, spec.top_k) for part in parts[1:])
        if rows:
            tokens = padding = None
            if use_hash:
                # The external kernel checks padding before reading the table
                # and requires IDs and table entries to use the same dtype.
                tokens, padding = self._hash_inputs(input_ids, len(hash_indices), hash_indices.dtype)
            self._op(weights, ids, token_experts, router_logits, True, routed_scaling_factor,
                     correction_bias, tokens, hash_indices, padding)
        return weights, ids


def resolve_moe_router_provider(
    spec: MoeRouterOpSpec,
    *,
    device_index: int | None = None,
) -> MoeRouterProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(MOE_ROUTER_REGISTRY).resolve(spec, caps).provider
