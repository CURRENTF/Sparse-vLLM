from __future__ import annotations

from dataclasses import dataclass

import torch

import sparsevllm.platforms as platforms
from sparsevllm.engine.cache_manager.base import (
    DecodeComputeView,
    MlaLatentPayload,
)
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum
from sparsevllm.triton_kernel.mla import (
    DEFAULT_GLM_MLA_DECODE_CONFIG,
    MlaDecodeLaunchConfig,
    allocate_mla_decode_workspace,
    run_mla_decode,
    validate_mla_decode_metadata,
)


_GLM_MLA_NUM_Q_HEADS = 20
_GLM_MLA_KV_LORA_RANK = 512
_GLM_MLA_ROPE_DIM = 64
_GLM_MLA_QK_HEAD_DIM = 256
_GLM_MLA_VALUE_HEAD_DIM = 256
_VALIDATED_H100_NAME = "NVIDIA H100 80GB HBM3"


@dataclass(frozen=True, slots=True)
class MlaAttentionOpSpec:
    """Construction-time contract for one MLA attention implementation."""

    num_q_heads: int
    kv_lora_rank: int
    rope_dim: int
    qk_head_dim: int
    value_head_dim: int
    activation_dtype: torch.dtype
    cache_dtype: torch.dtype
    tp_size: int
    cuda_graph: bool

    def __post_init__(self) -> None:
        dimensions = {
            "num_q_heads": self.num_q_heads,
            "kv_lora_rank": self.kv_lora_rank,
            "rope_dim": self.rope_dim,
            "qk_head_dim": self.qk_head_dim,
            "value_head_dim": self.value_head_dim,
            "tp_size": self.tp_size,
        }
        for name, value in dimensions.items():
            if int(value) <= 0:
                raise ValueError(f"MLA {name} must be positive, got {value}.")
        if self.num_q_heads % self.tp_size:
            raise ValueError(
                "MLA query heads must be divisible by tensor parallel size: "
                f"heads={self.num_q_heads} tp_size={self.tp_size}."
            )

    @property
    def local_q_heads(self) -> int:
        return int(self.num_q_heads // self.tp_size)

    @property
    def softmax_scale(self) -> float:
        return float(self.qk_head_dim**-0.5)


class MlaAttentionProvider:
    name = ""
    priority = 0

    def run(
        self,
        q_nope_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        view: DecodeComputeView,
        output: torch.Tensor,
        *,
        validation_scope: object | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError


MLA_ATTENTION_REGISTRY: OpRegistry[
    MlaAttentionOpSpec,
    MlaAttentionProvider,
] = OpRegistry("MLA attention")


@MLA_ATTENTION_REGISTRY.register
class MlaTritonProvider(MlaAttentionProvider):
    """H100 provider bound once with caller-independent decode workspace."""

    name = "triton_h100"
    priority = 100

    def __init__(
        self,
        *,
        op_spec: MlaAttentionOpSpec,
        device: torch.device | str,
        max_batch_size: int,
        launch_config: MlaDecodeLaunchConfig = DEFAULT_GLM_MLA_DECODE_CONFIG,
    ) -> None:
        self.spec = op_spec
        requested_device = torch.device(device)
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size <= 0:
            raise ValueError(
                "MLA max_batch_size must be positive, got "
                f"{self.max_batch_size}."
            )
        self.launch_config = launch_config
        self.workspace = allocate_mla_decode_workspace(
            batch_size=self.max_batch_size,
            head_count=self.spec.local_q_heads,
            device=requested_device,
            config=self.launch_config,
        )
        self.device = self.workspace.block_size.device
        self._validated_decode_metadata: tuple[
            object,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            int,
        ] | None = None

    @classmethod
    def supports(
        cls,
        spec: MlaAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA or caps.compute_capability != (9, 0):
            return SupportResult.no(
                f"requires CUDA SM90, got {caps.platform.name} "
                f"{caps.compute_capability}"
            )
        if caps.device_name != _VALIDATED_H100_NAME:
            return SupportResult.no(
                "requires validated NVIDIA H100 80GB HBM3 hardware, got "
                f"{caps.device_name}"
            )
        if not caps.supports_triton:
            return SupportResult.no("platform does not support Triton")
        if not caps.supports_bfloat16:
            return SupportResult.no("device does not support BF16")
        if spec.activation_dtype != torch.bfloat16:
            return SupportResult.no(
                f"requires BF16 activations, got {spec.activation_dtype}"
            )
        if spec.cache_dtype != torch.bfloat16:
            return SupportResult.no(
                f"requires BF16 cache storage, got {spec.cache_dtype}"
            )
        expected_shape = (
            _GLM_MLA_NUM_Q_HEADS,
            _GLM_MLA_KV_LORA_RANK,
            _GLM_MLA_ROPE_DIM,
            _GLM_MLA_QK_HEAD_DIM,
            _GLM_MLA_VALUE_HEAD_DIM,
        )
        actual_shape = (
            spec.num_q_heads,
            spec.kv_lora_rank,
            spec.rope_dim,
            spec.qk_head_dim,
            spec.value_head_dim,
        )
        if actual_shape != expected_shape:
            return SupportResult.no(
                f"requires GLM MLA shape {expected_shape}, got {actual_shape}"
            )
        if spec.tp_size not in {1, 2, 4}:
            return SupportResult.no(
                f"requires tensor parallel size 1, 2, or 4, got {spec.tp_size}"
            )
        if spec.cuda_graph:
            return SupportResult.no("CUDA Graph execution is not validated")
        return SupportResult.yes()

    def _validate_run_inputs(
        self,
        q_nope_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        view: DecodeComputeView,
        output: torch.Tensor,
    ) -> MlaLatentPayload:
        if not isinstance(view, DecodeComputeView):
            raise TypeError(
                "MlaTritonProvider.run requires DecodeComputeView, got "
                f"{type(view).__name__}."
            )
        if not isinstance(view.payload, MlaLatentPayload):
            raise TypeError(
                "MLA decode requires MlaLatentPayload, got "
                f"{type(view.payload).__name__}."
            )
        if q_nope_absorbed.ndim != 3:
            raise ValueError(
                "q_nope_absorbed must have shape [batch, local_heads, 512], "
                f"got {tuple(q_nope_absorbed.shape)}."
            )
        expected_query_shape = (
            int(q_nope_absorbed.shape[0]),
            self.spec.local_q_heads,
            self.spec.kv_lora_rank,
        )
        if tuple(q_nope_absorbed.shape) != expected_query_shape:
            raise ValueError(
                "q_nope_absorbed must have shape "
                f"{expected_query_shape}, got {tuple(q_nope_absorbed.shape)}."
            )
        expected_rope_shape = (
            expected_query_shape[0],
            expected_query_shape[1],
            self.spec.rope_dim,
        )
        if tuple(q_rope.shape) != expected_rope_shape:
            raise ValueError(
                f"q_rope must have shape {expected_rope_shape}, got "
                f"{tuple(q_rope.shape)}."
            )
        if output.shape != q_nope_absorbed.shape:
            raise ValueError(
                f"output must have shape {tuple(q_nope_absorbed.shape)}, got "
                f"{tuple(output.shape)}."
            )
        if expected_query_shape[0] > self.max_batch_size:
            raise ValueError(
                "MLA decode batch exceeds the bound workspace: "
                f"batch={expected_query_shape[0]} max_batch_size="
                f"{self.max_batch_size}."
            )
        tensors = {
            "q_nope_absorbed": q_nope_absorbed,
            "q_rope": q_rope,
            "output": output,
            "latent_cache": view.payload.latent_cache,
            "rope_cache": view.payload.rope_cache,
        }
        for name, tensor in tensors.items():
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected {self.device}."
                )
            expected_dtype = (
                self.spec.cache_dtype
                if name in {"latent_cache", "rope_cache"}
                else self.spec.activation_dtype
            )
            if tensor.dtype != expected_dtype:
                raise TypeError(
                    f"{name} must use {expected_dtype}, got {tensor.dtype}."
                )
        return view.payload

    @torch.no_grad()
    def run(
        self,
        q_nope_absorbed: torch.Tensor,
        q_rope: torch.Tensor,
        view: DecodeComputeView,
        output: torch.Tensor,
        *,
        validation_scope: object | None = None,
    ) -> torch.Tensor:
        payload = self._validate_run_inputs(
            q_nope_absorbed,
            q_rope,
            view,
            output,
        )
        cache_slot_count = int(payload.latent_cache.shape[0])
        metadata_key = (
            validation_scope,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            cache_slot_count,
        )
        cached_key = self._validated_decode_metadata
        metadata_is_validated = (
            validation_scope is not None
            and cached_key is not None
            and cached_key[0] is validation_scope
            and cached_key[1] is metadata_key[1]
            and cached_key[2] is metadata_key[2]
            and cached_key[3] is metadata_key[3]
            and cached_key[4] == metadata_key[4]
        )
        if not metadata_is_validated:
            validate_mla_decode_metadata(
                view.meta.active_slots,
                view.meta.req_indices,
                view.meta.context_lens,
                cache_slot_count=cache_slot_count,
            )
            self._validated_decode_metadata = (
                metadata_key if validation_scope is not None else None
            )
        return run_mla_decode(
            q_nope_absorbed,
            q_rope,
            payload.latent_cache,
            payload.rope_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            output,
            self.workspace,
            softmax_scale=self.spec.softmax_scale,
            config=self.launch_config,
            validate_metadata=False,
        )


def resolve_mla_attention_provider(
    spec: MlaAttentionOpSpec,
    *,
    device: torch.device | str,
    max_batch_size: int,
    launch_config: MlaDecodeLaunchConfig = DEFAULT_GLM_MLA_DECODE_CONFIG,
) -> MlaAttentionProvider:
    """Resolve and bind an MLA provider during model construction."""

    device = torch.device(device)
    device_index = 0 if device.index is None else int(device.index)
    caps = platforms.current_platform.get_device_caps(device_index)
    return OpResolver(MLA_ATTENTION_REGISTRY).resolve(
        spec,
        caps,
        op_spec=spec,
        device=device,
        max_batch_size=max_batch_size,
        launch_config=launch_config,
    ).provider


__all__ = [
    "MLA_ATTENTION_REGISTRY",
    "MlaAttentionOpSpec",
    "MlaAttentionProvider",
    "MlaTritonProvider",
    "resolve_mla_attention_provider",
]
