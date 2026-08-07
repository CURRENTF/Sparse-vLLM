from __future__ import annotations

from dataclasses import dataclass

import torch

import sparsevllm.platforms as platforms
from sparsevllm.engine.cache_manager.base import (
    DecodeComputeView,
    ExplicitKVPayload,
    MlaLatentPayload,
    PrefillComputeView,
)
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    SupportResult,
)
from sparsevllm.operators.sgl_fa3 import SglFa3DecodeKernel, sgl_fa3_support
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum
from sparsevllm.triton_kernel.mla import (
    DEFAULT_GLM_MLA_DECODE_CONFIG,
    GLM_MLA_MAX_WORKSPACE_CONFIG,
    MlaDecodeLaunchConfig,
    allocate_mla_decode_workspace,
    run_mla_decode,
    select_glm_mla_decode_config,
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
    supports_explicit_prefill = False

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
        launch_config: MlaDecodeLaunchConfig | None = None,
    ) -> None:
        self.spec = op_spec
        requested_device = torch.device(device)
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size <= 0:
            raise ValueError(
                "MLA max_batch_size must be positive, got "
                f"{self.max_batch_size}."
            )
        self._fixed_launch_config = launch_config
        self.launch_config = launch_config or DEFAULT_GLM_MLA_DECODE_CONFIG
        workspace_config = launch_config or GLM_MLA_MAX_WORKSPACE_CONFIG
        self.workspace = allocate_mla_decode_workspace(
            batch_size=self.max_batch_size,
            head_count=self.spec.local_q_heads,
            device=requested_device,
            config=workspace_config,
        )
        self.device = self.workspace.block_size.device
        self._validated_decode_metadata: tuple[
            object,
            list[
                tuple[
                    torch.Tensor,
                    torch.Tensor,
                    torch.Tensor,
                    int,
                    int | None,
                ]
            ],
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
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.no(
                "decode CUDA Graph requires platform graph capture support"
            )
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

    def _validate_metadata(
        self,
        view: DecodeComputeView | PrefillComputeView,
        payload: MlaLatentPayload,
        *,
        validation_scope: object | None,
    ) -> None:
        cache_slot_count = int(payload.latent_cache.shape[0])
        metadata_key = (
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            cache_slot_count,
            view.meta.max_context_len,
        )
        cached = self._validated_decode_metadata
        cached_entries = (
            cached[1]
            if validation_scope is not None
            and cached is not None
            and cached[0] is validation_scope
            else []
        )
        metadata_is_validated = any(
            entry[0] is metadata_key[0]
            and entry[1] is metadata_key[1]
            and entry[2] is metadata_key[2]
            and entry[3] == metadata_key[3]
            and entry[4] == metadata_key[4]
            for entry in cached_entries
        )
        if metadata_is_validated:
            return
        validate_mla_decode_metadata(
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            cache_slot_count=cache_slot_count,
            max_context_len=view.meta.max_context_len,
        )
        if validation_scope is None:
            self._validated_decode_metadata = None
        else:
            cached_entries.append(metadata_key)
            self._validated_decode_metadata = (validation_scope, cached_entries)

    def _launch_config_for(
        self,
        *,
        batch_size: int,
        max_context_len: int | None,
        active_slot_width: int,
    ) -> MlaDecodeLaunchConfig:
        if self._fixed_launch_config is not None:
            return self._fixed_launch_config
        context_capacity = (
            active_slot_width
            if max_context_len is None
            else int(max_context_len)
        )
        return select_glm_mla_decode_config(
            batch_size=batch_size,
            max_context_len=context_capacity,
            local_q_heads=self.spec.local_q_heads,
        )

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
        self._validate_metadata(
            view,
            payload,
            validation_scope=validation_scope,
        )
        launch_config = self._launch_config_for(
            batch_size=int(q_nope_absorbed.shape[0]),
            max_context_len=view.meta.max_context_len,
            active_slot_width=int(view.meta.active_slots.shape[1]),
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
            attn_score=view.meta.attn_score,
            max_context_len=view.meta.max_context_len,
            config=launch_config,
            validate_metadata=False,
        )


@MLA_ATTENTION_REGISTRY.register
class MlaSglFa3Provider(MlaTritonProvider):
    """SGL FA3 decode with the score-producing Triton path kept explicit."""

    name = "sgl_fa3_h100"
    priority = 200
    supports_explicit_prefill = True

    def __init__(
        self,
        *,
        op_spec: MlaAttentionOpSpec,
        device: torch.device | str,
        max_batch_size: int,
        launch_config: MlaDecodeLaunchConfig | None = None,
    ) -> None:
        super().__init__(
            op_spec=op_spec,
            device=device,
            max_batch_size=max_batch_size,
            launch_config=launch_config,
        )
        self.fa3 = SglFa3DecodeKernel(
            device=self.device,
            max_batch_size=self.max_batch_size,
            softmax_scale=self.spec.softmax_scale,
        )

    @classmethod
    def supports(
        cls,
        spec: MlaAttentionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        base = MlaTritonProvider.supports(spec, caps)
        if not base.supported:
            return base
        supported, reason = sgl_fa3_support()
        return SupportResult.yes(reason) if supported else SupportResult.no(reason)

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
        if view.meta.attn_score is not None:
            return super().run(
                q_nope_absorbed,
                q_rope,
                view,
                output,
                validation_scope=validation_scope,
            )
        payload = self._validate_run_inputs(
            q_nope_absorbed,
            q_rope,
            view,
            output,
        )
        self._validate_metadata(
            view,
            payload,
            validation_scope=validation_scope,
        )
        return self.fa3(
            q_rope,
            q_nope_absorbed,
            payload.rope_cache,
            payload.latent_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            output,
            num_splits=5,
            validation_scope=validation_scope,
        )

    @torch.no_grad()
    def run_explicit_prefill(
        self,
        q: torch.Tensor,
        view: PrefillComputeView,
        output: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        validation_scope: object | None = None,
    ) -> torch.Tensor:
        if not isinstance(view, PrefillComputeView):
            raise TypeError(
                "MlaSglFa3Provider.run_explicit_prefill requires "
                "PrefillComputeView, got "
                f"{type(view).__name__}."
            )
        if not isinstance(view.payload, ExplicitKVPayload):
            raise TypeError(
                "MLA explicit prefill requires ExplicitKVPayload, got "
                f"{type(view.payload).__name__}."
            )
        query_tokens = int(q.shape[0])
        expected_q_shape = (
            query_tokens,
            self.spec.local_q_heads,
            self.spec.qk_head_dim,
        )
        if tuple(q.shape) != expected_q_shape:
            raise ValueError(
                f"q must have shape {expected_q_shape}, got {tuple(q.shape)}."
            )
        expected_output_shape = (
            query_tokens,
            self.spec.local_q_heads,
            self.spec.value_head_dim,
        )
        if tuple(output.shape) != expected_output_shape:
            raise ValueError(
                f"output must have shape {expected_output_shape}, got "
                f"{tuple(output.shape)}."
            )
        batch_size = int(view.meta.context_lens.numel())
        if batch_size > self.max_batch_size:
            raise ValueError(
                "MLA prefill batch exceeds provider capacity: "
                f"batch={batch_size} max_batch_size={self.max_batch_size}."
            )
        if cu_seqlens_q.shape != (batch_size + 1,):
            raise ValueError(
                f"cu_seqlens_q must have shape ({batch_size + 1},), got "
                f"{tuple(cu_seqlens_q.shape)}."
            )
        if cu_seqlens_q.device != self.device or cu_seqlens_q.dtype != torch.int32:
            raise TypeError(
                "cu_seqlens_q must be int32 on the provider device, got "
                f"{cu_seqlens_q.device}/{cu_seqlens_q.dtype}."
            )
        if not 0 < int(max_seqlen_q) <= query_tokens:
            raise ValueError(
                "max_seqlen_q must be in [1, query_tokens], got "
                f"{max_seqlen_q} for {query_tokens}."
            )
        payload = view.payload
        tensors = {
            "q": q,
            "output": output,
            "k_cache": payload.k_cache,
            "v_cache": payload.v_cache,
        }
        for name, tensor in tensors.items():
            if tensor.device != self.device:
                raise ValueError(
                    f"{name} is on {tensor.device}, expected {self.device}."
                )
            expected_dtype = (
                self.spec.activation_dtype
            )
            if tensor.dtype != expected_dtype:
                raise TypeError(
                    f"{name} must use {expected_dtype}, got {tensor.dtype}."
                )
        return self.fa3.run_explicit_varlen(
            q,
            payload.k_cache,
            payload.v_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=int(max_seqlen_q),
            validation_scope=validation_scope,
        )

def resolve_mla_attention_provider(
    spec: MlaAttentionOpSpec,
    *,
    device: torch.device | str,
    max_batch_size: int,
    launch_config: MlaDecodeLaunchConfig | None = None,
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
    "MlaSglFa3Provider",
    "MlaTritonProvider",
    "resolve_mla_attention_provider",
]
