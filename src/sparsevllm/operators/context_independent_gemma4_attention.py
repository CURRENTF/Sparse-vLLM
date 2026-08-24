"""Experimental fixed-grid Gemma 4 decode provider for batch-only graphs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sparsevllm.kernels.triton.sglang_gemma4_decode_attention import (
    sglang_gemma4_decode,
)
from sparsevllm.layers.attention_backend import _require_explicit_payload
from sparsevllm.operators.gemma4 import (
    GEMMA4_REGISTRY,
    Gemma4OpSpec,
    TritonGemma4OperatorProvider,
)
from sparsevllm.operators.gemma4_attention import Gemma4AttentionBackend
from sparsevllm.operators.registry import ProviderRole, SupportResult
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass
class _Gemma4DecodeWorkspace:
    mid_output: torch.Tensor
    mid_lse: torch.Tensor
    num_kv_splits: torch.Tensor


class ContextIndependentGemma4AttentionBackend(Gemma4AttentionBackend):
    name = "triton_gemma4_sglang_context_independent"
    context_independent_cuda_graph = True

    def __init__(
        self,
        *,
        sliding_window: int | None,
        workspace: _Gemma4DecodeWorkspace,
        device_core_count: int,
    ) -> None:
        super().__init__(sliding_window=sliding_window)
        self.workspace = workspace
        self.device_core_count = int(device_core_count)

    def get_decode_workspace(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        workspace = self.workspace
        if (
            batch_size > workspace.mid_output.shape[0]
            or num_heads != workspace.mid_output.shape[1]
            or head_dim != workspace.mid_output.shape[3]
            or device != workspace.mid_output.device
        ):
            raise RuntimeError(
                "Context-independent Gemma 4 workspace does not match the "
                f"decode contract: {(batch_size, num_heads, head_dim, device)}."
            )
        return workspace.mid_output[:batch_size], workspace.mid_lse[:batch_size]

    def binding_metadata(self) -> dict[str, object]:
        metadata = super().binding_metadata()
        return {
            **metadata,
            "decode_routes": ["sglang_fixed_grid"],
            "cuda_graph_shape_policy": "batch_only",
        }

    def run_decode(
        self,
        q: torch.Tensor,
        view,
        *,
        mid_o: torch.Tensor,
        mid_o_logexpsum: torch.Tensor,
        max_len_in_batch: int,
        block_seq: int,
        num_heads: int,
        num_kv_heads: int,
        gqa_block_n: int = 16,
        gqa_num_warps: int = 2,
    ) -> torch.Tensor:
        batch_size = int(q.shape[0])
        mid_o = self.workspace.mid_output[:batch_size]
        mid_o_logexpsum = self.workspace.mid_lse[:batch_size]
        del (
            max_len_in_batch,
            block_seq,
            num_heads,
            num_kv_heads,
            gqa_block_n,
            gqa_num_warps,
        )
        payload = _require_explicit_payload(
            view, operation="context-independent Gemma 4 decode"
        )
        if payload.backend != "dense":
            raise RuntimeError(
                "Context-independent Gemma 4 decode requires dense explicit KV."
            )
        self._record_kernel_path("sglang_fixed_grid")
        return sglang_gemma4_decode(
            q,
            payload.k_cache,
            payload.v_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            mid_o,
            mid_o_logexpsum,
            self.workspace.num_kv_splits[: int(q.shape[0])],
            sliding_window=self.sliding_window,
            device_core_count=self.device_core_count,
            attn_score=view.meta.attn_score,
        )


@GEMMA4_REGISTRY.register_atomic(ProviderRole.REPO_NONSTANDARD)
class ContextIndependentGemma4OperatorProvider(TritonGemma4OperatorProvider):
    name = "triton_gemma4_context_independent"

    def __init__(
        self,
        *,
        spec: Gemma4OpSpec,
        caps: DeviceCaps,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.device = torch.device("cuda", caps.device_index)
        self.device_core_count = int(caps.multiprocessor_count or 1)
        self._workspaces: dict[tuple[int, int, int], _Gemma4DecodeWorkspace] = {}

    @classmethod
    def supports(cls, spec: Gemma4OpSpec, caps: DeviceCaps) -> SupportResult:
        if not spec.context_independent_cuda_graph:
            return SupportResult.unsupported("reserved for batch-only CUDA Graph")
        if caps.platform != PlatformEnum.CUDA or not caps.supports_triton:
            return SupportResult.unsupported("requires CUDA with Triton")
        if not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        if spec.activation_dtype not in {torch.bfloat16, torch.float16}:
            return SupportResult.unsupported("requires BF16 or FP16 activations")
        if any(head_dim not in {256, 512} for head_dim in spec.head_dims):
            return SupportResult.unsupported("requires head dimensions 256 or 512")
        return SupportResult.yes()

    @classmethod
    def bind(
        cls,
        spec: Gemma4OpSpec,
        caps: DeviceCaps,
        **kwargs,
    ) -> "ContextIndependentGemma4OperatorProvider":
        if kwargs:
            raise TypeError(f"Unexpected Gemma 4 bind arguments: {sorted(kwargs)}.")
        return cls(spec=spec, caps=caps)

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "composite_provider",
            "implementation_source": "sglang_triton_adapted",
            "decode_kernel_path": "sglang_fixed_grid",
            "cuda_graph_shape_policy": "batch_only",
        }

    def attention_backend(self, *, sliding_window: int | None):
        window_left = -1 if sliding_window is None else int(sliding_window) - 1
        matching = [
            contract
            for contract in self.spec.attention_contracts
            if int(contract[3]) == window_left
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "Gemma 4 batch-only provider requires one attention contract "
                f"for window_left={window_left}, got {matching}."
            )
        query_heads, _, head_dim, _ = matching[0]
        signature = (int(query_heads), int(head_dim), 8)
        workspace = self._workspaces.get(signature)
        if workspace is None:
            workspace = _Gemma4DecodeWorkspace(
                mid_output=torch.empty(
                    (
                        self.spec.max_batch_size,
                        signature[0],
                        signature[2],
                        signature[1],
                    ),
                    dtype=torch.float32,
                    device=self.device,
                ),
                mid_lse=torch.empty(
                    (self.spec.max_batch_size, signature[0], signature[2]),
                    dtype=torch.float32,
                    device=self.device,
                ),
                num_kv_splits=torch.empty(
                    (self.spec.max_batch_size,),
                    dtype=torch.int32,
                    device=self.device,
                ),
            )
            self._workspaces[signature] = workspace
        return self._register_attention_backend(
            ContextIndependentGemma4AttentionBackend(
                sliding_window=sliding_window,
                workspace=workspace,
                device_core_count=self.device_core_count,
            )
        )

    def close(self) -> None:
        super().close()
        self._workspaces.clear()


__all__ = [
    "ContextIndependentGemma4AttentionBackend",
    "ContextIndependentGemma4OperatorProvider",
]
