from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from sparsevllm.kernels.external.sgl.support import sgl_kernel_support

_COMMON_FWD_ARGUMENTS = (
    "q",
    "k",
    "v",
    "k_new",
    "v_new",
    "q_v",
    "out",
    "cu_seqlens_q",
    "cu_seqlens_k",
    "cu_seqlens_k_new",
    "seqused_q",
    "seqused_k",
    "max_seqlen_q",
    "max_seqlen_k",
    "page_table",
    "kv_batch_idx",
    "leftpad_k",
    "rotary_cos",
    "rotary_sin",
    "seqlens_rotary",
    "q_descale",
    "k_descale",
    "v_descale",
    "softmax_scale",
    "is_causal",
    "window_size_left",
    "window_size_right",
)
_FWD_TRAILING_ARGUMENTS = (
    "softcap",
    "is_rotary_interleaved",
    "scheduler_metadata",
    "num_splits",
    "pack_gqa",
    "sm_margin",
    "sinks",
)
_FWD_ARGUMENTS = _COMMON_FWD_ARGUMENTS + ("attention_chunk",) + _FWD_TRAILING_ARGUMENTS


@dataclass(frozen=True, slots=True)
class _SchedulerMetadataPlan:
    """Step-local FA3 scheduling metadata shared by every MLA layer."""

    validation_scope: object
    context_lens: torch.Tensor
    cu_seqlens_q: torch.Tensor
    batch_size: int
    max_seqlen_q: int
    max_seqlen_k: int
    num_heads: int
    num_heads_k: int
    headdim: int
    headdim_v: int
    qkv_dtype: torch.dtype
    num_splits: int
    metadata: torch.Tensor

    def matches(
        self,
        *,
        validation_scope: object,
        context_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        batch_size: int,
        max_seqlen_q: int,
        max_seqlen_k: int,
        num_heads: int,
        num_heads_k: int,
        headdim: int,
        headdim_v: int,
        qkv_dtype: torch.dtype,
        num_splits: int,
    ) -> bool:
        return (
            self.validation_scope is validation_scope
            and self.context_lens.data_ptr() == context_lens.data_ptr()
            and self.cu_seqlens_q.data_ptr() == cu_seqlens_q.data_ptr()
            and self.batch_size == int(batch_size)
            and self.max_seqlen_q == int(max_seqlen_q)
            and self.max_seqlen_k == int(max_seqlen_k)
            and self.num_heads == int(num_heads)
            and self.num_heads_k == int(num_heads_k)
            and self.headdim == int(headdim)
            and self.headdim_v == int(headdim_v)
            and self.qkv_dtype == qkv_dtype
            and self.num_splits == int(num_splits)
        )


def _sgl_fa3_op():
    # Register FA3 without importing the unrelated optional FA4 wrapper.
    importlib.import_module("sgl_kernel.flash_ops")
    return torch.ops.sgl_kernel.fwd.default


def sgl_fa3_support() -> tuple[bool, str]:
    supported, reason = sgl_kernel_support("FA3 raw op")
    if not supported:
        return supported, reason
    try:
        op = _sgl_fa3_op()
        argument_names = tuple(argument.name for argument in op._schema.arguments)
    except Exception as error:
        return False, (
            "sgl-kernel FA3 raw op failed to load: "
            f"{type(error).__name__}: {error}"
        )
    if argument_names != _FWD_ARGUMENTS:
        return False, f"unsupported sgl-kernel FA3 fwd schema: {argument_names}"
    return True, reason


class SglFa3DecodeKernel:
    """Allocation-free GLM MLA decode adapter for the SGL FA3 raw op."""

    def __init__(
        self,
        *,
        device: torch.device,
        max_batch_size: int,
        softmax_scale: float,
        num_splits: int = 0,
    ) -> None:
        supported, reason = sgl_fa3_support()
        if not supported:
            raise RuntimeError(reason)

        self._op = _sgl_fa3_op()
        scheduler_packet = getattr(
            torch.ops.sgl_kernel,
            "get_scheduler_metadata",
            None,
        )
        self._scheduler_op = (
            None if scheduler_packet is None else scheduler_packet.default
        )
        self._scheduler_plan: _SchedulerMetadataPlan | None = None
        self._captured_scheduler_plans: list[_SchedulerMetadataPlan] = []
        self._cu_seqlens_q = torch.arange(
            int(max_batch_size) + 1,
            dtype=torch.int32,
            device=device,
        )
        self.softmax_scale = float(softmax_scale)
        self.num_splits = int(num_splits)
        if self.num_splits < 0:
            raise ValueError(
                f"FA3 num_splits must be non-negative, got {self.num_splits}."
            )

    def __call__(
        self,
        q_rope: torch.Tensor,
        q_latent: torch.Tensor,
        rope_cache: torch.Tensor,
        latent_cache: torch.Tensor,
        page_table: torch.Tensor,
        request_indices: torch.Tensor,
        context_lens: torch.Tensor,
        output: torch.Tensor,
        *,
        num_splits: int | None = None,
        validation_scope: object | None = None,
    ) -> torch.Tensor:
        batch_size = int(q_rope.shape[0])
        return self.run_varlen(
            q_rope,
            q_latent,
            rope_cache,
            latent_cache,
            page_table,
            request_indices,
            context_lens,
            output,
            cu_seqlens_q=self._cu_seqlens_q[: batch_size + 1],
            max_seqlen_q=1,
            num_splits=num_splits,
            validation_scope=validation_scope,
        )

    def _scheduler_metadata(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        page_table: torch.Tensor,
        context_lens: torch.Tensor,
        cu_seqlens_q: torch.Tensor,
        *,
        headdim_v: int,
        max_seqlen_q: int,
        num_splits: int,
        validation_scope: object | None,
    ) -> torch.Tensor | None:
        if self._scheduler_op is None:
            return None
        batch_size = int(context_lens.numel())
        max_seqlen_k = int(page_table.shape[1])
        num_heads = int(q.shape[1])
        num_heads_k = int(k_cache.shape[1])
        headdim = int(q.shape[-1])
        headdim_v = int(headdim_v)
        is_capturing = torch.cuda.is_current_stream_capturing()
        plans = (
            reversed(self._captured_scheduler_plans)
            if is_capturing
            else (self._scheduler_plan,)
        )
        plan = next(
            (
                candidate
                for candidate in plans
                if validation_scope is not None
                and candidate is not None
                and candidate.matches(
                    validation_scope=validation_scope,
                    context_lens=context_lens,
                    cu_seqlens_q=cu_seqlens_q,
                    batch_size=batch_size,
                    max_seqlen_q=max_seqlen_q,
                    max_seqlen_k=max_seqlen_k,
                    num_heads=num_heads,
                    num_heads_k=num_heads_k,
                    headdim=headdim,
                    headdim_v=headdim_v,
                    qkv_dtype=q.dtype,
                    num_splits=num_splits,
                )
            ),
            None,
        )
        if plan is not None:
            return plan.metadata

        metadata = self._scheduler_op(
            batch_size,
            int(max_seqlen_q),
            max_seqlen_k,
            num_heads,
            num_heads_k,
            headdim,
            headdim_v,
            q.dtype,
            context_lens,
            cu_seqlens_q,
            None,
            None,
            None,
            None,
            1,
            0,
            True,
            -1,
            -1,
            0,
            False,
            int(num_splits),
            None,
            0,
        )
        if validation_scope is not None:
            plan = _SchedulerMetadataPlan(
                validation_scope=validation_scope,
                context_lens=context_lens,
                cu_seqlens_q=cu_seqlens_q,
                batch_size=batch_size,
                max_seqlen_q=int(max_seqlen_q),
                max_seqlen_k=max_seqlen_k,
                num_heads=num_heads,
                num_heads_k=num_heads_k,
                headdim=headdim,
                headdim_v=headdim_v,
                qkv_dtype=q.dtype,
                num_splits=int(num_splits),
                metadata=metadata,
            )
            if is_capturing:
                # CUDA graphs retain raw pointers, not Python tensor owners.
                # Keep every capture's metadata alive for the graph lifetime.
                self._captured_scheduler_plans.append(plan)
            else:
                self._scheduler_plan = plan
        return metadata

    def run_varlen(
        self,
        q_rope: torch.Tensor,
        q_latent: torch.Tensor,
        rope_cache: torch.Tensor,
        latent_cache: torch.Tensor,
        page_table: torch.Tensor,
        request_indices: torch.Tensor,
        context_lens: torch.Tensor,
        output: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        num_splits: int | None = None,
        validation_scope: object | None = None,
    ) -> torch.Tensor:
        split_count = self.num_splits if num_splits is None else int(num_splits)
        if split_count < 0:
            raise ValueError(
                f"FA3 num_splits must be non-negative, got {split_count}."
            )
        scheduler_metadata = self._scheduler_metadata(
            q_rope,
            rope_cache,
            page_table,
            context_lens,
            cu_seqlens_q,
            headdim_v=int(q_latent.shape[-1]),
            max_seqlen_q=int(max_seqlen_q),
            num_splits=split_count,
            validation_scope=validation_scope,
        )
        args: list[object] = [
            q_rope,
            rope_cache.unsqueeze(1),
            latent_cache.unsqueeze(1),
            None,
            None,
            q_latent,
            output,
            cu_seqlens_q,
            None,
            None,
            None,
            context_lens,
            int(max_seqlen_q),
            None,
            page_table,
            request_indices,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.softmax_scale,
            True,
            -1,
            -1,
        ]
        args.append(0)
        args.extend(
            (0.0, True, scheduler_metadata, split_count, None, 0, None)
        )
        result: Sequence[torch.Tensor] = self._op(*args)
        if not result or result[0].data_ptr() != output.data_ptr():
            raise RuntimeError("sgl-kernel FA3 did not write to the supplied output")
        return output

    def run_explicit_varlen(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        page_table: torch.Tensor,
        request_indices: torch.Tensor,
        context_lens: torch.Tensor,
        output: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        max_seqlen_q: int,
        validation_scope: object | None = None,
    ) -> torch.Tensor:
        """Run causal varlen attention over page-size-one explicit KV."""

        scheduler_metadata = self._scheduler_metadata(
            q,
            k_cache,
            page_table,
            context_lens,
            cu_seqlens_q,
            headdim_v=int(v_cache.shape[-1]),
            max_seqlen_q=int(max_seqlen_q),
            num_splits=self.num_splits,
            validation_scope=validation_scope,
        )

        args: list[object] = [
            q,
            k_cache.unsqueeze(1),
            v_cache.unsqueeze(1),
            None,
            None,
            None,
            output,
            cu_seqlens_q,
            None,
            None,
            None,
            context_lens,
            int(max_seqlen_q),
            None,
            page_table,
            request_indices,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.softmax_scale,
            True,
            -1,
            -1,
        ]
        args.append(0)
        args.extend(
            (0.0, True, scheduler_metadata, self.num_splits, None, 0, None)
        )
        result: Sequence[torch.Tensor] = self._op(*args)
        if not result or result[0].data_ptr() != output.data_ptr():
            raise RuntimeError("sgl-kernel FA3 did not write to the supplied output")
        return output

    def run_contiguous_explicit_varlen(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        output: torch.Tensor,
        *,
        cu_seqlens_q: torch.Tensor,
        cu_seqlens_k: torch.Tensor,
        max_seqlen_q: int,
        max_seqlen_k: int,
    ) -> torch.Tensor:
        """Run causal varlen attention over packed contiguous KV."""

        args: list[object] = [
            q,
            k,
            v,
            None,
            None,
            None,
            output,
            cu_seqlens_q,
            cu_seqlens_k,
            None,
            None,
            None,
            int(max_seqlen_q),
            int(max_seqlen_k),
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            self.softmax_scale,
            True,
            -1,
            -1,
        ]
        args.append(0)
        args.extend((0.0, True, None, self.num_splits, None, 0, None))
        result: Sequence[torch.Tensor] = self._op(*args)
        if not result or result[0].data_ptr() != output.data_ptr():
            raise RuntimeError("sgl-kernel FA3 did not write to the supplied output")
        return output


__all__ = ["SglFa3DecodeKernel", "sgl_fa3_support"]
