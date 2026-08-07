from __future__ import annotations

import importlib.metadata
import importlib.util
import re
from collections.abc import Sequence
from dataclasses import dataclass

import torch


_SGL_KERNEL_DISTRIBUTIONS = ("sglang-kernel", "sgl-kernel")
_MIN_VALIDATED_VERSIONS = {
    (0, 3): (0, 3, 14),
    (0, 4): (0, 4, 5),
}

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
_FWD_04_EXTRA_ARGUMENTS = (
    "sparse_mask_fine",
    "only_qv",
)


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


def _parse_version(version: str) -> tuple[int, int, int] | None:
    parts = str(version).split(".")
    if len(parts) < 3:
        return None
    parsed: list[int] = []
    for part in parts[:3]:
        match = re.match(r"(\d+)", part)
        if match is None:
            return None
        parsed.append(int(match.group(1)))
    return tuple(parsed)  # type: ignore[return-value]


def _installed_sgl_kernel_version() -> tuple[str, tuple[int, int, int]] | None:
    for distribution in _SGL_KERNEL_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        parsed = _parse_version(version)
        if parsed is None:
            return version, (-1, -1, -1)
        return version, parsed
    return None


def sgl_fa3_support() -> tuple[bool, str]:
    """Probe package metadata without importing optional SGL Python wrappers."""

    try:
        module_spec = importlib.util.find_spec("sgl_kernel")
    except (ImportError, ValueError):
        module_spec = None
    if module_spec is None:
        return False, "sgl-kernel is not installed"
    installed = _installed_sgl_kernel_version()
    if installed is None:
        return False, "sgl-kernel package metadata is unavailable"
    version, parsed = installed
    minimum = _MIN_VALIDATED_VERSIONS.get(parsed[:2])
    if minimum is None:
        return (
            False,
            "requires validated sgl-kernel 0.3.x or sglang-kernel 0.4.x "
            f"API, got {version}",
        )
    if parsed < minimum:
        package = "sglang-kernel" if parsed[:2] == (0, 4) else "sgl-kernel"
        required = ".".join(str(value) for value in minimum)
        return False, f"requires {package} >= {required}, got {version}"
    return True, f"SGL kernel {version} FA3 raw op is available"


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

        # Importing flash_ops registers FA3 only.  Importing
        # sgl_kernel.flash_attn would also import the optional FA4 wrapper,
        # whose CUTLASS Python dependency is unrelated to this provider.
        from sgl_kernel import flash_ops as _flash_ops  # noqa: F401

        op = torch.ops.sgl_kernel.fwd.default
        argument_names = tuple(argument.name for argument in op._schema.arguments)
        legacy = _COMMON_FWD_ARGUMENTS + _FWD_TRAILING_ARGUMENTS
        chunked = (
            _COMMON_FWD_ARGUMENTS
            + ("attention_chunk",)
            + _FWD_TRAILING_ARGUMENTS
        )
        chunked_04 = chunked + _FWD_04_EXTRA_ARGUMENTS
        if argument_names == legacy:
            self._has_attention_chunk = False
            self._has_04_arguments = False
        elif argument_names == chunked:
            self._has_attention_chunk = True
            self._has_04_arguments = False
        elif argument_names == chunked_04:
            self._has_attention_chunk = True
            self._has_04_arguments = True
        else:
            raise RuntimeError(
                "Unsupported sgl-kernel FA3 fwd schema: "
                f"arguments={argument_names}."
            )
        self._op = op
        scheduler_packet = getattr(
            torch.ops.sgl_kernel,
            "get_scheduler_metadata",
            None,
        )
        # sgl-kernel 0.3.14 exposes the FA3 scheduler only internally. Keep
        # its validated per-call scheduling path; 0.4.5 exposes the reusable
        # raw op used below.
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
        if self._has_attention_chunk:
            args.append(0)
        args.extend(
            (0.0, True, scheduler_metadata, split_count, None, 0, None)
        )
        if self._has_04_arguments:
            args.extend((None, False))
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
        if self._has_attention_chunk:
            args.append(0)
        args.extend(
            (0.0, True, scheduler_metadata, self.num_splits, None, 0, None)
        )
        if self._has_04_arguments:
            args.extend((None, False))
        result: Sequence[torch.Tensor] = self._op(*args)
        if not result or result[0].data_ptr() != output.data_ptr():
            raise RuntimeError("sgl-kernel FA3 did not write to the supplied output")
        return output


__all__ = ["SglFa3DecodeKernel", "sgl_fa3_support"]
