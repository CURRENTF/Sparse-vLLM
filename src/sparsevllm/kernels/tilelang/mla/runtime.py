"""Lazy TileLang adapter for GLM TP1/TP2/TP4 MLA decode.

The repository-owned TileLang kernel is shape-specialized.  This adapter keeps
compilation, padded-query storage, and split-KV workspaces outside the kernel
call and caches them by the CUDA Graph's static batch/context shape.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata

import torch

_VALIDATED_TILELANG_VERSION = "0.1.9"
_VALIDATED_TVM_FFI_VERSION = "0.1.10"
_VALID_SPLITS = (1, 2, 4, 8, 16, 32)
_SUPPORTED_VALID_HEADS = (5, 10, 20)
_SCORE_MODES = ("direct", "atomic", "partial")
_HEAD_TILE_SIZE = 16
_LATENT_DIM = 512
_ROPE_DIM = 64
_BLOCK_N = 64
_CALIBRATED_CONTEXT_BUCKETS = (1024, 4096, 8192, 16384, 32768, 65536)
_CALIBRATED_BATCH_BUCKETS = (1, 8, 32)
_FUSED_SCORE_SPLITS = {
    5: {
        1: (16, 32, 32, 32, 32, 32),
        8: (16, 16, 16, 16, 16, 32),
        32: (4, 4, 4, 8, 8, 16),
    },
    10: {
        1: (16, 32, 32, 32, 32, 32),
        8: (16, 16, 16, 16, 16, 32),
        32: (4, 4, 4, 8, 4, 4),
    },
    20: {
        1: (16, 32, 32, 32, 32, 32),
        8: (16, 16, 16, 16, 16, 16),
        32: (4, 4, 4, 4, 8, 8),
    },
}
_OUTPUT_ONLY_SPLITS = {
    5: {
        1: (16, 32, 32, 32, 32, 32),
        8: (16, 16, 16, 16, 16, 32),
        32: (4, 4, 4, 8, 4, 4),
    },
    10: {
        1: (16, 32, 32, 32, 32, 32),
        8: (16, 16, 16, 16, 16, 32),
        32: (4, 4, 4, 8, 8, 4),
    },
    20: {
        1: (16, 32, 32, 32, 32, 32),
        8: (16, 16, 16, 16, 16, 16),
        32: (4, 4, 4, 4, 4, 8),
    },
}


def _padded_head_count(valid_heads: int) -> int:
    if valid_heads not in _SUPPORTED_VALID_HEADS:
        raise ValueError(
            "TileLang MLA valid_heads must be one of "
            f"{_SUPPORTED_VALID_HEADS}, got {valid_heads}."
        )
    return 32 if valid_heads > _HEAD_TILE_SIZE else _HEAD_TILE_SIZE


def tilelang_mla_support() -> tuple[bool, str]:
    """Check the optional package without importing or initializing TileLang."""

    try:
        version = metadata.version("tilelang")
    except metadata.PackageNotFoundError:
        return False, "tilelang is not installed"
    if version != _VALIDATED_TILELANG_VERSION:
        return False, (
            f"requires validated tilelang=={_VALIDATED_TILELANG_VERSION}, got "
            f"{version!r}"
        )
    try:
        tvm_ffi_version = metadata.version("apache-tvm-ffi")
    except metadata.PackageNotFoundError:
        return False, "apache-tvm-ffi is not installed"
    if tvm_ffi_version != _VALIDATED_TVM_FFI_VERSION:
        return False, (
            "requires validated apache-tvm-ffi=="
            f"{_VALIDATED_TVM_FFI_VERSION}, got {tvm_ffi_version!r}"
        )
    return True, f"tilelang {version}, apache-tvm-ffi {tvm_ffi_version}"


@dataclass(frozen=True, slots=True)
class TileMlaLaunchConfig:
    num_split: int
    block_n: int = _BLOCK_N
    block_h: int = _HEAD_TILE_SIZE
    score_mode: str = "direct"

    def __post_init__(self) -> None:
        if self.num_split not in _VALID_SPLITS:
            raise ValueError(
                f"TileLang MLA num_split must be one of {_VALID_SPLITS}, "
                f"got {self.num_split}."
            )
        if self.block_h not in (16, 32):
            raise ValueError(
                f"TileLang MLA block_h must be 16 or 32, got {self.block_h}."
            )
        if self.score_mode not in _SCORE_MODES:
            raise ValueError(
                f"TileLang MLA score_mode must be one of {_SCORE_MODES}, "
                f"got {self.score_mode!r}."
            )


def select_tile_mla_config(
    *,
    batch_size: int,
    context_capacity: int,
    need_score: bool,
    local_q_heads: int = 10,
) -> TileMlaLaunchConfig:
    """Select an offline-calibrated split; never benchmark in the hot path.

    The table is the GPU4 H100 sweep over BS 1/8/32 and context
    1K/4K/8K/16K/32K/64K. Shapes between calibration points use the next
    larger bucket; larger shapes reuse the largest calibrated bucket.
    """

    if batch_size <= 0 or context_capacity <= 0:
        raise ValueError(
            "TileLang MLA batch/context must be positive, got "
            f"batch={batch_size} context={context_capacity}."
        )
    batch_bucket = next(
        (bucket for bucket in _CALIBRATED_BATCH_BUCKETS if batch_size <= bucket),
        _CALIBRATED_BATCH_BUCKETS[-1],
    )
    context_index = next(
        (
            index
            for index, bucket in enumerate(_CALIBRATED_CONTEXT_BUCKETS)
            if context_capacity <= bucket
        ),
        len(_CALIBRATED_CONTEXT_BUCKETS) - 1,
    )
    _padded_head_count(local_q_heads)
    table = _FUSED_SCORE_SPLITS if need_score else _OUTPUT_ONLY_SPLITS
    split = table[local_q_heads][batch_bucket][context_index]
    block_h = 32 if local_q_heads == 20 and batch_bucket > 1 else 16
    score_mode = "direct"
    if need_score and local_q_heads == 20 and block_h == 16:
        score_mode = "atomic" if context_capacity <= 4096 else "partial"
    return TileMlaLaunchConfig(
        num_split=split,
        block_h=block_h,
        score_mode=score_mode,
    )


@dataclass(slots=True)
class TileMlaWorkspace:
    padded_latent: torch.Tensor
    padded_rope: torch.Tensor
    glse: torch.Tensor
    partial_output: torch.Tensor
    score: torch.Tensor


@dataclass(frozen=True, slots=True)
class _KernelKey:
    batch_size: int
    cache_slot_count: int
    active_slot_rows: int
    active_slot_width: int
    score_capacity: int
    num_split: int
    block_h: int
    score_mode: str
    need_score: bool


@dataclass(slots=True)
class _BoundKernel:
    call: Callable[..., object]
    workspace: TileMlaWorkspace


class TileMlaDecodeKernel:
    """Shape-cached GLM TP1/TP2/TP4 TileLang MLA runner."""

    def __init__(
        self,
        *,
        device: torch.device | str,
        softmax_scale: float,
        valid_heads: int = 10,
        fixed_config: TileMlaLaunchConfig | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.softmax_scale = float(softmax_scale)
        self.valid_heads = int(valid_heads)
        self.padded_heads = _padded_head_count(self.valid_heads)
        if fixed_config is not None:
            if self.padded_heads % fixed_config.block_h:
                raise ValueError(
                    "TileLang MLA padded heads must be divisible by block_h: "
                    f"padded_heads={self.padded_heads} "
                    f"block_h={fixed_config.block_h}."
                )
        self.fixed_config = fixed_config
        self._kernels: dict[_KernelKey, _BoundKernel] = {}

    def _config_for(
        self,
        *,
        batch_size: int,
        context_capacity: int,
        need_score: bool,
    ) -> TileMlaLaunchConfig:
        if self.fixed_config is not None:
            return self.fixed_config
        return select_tile_mla_config(
            batch_size=batch_size,
            context_capacity=context_capacity,
            need_score=need_score,
            local_q_heads=self.valid_heads,
        )

    def _bind(self, key: _KernelKey) -> _BoundKernel:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "TileLang MLA shape was not warmed before CUDA Graph capture: "
                f"{key}."
            )
        supported, reason = tilelang_mla_support()
        if not supported:
            raise RuntimeError(reason)
        # Importing TileLang can initialize its compiler, so keep it behind the
        # selected provider and outside module import/resolver paths.
        from sparsevllm.kernels.tilelang.mla.decode import (
            build_glm_mla_decode_kernel,
        )

        config = TileMlaLaunchConfig(
            key.num_split,
            block_h=key.block_h,
            score_mode=key.score_mode,
        )
        kernel = build_glm_mla_decode_kernel(
            batch=key.batch_size,
            h_q=self.padded_heads,
            h_kv=1,
            valid_output_heads=self.valid_heads,
            cache_slots=key.cache_slot_count,
            slot_rows=key.active_slot_rows,
            active_slot_width=key.active_slot_width,
            max_seqlen_pad=key.score_capacity,
            dv=_LATENT_DIM,
            dpe=_ROPE_DIM,
            block_N=config.block_n,
            block_H=config.block_h,
            num_split=config.num_split,
            block_size=config.block_n,
            softmax_scale=self.softmax_scale,
            need_score=key.need_score,
            score_mode=config.score_mode,
        )
        dtype = torch.bfloat16
        workspace = TileMlaWorkspace(
            padded_latent=torch.empty(
                key.batch_size,
                self.padded_heads,
                _LATENT_DIM,
                dtype=dtype,
                device=self.device,
            ),
            padded_rope=torch.empty(
                key.batch_size,
                self.padded_heads,
                _ROPE_DIM,
                dtype=dtype,
                device=self.device,
            ),
            glse=torch.empty(
                key.batch_size,
                self.padded_heads,
                key.num_split,
                dtype=dtype,
                device=self.device,
            ),
            partial_output=torch.empty(
                key.batch_size,
                self.padded_heads,
                key.num_split,
                _LATENT_DIM,
                dtype=dtype,
                device=self.device,
            ),
            score=torch.empty(
                key.batch_size,
                self.padded_heads // config.block_h
                if config.score_mode == "partial"
                else 1,
                key.score_capacity,
                dtype=torch.float32,
                device=self.device,
            ),
        )
        return _BoundKernel(call=kernel, workspace=workspace)

    def _validate(
        self,
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        latent_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        active_slots: torch.Tensor,
        request_indices: torch.Tensor,
        context_lens: torch.Tensor,
        output: torch.Tensor,
        attn_score: torch.Tensor | None,
        max_context_len: int,
    ) -> tuple[int, int]:
        batch_size = int(q_latent.shape[0])
        expected = {
            "q_latent": (batch_size, self.valid_heads, _LATENT_DIM),
            "q_rope": (batch_size, self.valid_heads, _ROPE_DIM),
            "output": (batch_size, self.valid_heads, _LATENT_DIM),
            "latent_cache": (int(latent_cache.shape[0]), 1, _LATENT_DIM),
            "rope_cache": (int(latent_cache.shape[0]), 1, _ROPE_DIM),
            "request_indices": (batch_size,),
            "context_lens": (batch_size,),
        }
        actual = {
            "q_latent": tuple(q_latent.shape),
            "q_rope": tuple(q_rope.shape),
            "output": tuple(output.shape),
            "latent_cache": tuple(latent_cache.shape),
            "rope_cache": tuple(rope_cache.shape),
            "request_indices": tuple(request_indices.shape),
            "context_lens": tuple(context_lens.shape),
        }
        for name, shape in expected.items():
            if actual[name] != shape:
                raise ValueError(
                    f"TileLang MLA {name} must have shape {shape}, "
                    f"got {actual[name]}."
                )
        if active_slots.ndim != 2:
            raise ValueError(
                "TileLang MLA active_slots must be 2D, got "
                f"{tuple(active_slots.shape)}."
            )
        tensors = {
            "q_latent": (q_latent, torch.bfloat16),
            "q_rope": (q_rope, torch.bfloat16),
            "latent_cache": (latent_cache, torch.bfloat16),
            "rope_cache": (rope_cache, torch.bfloat16),
            "active_slots": (active_slots, torch.int32),
            "request_indices": (request_indices, torch.int32),
            "context_lens": (context_lens, torch.int32),
            "output": (output, torch.bfloat16),
        }
        for name, (tensor, dtype) in tensors.items():
            if tensor.device != q_latent.device or tensor.dtype != dtype:
                raise TypeError(
                    f"TileLang MLA {name} must be {dtype} on "
                    f"{q_latent.device}, got {tensor.dtype} on {tensor.device}."
                )
            if name not in {"q_latent", "q_rope"} and not tensor.is_contiguous():
                raise ValueError(
                    f"TileLang MLA {name} must be contiguous, got stride "
                    f"{tuple(tensor.stride())}."
                )
        score_capacity = int(active_slots.shape[1])
        if attn_score is not None:
            if attn_score.ndim != 2 or int(attn_score.shape[0]) != batch_size:
                raise ValueError(
                    "TileLang MLA reduced attn_score must have shape "
                    f"[batch, capacity], got {tuple(attn_score.shape)}."
                )
            if (
                attn_score.dtype != torch.float32
                or attn_score.device != q_latent.device
            ):
                raise TypeError(
                    "TileLang MLA attn_score must be FP32 on the query device, "
                    f"got {attn_score.dtype} on {attn_score.device}."
                )
            if not attn_score.is_contiguous():
                raise ValueError(
                    "TileLang MLA attn_score must be contiguous, got stride "
                    f"{tuple(attn_score.stride())}."
                )
            score_capacity = int(attn_score.shape[1])
        if score_capacity <= 0 or score_capacity % _BLOCK_N:
            raise ValueError(
                "TileLang MLA context/score capacity must be a positive "
                f"multiple of {_BLOCK_N}, got {score_capacity}."
            )
        if score_capacity > int(active_slots.shape[1]):
            raise ValueError(
                "TileLang MLA score capacity exceeds active slot width: "
                f"score={score_capacity} slots={active_slots.shape[1]}."
            )
        if not 0 < int(max_context_len) <= score_capacity:
            raise ValueError(
                "TileLang MLA max_context_len must fit the context/score "
                f"capacity, got max={max_context_len} capacity={score_capacity}."
            )
        return batch_size, score_capacity

    @torch.no_grad()
    def __call__(
        self,
        q_latent: torch.Tensor,
        q_rope: torch.Tensor,
        latent_cache: torch.Tensor,
        rope_cache: torch.Tensor,
        active_slots: torch.Tensor,
        request_indices: torch.Tensor,
        context_lens: torch.Tensor,
        output: torch.Tensor,
        *,
        attn_score: torch.Tensor | None,
        max_context_len: int,
    ) -> torch.Tensor:
        batch_size, score_capacity = self._validate(
            q_latent,
            q_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            output,
            attn_score,
            max_context_len,
        )
        need_score = attn_score is not None
        config = self._config_for(
            batch_size=batch_size,
            context_capacity=int(max_context_len),
            need_score=need_score,
        )
        key = _KernelKey(
            batch_size=batch_size,
            cache_slot_count=int(latent_cache.shape[0]),
            active_slot_rows=int(active_slots.shape[0]),
            active_slot_width=int(active_slots.shape[1]),
            score_capacity=score_capacity,
            num_split=config.num_split,
            block_h=config.block_h,
            score_mode=config.score_mode,
            need_score=need_score,
        )
        bound = self._kernels.get(key)
        if bound is None:
            bound = self._bind(key)
            self._kernels[key] = bound

        import triton

        from sparsevllm.kernels.tilelang.mla.decode import pad_glm_q_kernel

        workspace = bound.workspace
        pad_glm_q_kernel[
            (triton.cdiv(batch_size * self.padded_heads * _LATENT_DIM, 256),)
        ](
            q_latent,
            q_rope,
            workspace.padded_latent,
            workspace.padded_rope,
            q_latent.stride(0),
            q_latent.stride(1),
            q_latent.stride(2),
            q_rope.stride(0),
            q_rope.stride(1),
            q_rope.stride(2),
            batch_size=batch_size,
            valid_heads=self.valid_heads,
            padded_heads=self.padded_heads,
            latent_dim=_LATENT_DIM,
            rope_dim=_ROPE_DIM,
            BLOCK=256,
        )
        score_output = workspace.score
        if attn_score is not None:
            if config.score_mode == "partial":
                score_output.fill_(-1e20)
            else:
                score_output = attn_score.unsqueeze(1)
        bound.call(
            workspace.padded_latent,
            workspace.padded_rope,
            latent_cache,
            rope_cache,
            active_slots,
            request_indices,
            context_lens,
            workspace.glse,
            workspace.partial_output,
            output,
            score_output,
        )
        if attn_score is not None and config.score_mode == "partial":
            torch.amax(score_output, dim=1, out=attn_score)
        return output


__all__ = [
    "TileMlaDecodeKernel",
    "TileMlaLaunchConfig",
    "TileMlaWorkspace",
    "select_tile_mla_config",
    "tilelang_mla_support",
]
