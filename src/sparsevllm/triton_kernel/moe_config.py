from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class MoeGemmConfig:
    block_m: int
    block_n: int
    block_k: int
    group_m: int
    num_warps: int
    num_stages: int

    def as_triton_kwargs(self) -> dict[str, int]:
        return {
            "BLOCK_SIZE_M": self.block_m,
            "BLOCK_SIZE_N": self.block_n,
            "BLOCK_SIZE_K": self.block_k,
            "GROUP_SIZE_M": self.group_m,
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }


TOKEN_BUCKETS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


@lru_cache(maxsize=None)
def device_capability(device_type: str, device_index: int) -> tuple[int, int]:
    if device_type != "cuda":
        raise ValueError(f"MoE Triton config requires a CUDA device, got {device_type}.")
    return torch.cuda.get_device_capability(device_index)


def token_bucket(num_tokens: int) -> int:
    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
    for bucket in TOKEN_BUCKETS:
        if num_tokens <= bucket:
            return bucket
    return 2048


def _heuristic_config(
    *,
    num_tokens: int,
    top_k: int,
    output_size: int,
) -> MoeGemmConfig:
    assignments = num_tokens * top_k
    small = assignments <= 32
    return MoeGemmConfig(
        block_m=16,
        block_n=128 if small or output_size > 4096 else 64,
        block_k=32 if small else 64,
        group_m=8,
        num_warps=4,
        num_stages=4 if small else 3,
    )


# Qwen3-30B-A3B BF16 configurations tuned offline on H20 (SM90). The outer key
# is local expert count; nested keys are (token bucket, stage). Other shapes use
# the deterministic heuristic below.
_A = MoeGemmConfig(16, 64, 64, 8, 4, 3)
_B = MoeGemmConfig(16, 128, 64, 8, 4, 3)
_C = MoeGemmConfig(16, 128, 64, 8, 8, 3)
_D = MoeGemmConfig(16, 128, 32, 8, 4, 4)
_F = MoeGemmConfig(64, 64, 64, 8, 8, 3)


def _stage_table(
    w13: tuple[MoeGemmConfig, ...],
    w2: tuple[MoeGemmConfig, ...],
) -> dict[tuple[int, str], MoeGemmConfig]:
    buckets = (*TOKEN_BUCKETS, 2048)
    return {
        **{(bucket, "w13"): config for bucket, config in zip(buckets, w13)},
        **{(bucket, "w2"): config for bucket, config in zip(buckets, w2)},
    }


_H20_QWEN3_MOE_CONFIGS = {
    128: _stage_table(
        (_D, _D, _D, _A, _A, _A, _B, _B, _B, _F, _F, _F),
        (_D, _D, _D, _B, _B, _B, _B, _B, _B, _F, _F, _F),
    ),
    64: _stage_table(
        (_D, _D, _D, _C, _B, _A, _A, _B, _B, _F, _F, _F),
        (_D, _D, _D, _B, _A, _B, _A, _A, _B, _F, _F, _F),
    ),
    32: _stage_table(
        (_D, _D, _D, _A, _C, _A, _B, _B, _B, _F, _F, _F),
        (_D, _D, _D, _A, _A, _B, _C, _C, _B, _F, _F, _F),
    ),
}


@lru_cache(maxsize=None)
def _resolve_moe_gemm_config(
    dtype: torch.dtype,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    stage: str,
    device_capability: tuple[int, int],
) -> MoeGemmConfig:
    output_size = 2 * intermediate_size if stage == "w13" else hidden_size
    heuristic = _heuristic_config(
        num_tokens=num_tokens,
        top_k=top_k,
        output_size=output_size,
    )
    is_tuned_shape = (
        device_capability == (9, 0)
        and dtype == torch.bfloat16
        and top_k == 8
        and num_local_experts in {32, 64, 128}
        and hidden_size == 2048
        and intermediate_size == 768
    )
    if not is_tuned_shape:
        return heuristic
    table = _H20_QWEN3_MOE_CONFIGS[num_local_experts]
    return table.get((token_bucket(num_tokens), stage), heuristic)


def resolve_moe_gemm_config(
    *,
    dtype: torch.dtype,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    stage: str,
    device_capability: tuple[int, int] | None = None,
) -> MoeGemmConfig:
    if stage not in {"w13", "w2"}:
        raise ValueError(f"MoE GEMM stage must be 'w13' or 'w2', got {stage!r}.")
    if device_capability is None:
        device_capability = torch.cuda.get_device_capability()
    return _resolve_moe_gemm_config(
        dtype,
        int(num_tokens),
        int(top_k),
        int(num_local_experts),
        int(hidden_size),
        int(intermediate_size),
        stage,
        device_capability,
    )
