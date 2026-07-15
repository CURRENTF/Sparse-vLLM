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


@dataclass(frozen=True)
class MoeGemmShape:
    hardware: str
    capability: tuple[int, int]
    dtype: torch.dtype
    top_k: int
    num_local_experts: int
    hidden_size: int
    intermediate_size: int


TUNED_TOKEN_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048)


@lru_cache(maxsize=None)
def device_info(
    device_type: str,
    device_index: int,
) -> tuple[str, tuple[int, int]]:
    if device_type != "cuda":
        raise ValueError(f"MoE Triton config requires a CUDA device, got {device_type}.")
    return (
        torch.cuda.get_device_name(device_index),
        torch.cuda.get_device_capability(device_index),
    )


def _hardware_family(device_name: str) -> str:
    name = device_name.upper()
    if "H20" in name:
        return "H20"
    return name


def token_bucket(num_tokens: int) -> int:
    num_tokens = int(num_tokens)
    if num_tokens <= 0:
        raise ValueError(f"num_tokens must be positive, got {num_tokens}.")
    return min(TUNED_TOKEN_COUNTS, key=lambda value: abs(value - num_tokens))


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


_A = MoeGemmConfig(16, 64, 64, 8, 4, 3)
_B = MoeGemmConfig(16, 128, 64, 8, 4, 3)
_C = MoeGemmConfig(16, 128, 64, 8, 8, 3)
_D = MoeGemmConfig(16, 128, 32, 8, 4, 4)
_F = MoeGemmConfig(64, 64, 64, 8, 8, 3)


def _stage_table(
    w13: tuple[MoeGemmConfig, ...],
    w2: tuple[MoeGemmConfig, ...],
) -> dict[str, dict[int, MoeGemmConfig]]:
    if len(w13) != len(TUNED_TOKEN_COUNTS) or len(w2) != len(TUNED_TOKEN_COUNTS):
        raise ValueError("Each tuned MoE stage must cover every token count.")
    return {
        "w13": dict(zip(TUNED_TOKEN_COUNTS, w13)),
        "w2": dict(zip(TUNED_TOKEN_COUNTS, w2)),
    }


# Qwen3-30B-A3B BF16 profiles tuned offline on H20. Profiles are keyed by the
# kernel-relevant hardware and GEMM shape rather than by model name.
_TUNED_CONFIGS = {
    MoeGemmShape("H20", (9, 0), torch.bfloat16, 8, 128, 2048, 768): _stage_table(
        (_D, _D, _D, _A, _A, _A, _B, _B, _B, _F, _F, _F),
        (_D, _D, _D, _B, _B, _B, _B, _B, _B, _F, _F, _F),
    ),
    MoeGemmShape("H20", (9, 0), torch.bfloat16, 8, 64, 2048, 768): _stage_table(
        (_D, _D, _D, _C, _B, _A, _A, _B, _B, _F, _F, _F),
        (_D, _D, _D, _B, _A, _B, _A, _A, _B, _F, _F, _F),
    ),
    MoeGemmShape("H20", (9, 0), torch.bfloat16, 8, 32, 2048, 768): _stage_table(
        (_D, _D, _D, _A, _C, _A, _B, _B, _B, _F, _F, _F),
        (_D, _D, _D, _A, _A, _B, _C, _C, _B, _F, _F, _F),
    ),
    MoeGemmShape("NVIDIA H100 80GB HBM3", (9, 0), torch.bfloat16, 8, 128, 2048, 768): _stage_table(
        (_D, _A, _A, _A, _A, _A, _A, _A, _F, _F, _F, _F),
        (_D, _A, _A, _A, _A, _A, _A, _A, _A, _A, _C, _C),
    ),
    MoeGemmShape("NVIDIA H100 80GB HBM3", (9, 0), torch.bfloat16, 8, 64, 2048, 768): _stage_table(
        (_A, _A, _A, _A, _A, _F, _F, _F, _F, _F, _F, _F),
        (_A, _A, _A, _A, _A, _A, _A, _A, _A, _A, _B, _B),
    ),
    MoeGemmShape("NVIDIA H100 80GB HBM3", (9, 0), torch.bfloat16, 8, 32, 2048, 768): _stage_table(
        (_A, _D, _D, _A, _A, _A, _A, _A, _A, _A, _F, _F),
        (_A, _D, _D, _A, _A, _A, _A, _A, _A, _A, _A, _B),
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
    device_name: str,
    capability: tuple[int, int],
) -> MoeGemmConfig:
    shape = MoeGemmShape(
        hardware=_hardware_family(device_name),
        capability=capability,
        dtype=dtype,
        top_k=top_k,
        num_local_experts=num_local_experts,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
    )
    table = _TUNED_CONFIGS.get(shape)
    if table is not None:
        return table[stage][token_bucket(num_tokens)]

    output_size = 2 * intermediate_size if stage == "w13" else hidden_size
    return _heuristic_config(
        num_tokens=num_tokens,
        top_k=top_k,
        output_size=output_size,
    )


def resolve_moe_gemm_config(
    *,
    dtype: torch.dtype,
    num_tokens: int,
    top_k: int,
    num_local_experts: int,
    hidden_size: int,
    intermediate_size: int,
    stage: str,
    device_name: str | None = None,
    device_capability: tuple[int, int] | None = None,
) -> MoeGemmConfig:
    if stage not in {"w13", "w2"}:
        raise ValueError(f"MoE GEMM stage must be 'w13' or 'w2', got {stage!r}.")
    if device_name is None:
        device_name = torch.cuda.get_device_name()
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
        device_name,
        device_capability,
    )
