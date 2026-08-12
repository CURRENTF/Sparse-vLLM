from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParallelMode(str, Enum):
    STANDARD = "standard"
    OUTER_TP_MOE = "outer_tp_moe_tp_ep"


@dataclass(frozen=True)
class ParallelTopology:
    tensor_parallel_size: int
    expert_parallel_size: int
    data_parallel_size: int
    mode: ParallelMode = ParallelMode.STANDARD

    def __post_init__(self) -> None:
        for field in (
            "tensor_parallel_size",
            "expert_parallel_size",
            "data_parallel_size",
        ):
            object.__setattr__(self, field, int(getattr(self, field)))
        object.__setattr__(self, "mode", ParallelMode(self.mode))
        sizes = (
            self.tensor_parallel_size,
            self.expert_parallel_size,
            self.data_parallel_size,
        )
        if any(int(size) <= 0 for size in sizes):
            raise ValueError(
                "Parallel sizes must be positive, "
                f"got TP={sizes[0]}, EP={sizes[1]}, DP={sizes[2]}."
            )
        if self.mode is ParallelMode.OUTER_TP_MOE:
            if self.data_parallel_size != 1:
                raise ValueError(
                    "Outer-TP MoE parallelism requires DP=1, "
                    f"got DP={self.data_parallel_size}."
                )
            if self.tensor_parallel_size % self.expert_parallel_size:
                raise ValueError(
                    "Outer-TP MoE requires TP divisible by EP, "
                    f"got TP={self.tensor_parallel_size}, EP={self.expert_parallel_size}."
                )

    @property
    def is_outer_tp_moe(self) -> bool:
        return self.mode is ParallelMode.OUTER_TP_MOE

    @property
    def attention_tp_size(self) -> int:
        return self.tensor_parallel_size

    @property
    def moe_tp_size(self) -> int:
        return (
            self.tensor_parallel_size // self.expert_parallel_size
            if self.is_outer_tp_moe
            else self.tensor_parallel_size
        )

    @property
    def world_size(self) -> int:
        return (
            self.tensor_parallel_size
            if self.is_outer_tp_moe
            else self.tensor_parallel_size
            * self.expert_parallel_size
            * self.data_parallel_size
        )
