from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class ParallelMode(str, Enum):
    STANDARD = "standard"
    OUTER_TP_MOE = "outer_tp_moe_tp_ep"
    DPA_EP = "dpa_ep"


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
        if any(size <= 0 for size in sizes):
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
        elif self.mode is ParallelMode.DPA_EP and (
            self.tensor_parallel_size != 1
            or self.expert_parallel_size != self.data_parallel_size
        ):
            raise ValueError(
                "DPA+EP requires TP=1 and matching DP/EP sizes, "
                f"got TP={self.tensor_parallel_size}, "
                f"EP={self.expert_parallel_size}, DP={self.data_parallel_size}."
            )

    @property
    def is_outer_tp_moe(self) -> bool:
        return self.mode is ParallelMode.OUTER_TP_MOE

    @property
    def is_dpa_ep(self) -> bool:
        return self.mode is ParallelMode.DPA_EP

    @property
    def attention_tp_size(self) -> int:
        return 1 if self.is_dpa_ep else self.tensor_parallel_size

    @property
    def moe_tp_size(self) -> int:
        if self.is_dpa_ep:
            return 1
        return (
            self.tensor_parallel_size // self.expert_parallel_size
            if self.is_outer_tp_moe
            else self.tensor_parallel_size
        )

    @property
    def world_size(self) -> int:
        if self.is_dpa_ep:
            return self.data_parallel_size
        return (
            self.tensor_parallel_size
            if self.is_outer_tp_moe
            else self.tensor_parallel_size
            * self.expert_parallel_size
            * self.data_parallel_size
        )


def world_rank_from_parallel_ranks(
    topology: ParallelTopology,
    dp_rank: int,
    ep_rank: int,
    tp_rank: int,
) -> int:
    if topology.mode is not ParallelMode.STANDARD:
        raise ValueError(
            f"{topology.mode.value} does not use standard DP/EP/TP rank mapping."
        )
    dp_rank, ep_rank, tp_rank = int(dp_rank), int(ep_rank), int(tp_rank)
    for name, rank, size in (
        ("dp_rank", dp_rank, topology.data_parallel_size),
        ("ep_rank", ep_rank, topology.expert_parallel_size),
        ("tp_rank", tp_rank, topology.tensor_parallel_size),
    ):
        if not 0 <= rank < size:
            raise ValueError(f"{name} must be in [0, {size}), got {rank}.")
    return (
        (dp_rank * topology.expert_parallel_size + ep_rank)
        * topology.tensor_parallel_size
        + tp_rank
    )


def parallel_ranks_from_world_rank(
    topology: ParallelTopology,
    world_rank: int,
) -> tuple[int, int, int]:
    if topology.mode is not ParallelMode.STANDARD:
        raise ValueError(
            f"{topology.mode.value} does not use standard DP/EP/TP rank mapping."
        )
    world_rank = int(world_rank)
    if not 0 <= world_rank < topology.world_size:
        raise ValueError(
            f"world_rank must be in [0, {topology.world_size}), got {world_rank}."
        )
    dp_ep_rank, tp_rank = divmod(world_rank, topology.tensor_parallel_size)
    dp_rank, ep_rank = divmod(dp_ep_rank, topology.expert_parallel_size)
    return dp_rank, ep_rank, tp_rank


def _standard_group_ranks(
    topology: ParallelTopology,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    tp_size = topology.tensor_parallel_size
    ep_size = topology.expert_parallel_size
    dp_size = topology.data_parallel_size

    def world_rank(dp_rank: int, ep_rank: int, tp_rank: int) -> int:
        return world_rank_from_parallel_ranks(topology, dp_rank, ep_rank, tp_rank)

    tensor_groups = tuple(
        tuple(world_rank(dp_rank, ep_rank, tp_rank) for tp_rank in range(tp_size))
        for dp_rank in range(dp_size)
        for ep_rank in range(ep_size)
    )
    return {
        "tensor": tensor_groups,
        "expert": tuple(
            tuple(world_rank(dp_rank, ep_rank, tp_rank) for ep_rank in range(ep_size))
            for dp_rank in range(dp_size)
            for tp_rank in range(tp_size)
        ),
        "data": tuple(
            tuple(world_rank(dp_rank, ep_rank, tp_rank) for dp_rank in range(dp_size))
            for ep_rank in range(ep_size)
            for tp_rank in range(tp_size)
        ),
        "moe_tensor": tensor_groups,
    }


def _outer_tp_moe_group_ranks(
    topology: ParallelTopology,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    outer_tp_size = topology.attention_tp_size
    moe_ep_size = topology.expert_parallel_size
    moe_tp_size = topology.moe_tp_size
    return {
        "tensor": (tuple(range(outer_tp_size)),),
        "expert": tuple(
            tuple(
                ep_rank * moe_tp_size + moe_tp_rank
                for ep_rank in range(moe_ep_size)
            )
            for moe_tp_rank in range(moe_tp_size)
        ),
        "data": tuple((rank,) for rank in range(outer_tp_size)),
        "moe_tensor": tuple(
            tuple(range(ep_rank * moe_tp_size, (ep_rank + 1) * moe_tp_size))
            for ep_rank in range(moe_ep_size)
        ),
    }


def _dpa_ep_group_ranks(
    topology: ParallelTopology,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    world = tuple(range(topology.world_size))
    singletons = tuple((rank,) for rank in world)
    return {
        "tensor": singletons,
        "expert": (world,),
        "data": (world,),
        "moe_tensor": singletons,
    }


def parallel_group_ranks(
    topology: ParallelTopology,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    if topology.is_dpa_ep:
        return _dpa_ep_group_ranks(topology)
    if topology.is_outer_tp_moe:
        return _outer_tp_moe_group_ranks(topology)
    return _standard_group_ranks(topology)
