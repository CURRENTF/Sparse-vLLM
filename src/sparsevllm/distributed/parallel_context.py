from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist


def _validate_sizes(tp_size: int, ep_size: int, dp_size: int) -> tuple[int, int, int]:
    sizes = (int(tp_size), int(ep_size), int(dp_size))
    if any(size <= 0 for size in sizes):
        raise ValueError(
            "Parallel sizes must be positive, "
            f"got TP={sizes[0]}, EP={sizes[1]}, DP={sizes[2]}."
        )
    return sizes


def world_rank_from_parallel_ranks(
    dp_rank: int,
    ep_rank: int,
    tp_rank: int,
    *,
    tp_size: int,
    ep_size: int,
    dp_size: int,
) -> int:
    tp_size, ep_size, dp_size = _validate_sizes(tp_size, ep_size, dp_size)
    dp_rank, ep_rank, tp_rank = int(dp_rank), int(ep_rank), int(tp_rank)
    if not 0 <= dp_rank < dp_size:
        raise ValueError(f"dp_rank must be in [0, {dp_size}), got {dp_rank}.")
    if not 0 <= ep_rank < ep_size:
        raise ValueError(f"ep_rank must be in [0, {ep_size}), got {ep_rank}.")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank must be in [0, {tp_size}), got {tp_rank}.")
    return ((dp_rank * ep_size) + ep_rank) * tp_size + tp_rank


def parallel_ranks_from_world_rank(
    world_rank: int,
    *,
    tp_size: int,
    ep_size: int,
    dp_size: int,
) -> tuple[int, int, int]:
    tp_size, ep_size, dp_size = _validate_sizes(tp_size, ep_size, dp_size)
    world_size = tp_size * ep_size * dp_size
    world_rank = int(world_rank)
    if not 0 <= world_rank < world_size:
        raise ValueError(f"world_rank must be in [0, {world_size}), got {world_rank}.")
    dp_ep_rank, tp_rank = divmod(world_rank, tp_size)
    dp_rank, ep_rank = divmod(dp_ep_rank, ep_size)
    return dp_rank, ep_rank, tp_rank


def parallel_group_ranks(
    *,
    tp_size: int,
    ep_size: int,
    dp_size: int,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    tp_size, ep_size, dp_size = _validate_sizes(tp_size, ep_size, dp_size)

    tensor_groups = tuple(
        tuple(
            world_rank_from_parallel_ranks(
                dp_rank,
                ep_rank,
                tp_rank,
                tp_size=tp_size,
                ep_size=ep_size,
                dp_size=dp_size,
            )
            for tp_rank in range(tp_size)
        )
        for dp_rank in range(dp_size)
        for ep_rank in range(ep_size)
    )
    expert_groups = tuple(
        tuple(
            world_rank_from_parallel_ranks(
                dp_rank,
                ep_rank,
                tp_rank,
                tp_size=tp_size,
                ep_size=ep_size,
                dp_size=dp_size,
            )
            for ep_rank in range(ep_size)
        )
        for dp_rank in range(dp_size)
        for tp_rank in range(tp_size)
    )
    data_groups = tuple(
        tuple(
            world_rank_from_parallel_ranks(
                dp_rank,
                ep_rank,
                tp_rank,
                tp_size=tp_size,
                ep_size=ep_size,
                dp_size=dp_size,
            )
            for dp_rank in range(dp_size)
        )
        for ep_rank in range(ep_size)
        for tp_rank in range(tp_size)
    )
    return {
        "tensor": tensor_groups,
        "expert": expert_groups,
        "data": data_groups,
    }


@dataclass(frozen=True)
class ParallelGroup:
    process_group: dist.ProcessGroup | None
    ranks: tuple[int, ...]
    rank: int
    size: int

    def __post_init__(self) -> None:
        if self.size != len(self.ranks):
            raise ValueError(
                f"ParallelGroup size={self.size} does not match ranks={self.ranks}."
            )
        if not 0 <= self.rank < self.size:
            raise ValueError(
                f"ParallelGroup rank must be in [0, {self.size}), got {self.rank}."
            )


@dataclass(frozen=True)
class ParallelContext:
    world: ParallelGroup
    tensor: ParallelGroup
    expert: ParallelGroup
    data: ParallelGroup

    @property
    def world_rank(self) -> int:
        return self.world.rank

    @property
    def world_size(self) -> int:
        return self.world.size

    @property
    def tp_rank(self) -> int:
        return self.tensor.rank

    @property
    def tp_size(self) -> int:
        return self.tensor.size

    @property
    def ep_rank(self) -> int:
        return self.expert.rank

    @property
    def ep_size(self) -> int:
        return self.expert.size

    @property
    def dp_rank(self) -> int:
        return self.data.rank

    @property
    def dp_size(self) -> int:
        return self.data.size

    @staticmethod
    def _all_reduce(tensor: torch.Tensor, group: ParallelGroup) -> torch.Tensor:
        if group.size > 1:
            dist.all_reduce(tensor, group=group.process_group)
        return tensor

    def world_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._all_reduce(tensor, self.world)

    def tp_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._all_reduce(tensor, self.tensor)

    def ep_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._all_reduce(tensor, self.expert)

    def dp_all_reduce(self, tensor: torch.Tensor) -> torch.Tensor:
        return self._all_reduce(tensor, self.data)

    def tp_gather(self, tensor: torch.Tensor, dst: int = 0) -> list[torch.Tensor] | None:
        dst = int(dst)
        if not 0 <= dst < self.tp_size:
            raise ValueError(f"TP gather dst must be in [0, {self.tp_size}), got {dst}.")
        if self.tp_size == 1:
            return [tensor]
        gather_list = [torch.empty_like(tensor) for _ in range(self.tp_size)] if self.tp_rank == dst else None
        dist.gather(
            tensor,
            gather_list=gather_list,
            dst=self.tensor.ranks[dst],
            group=self.tensor.process_group,
        )
        return gather_list

    def world_barrier(self, *, device_ids: list[int] | None = None) -> None:
        if self.world_size > 1:
            dist.barrier(group=self.world.process_group, device_ids=device_ids)


_PARALLEL_CONTEXT: ParallelContext | None = None


def _local_group(
    groups: tuple[tuple[int, ...], ...],
    process_groups: dict[tuple[int, ...], dist.ProcessGroup | None],
    world_rank: int,
) -> ParallelGroup:
    for ranks in groups:
        if world_rank in ranks:
            return ParallelGroup(
                process_group=process_groups[ranks],
                ranks=ranks,
                rank=ranks.index(world_rank),
                size=len(ranks),
            )
    raise RuntimeError(f"No parallel group contains world rank {world_rank}.")


def init_parallel_context(
    *,
    tp_size: int,
    ep_size: int,
    dp_size: int,
) -> ParallelContext:
    global _PARALLEL_CONTEXT
    if _PARALLEL_CONTEXT is not None:
        raise RuntimeError("ParallelContext is already initialized.")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before ParallelContext.")

    tp_size, ep_size, dp_size = _validate_sizes(tp_size, ep_size, dp_size)
    expected_world_size = tp_size * ep_size * dp_size
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    if world_size != expected_world_size:
        raise ValueError(
            "Distributed world size does not match parallel configuration: "
            f"world_size={world_size}, TP={tp_size}, EP={ep_size}, DP={dp_size}."
        )

    ranks_by_dimension = parallel_group_ranks(
        tp_size=tp_size,
        ep_size=ep_size,
        dp_size=dp_size,
    )
    world_ranks = tuple(range(world_size))
    process_groups: dict[tuple[int, ...], dist.ProcessGroup | None] = {
        world_ranks: dist.group.WORLD,
    }
    for dimension in ("tensor", "expert", "data"):
        for ranks in ranks_by_dimension[dimension]:
            if ranks in process_groups:
                continue
            process_groups[ranks] = None if len(ranks) == 1 else dist.new_group(list(ranks))

    _PARALLEL_CONTEXT = ParallelContext(
        world=ParallelGroup(
            process_group=dist.group.WORLD,
            ranks=world_ranks,
            rank=world_rank,
            size=world_size,
        ),
        tensor=_local_group(ranks_by_dimension["tensor"], process_groups, world_rank),
        expert=_local_group(ranks_by_dimension["expert"], process_groups, world_rank),
        data=_local_group(ranks_by_dimension["data"], process_groups, world_rank),
    )
    return _PARALLEL_CONTEXT


def get_parallel_context() -> ParallelContext:
    if _PARALLEL_CONTEXT is None:
        raise RuntimeError("ParallelContext is not initialized.")
    return _PARALLEL_CONTEXT


def reset_parallel_context() -> None:
    global _PARALLEL_CONTEXT
    _PARALLEL_CONTEXT = None
