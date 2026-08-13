from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist

from sparsevllm.distributed.topology import ParallelTopology, parallel_group_ranks
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
    moe_tensor: ParallelGroup | None = None

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
    def attention(self) -> ParallelGroup:
        return self.tensor

    @property
    def attention_tp_rank(self) -> int:
        return self.attention.rank

    @property
    def attention_tp_size(self) -> int:
        return self.attention.size

    @property
    def ep_rank(self) -> int:
        return self.expert.rank

    @property
    def ep_size(self) -> int:
        return self.expert.size

    @property
    def moe_tp_rank(self) -> int:
        return (self.moe_tensor or self.tensor).rank

    @property
    def moe_tp_size(self) -> int:
        return (self.moe_tensor or self.tensor).size

    @property
    def dp_rank(self) -> int:
        return self.data.rank

    @property
    def dp_size(self) -> int:
        return self.data.size

    @staticmethod
    def _all_reduce(
        tensor: torch.Tensor,
        group: ParallelGroup,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        if group.size > 1:
            dist.all_reduce(tensor, op=op, group=group.process_group)
        return tensor

    def world_all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        return self._all_reduce(tensor, self.world, op)

    def attention_tp_all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        return self._all_reduce(tensor, self.attention, op)

    def tp_all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        return self.attention_tp_all_reduce(tensor, op)

    def ep_all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        return self._all_reduce(tensor, self.expert, op)

    def moe_tp_all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        return self._all_reduce(tensor, self.moe_tensor or self.tensor, op)

    def ep_broadcast(
        self,
        tensor: torch.Tensor,
        *,
        src_ep_rank: int = 0,
    ) -> torch.Tensor:
        src_ep_rank = int(src_ep_rank)
        if not 0 <= src_ep_rank < self.ep_size:
            raise ValueError(
                f"EP broadcast source must be in [0, {self.ep_size}), got {src_ep_rank}."
            )
        if self.ep_size > 1:
            dist.broadcast(
                tensor,
                src=self.expert.ranks[src_ep_rank],
                group=self.expert.process_group,
            )
        return tensor

    def dp_all_reduce(
        self,
        tensor: torch.Tensor,
        op: dist.ReduceOp = dist.ReduceOp.SUM,
    ) -> torch.Tensor:
        return self._all_reduce(tensor, self.data, op)

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
    topology: ParallelTopology,
) -> ParallelContext:
    global _PARALLEL_CONTEXT
    if _PARALLEL_CONTEXT is not None:
        raise RuntimeError("ParallelContext is already initialized.")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before ParallelContext.")

    tp_size = topology.tensor_parallel_size
    ep_size = topology.expert_parallel_size
    dp_size = topology.data_parallel_size
    expected_world_size = topology.world_size
    world_size = dist.get_world_size()
    world_rank = dist.get_rank()
    if world_size != expected_world_size:
        raise ValueError(
            "Distributed world size does not match parallel configuration: "
            f"world_size={world_size}, TP={tp_size}, EP={ep_size}, DP={dp_size}."
        )

    ranks_by_dimension = parallel_group_ranks(topology)
    world_ranks = tuple(range(world_size))
    process_groups: dict[tuple[int, ...], dist.ProcessGroup | None] = {
        world_ranks: dist.group.WORLD,
    }
    for dimension in ("tensor", "expert", "data", "moe_tensor"):
        for ranks in ranks_by_dimension[dimension]:
            if ranks in process_groups:
                continue
            process_groups[ranks] = None if len(ranks) == 1 else dist.new_group(list(ranks))

    context = ParallelContext(
        world=ParallelGroup(
            process_group=dist.group.WORLD,
            ranks=world_ranks,
            rank=world_rank,
            size=world_size,
        ),
        tensor=_local_group(ranks_by_dimension["tensor"], process_groups, world_rank),
        expert=_local_group(ranks_by_dimension["expert"], process_groups, world_rank),
        data=_local_group(ranks_by_dimension["data"], process_groups, world_rank),
        moe_tensor=_local_group(
            ranks_by_dimension["moe_tensor"], process_groups, world_rank
        ),
    )
    _PARALLEL_CONTEXT = context
    return _PARALLEL_CONTEXT


def get_parallel_context() -> ParallelContext:
    if _PARALLEL_CONTEXT is None:
        raise RuntimeError("ParallelContext is not initialized.")
    return _PARALLEL_CONTEXT


def reset_parallel_context() -> None:
    global _PARALLEL_CONTEXT
    _PARALLEL_CONTEXT = None
