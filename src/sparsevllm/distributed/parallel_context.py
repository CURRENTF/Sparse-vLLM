from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
import torch.distributed as dist

from sparsevllm.distributed.topology import ParallelTopology
from sparsevllm.operators.all_reduce import AllReduceProvider, resolve_all_reduce_provider


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


def hybrid_moe_group_ranks(
    *,
    topology: ParallelTopology,
) -> dict[str, tuple[tuple[int, ...], ...]]:
    if not topology.is_outer_tp_moe:
        raise ValueError("Hybrid MoE groups require an Outer-TP MoE topology.")
    outer_tp_size = topology.attention_tp_size
    moe_ep_size = topology.expert_parallel_size
    moe_tp_size = topology.moe_tp_size
    attention_groups = (tuple(range(outer_tp_size)),)
    moe_tensor_groups = tuple(
        tuple(range(ep_rank * moe_tp_size, (ep_rank + 1) * moe_tp_size))
        for ep_rank in range(moe_ep_size)
    )
    moe_expert_groups = tuple(
        tuple(ep_rank * moe_tp_size + moe_tp_rank for ep_rank in range(moe_ep_size))
        for moe_tp_rank in range(moe_tp_size)
    )
    singleton_groups = tuple((rank,) for rank in range(outer_tp_size))
    return {
        "attention": attention_groups,
        "moe_tensor": moe_tensor_groups,
        "moe_expert": moe_expert_groups,
        "data": singleton_groups,
    }


@dataclass(frozen=True)
class ParallelGroup:
    process_group: dist.ProcessGroup | None
    ranks: tuple[int, ...]
    rank: int
    size: int
    all_reduce_provider: AllReduceProvider | None = field(default=None, compare=False, repr=False)

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
            if op != dist.ReduceOp.SUM or group.all_reduce_provider is None:
                dist.all_reduce(tensor, op=op, group=group.process_group)
            else:
                tensor = group.all_reduce_provider.run(tensor)
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

    if topology.is_outer_tp_moe:
        hybrid_groups = hybrid_moe_group_ranks(topology=topology)
        ranks_by_dimension = {
            "tensor": hybrid_groups["attention"],
            "expert": hybrid_groups["moe_expert"],
            "data": hybrid_groups["data"],
            "moe_tensor": hybrid_groups["moe_tensor"],
        }
    else:
        ranks_by_dimension = parallel_group_ranks(
            tp_size=tp_size,
            ep_size=ep_size,
            dp_size=dp_size,
        )
        ranks_by_dimension["moe_tensor"] = ranks_by_dimension["tensor"]
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
    providers: dict[tuple[int, ...], AllReduceProvider] = {}

    def bind_provider(group: ParallelGroup) -> ParallelGroup:
        if group.size == 1:
            return group
        provider = providers.get(group.ranks)
        if provider is None:
            provider = providers[group.ranks] = resolve_all_reduce_provider(
                group.process_group, group.size
            )
        return replace(group, all_reduce_provider=provider)

    _PARALLEL_CONTEXT = ParallelContext(
        world=bind_provider(context.world),
        tensor=bind_provider(context.tensor),
        expert=bind_provider(context.expert),
        data=bind_provider(context.data),
        moe_tensor=bind_provider(context.moe_tensor or context.tensor),
    )
    return _PARALLEL_CONTEXT


def get_parallel_context() -> ParallelContext:
    if _PARALLEL_CONTEXT is None:
        raise RuntimeError("ParallelContext is not initialized.")
    return _PARALLEL_CONTEXT


def reset_parallel_context() -> None:
    global _PARALLEL_CONTEXT
    _PARALLEL_CONTEXT = None
