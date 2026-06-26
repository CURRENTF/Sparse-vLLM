from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch.distributed as dist


@dataclass(frozen=True)
class ParallelContext:
    global_rank: int
    global_world_size: int
    tp_size: int
    tp_rank: int
    ep_size: int
    ep_rank: int
    local_rank: int
    tp_group: object | None = None
    ep_group: object | None = None

    @property
    def is_global_rank0(self) -> bool:
        return self.global_rank == 0


_CURRENT_PARALLEL_CONTEXT: ParallelContext | None = None


def _dist_world_group_or_none(size: int) -> object | None:
    if size <= 1:
        return None
    if not dist.is_available() or not dist.is_initialized():
        return None
    return dist.group.WORLD


def build_parallel_context(config, global_rank: int) -> ParallelContext:
    tp_size = int(getattr(config, "tensor_parallel_size", 1))
    ep_size = int(getattr(config, "expert_parallel_size", 1))
    global_world_size = tp_size * ep_size
    global_rank = int(global_rank)
    if global_rank < 0 or global_rank >= global_world_size:
        raise ValueError(
            "global_rank must be in launched process world: "
            f"rank={global_rank} world_size={global_world_size}."
        )

    tp_rank = global_rank % tp_size
    ep_rank = global_rank // tp_size

    # TP+EP hybrid is rejected by Config in v1. For the supported layouts,
    # the active multi-rank group is the process world and singleton groups are unused.
    tp_group = _dist_world_group_or_none(tp_size)
    ep_group = _dist_world_group_or_none(ep_size)
    return ParallelContext(
        global_rank=global_rank,
        global_world_size=global_world_size,
        tp_size=tp_size,
        tp_rank=tp_rank,
        ep_size=ep_size,
        ep_rank=ep_rank,
        local_rank=global_rank,
        tp_group=tp_group,
        ep_group=ep_group,
    )


def default_parallel_context() -> ParallelContext:
    if dist.is_available() and dist.is_initialized():
        rank = int(dist.get_rank())
        world_size = int(dist.get_world_size())
        return ParallelContext(
            global_rank=rank,
            global_world_size=world_size,
            tp_size=world_size,
            tp_rank=rank,
            ep_size=1,
            ep_rank=0,
            local_rank=rank,
            tp_group=_dist_world_group_or_none(world_size),
            ep_group=None,
        )
    return ParallelContext(
        global_rank=0,
        global_world_size=1,
        tp_size=1,
        tp_rank=0,
        ep_size=1,
        ep_rank=0,
        local_rank=0,
        tp_group=None,
        ep_group=None,
    )


def set_parallel_context(context: ParallelContext) -> None:
    global _CURRENT_PARALLEL_CONTEXT
    _CURRENT_PARALLEL_CONTEXT = context


def reset_parallel_context() -> None:
    global _CURRENT_PARALLEL_CONTEXT
    _CURRENT_PARALLEL_CONTEXT = None


def get_parallel_context() -> ParallelContext:
    if _CURRENT_PARALLEL_CONTEXT is not None:
        return _CURRENT_PARALLEL_CONTEXT
    return default_parallel_context()


@contextmanager
def parallel_context_scope(context: ParallelContext) -> Iterator[None]:
    global _CURRENT_PARALLEL_CONTEXT
    previous = _CURRENT_PARALLEL_CONTEXT
    set_parallel_context(context)
    try:
        yield
    finally:
        _CURRENT_PARALLEL_CONTEXT = previous


def get_tp_size() -> int:
    return get_parallel_context().tp_size


def get_tp_rank() -> int:
    return get_parallel_context().tp_rank


def get_tp_group() -> object | None:
    return get_parallel_context().tp_group


def get_ep_size() -> int:
    return get_parallel_context().ep_size


def get_ep_rank() -> int:
    return get_parallel_context().ep_rank


def get_ep_group() -> object | None:
    return get_parallel_context().ep_group
