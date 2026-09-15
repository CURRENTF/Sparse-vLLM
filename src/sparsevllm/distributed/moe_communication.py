"""Token transport around local expert execution.

Transport owns token layout and reduction; expert providers keep their existing
token-major input and local-expert weight contracts. Routing after an AG avoids
two small collectives for route IDs and weights. An expert-directed transport
can instead route before dispatch without changing the local expert callable.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.distributed as dist

from sparsevllm.distributed.parallel_context import ParallelContext
from sparsevllm.operators.agrs import prepare_parallel_agrs
from sparsevllm.utils.context import get_context


@dataclass(frozen=True)
class MoeDispatch:
    hidden_states: torch.Tensor
    local_rows: int
    capacity: int
    routing_metadata: torch.Tensor | None = None


class MoeCommunication:
    """Execution boundary that lets transport choose where routing happens.

    Replicated dispatch routes after dispatch. An expert-directed backend can
    override this operation to route before dispatch, while retaining the
    model's router and compatible local expert provider.
    """

    def _run(self, hidden_states, *, route, experts, chunk_size, capacity, routing_metadata=None):
        dispatch = self.dispatch(hidden_states, capacity=capacity, routing_metadata=routing_metadata)
        outputs = []
        for index, chunk in enumerate(dispatch.hidden_states.split(chunk_size, dim=0)):
            if dispatch.routing_metadata is None:
                ids, weights = route(chunk)
            else:
                metadata = dispatch.routing_metadata[index * chunk_size:index * chunk_size + len(chunk)]
                ids, weights = route(chunk, metadata)
            outputs.append(experts(chunk, ids, weights))
        output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, dim=0)
        return self.combine(output, dispatch)

    def _finish(self, output):
        return output

    def run(self, hidden_states, *, shared_experts=None, capacity=None, routing_metadata=None, **kwargs):
        if routing_metadata is not None and (
            routing_metadata.ndim != 2 or routing_metadata.shape[0] != len(hidden_states)
            or routing_metadata.shape[1] == 0 or routing_metadata.device != hidden_states.device
            or routing_metadata.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("MoE routing metadata must be integer rows aligned with local tokens.")
        if capacity is None:
            capacity = get_context().moe_token_capacity
            if capacity is None:
                # Startup warmup/capture uses identical shapes on every replica.
                capacity = hidden_states.shape[0]
        output = self._run(hidden_states, capacity=capacity, routing_metadata=routing_metadata, **kwargs)
        if shared_experts is not None and len(hidden_states):
            output = output + shared_experts(hidden_states)
        return self._finish(output)

    def run_with_shared_experts(self, hidden_states, *, shared_experts, **kwargs):
        """Sum routed and shared partials before completing the TP reduction."""
        return self.run(hidden_states, shared_experts=shared_experts, **kwargs)


class AllReduceMoeCommunication(MoeCommunication):
    """Replicated tokens, partial expert outputs, replicated reduced output."""

    def __init__(self, reduce: Callable[[torch.Tensor], torch.Tensor]):
        self._reduce = reduce

    def dispatch(
        self, hidden_states: torch.Tensor, *, capacity: int | None = None, routing_metadata=None
    ) -> MoeDispatch:
        return MoeDispatch(
            hidden_states, hidden_states.shape[0], hidden_states.shape[0], routing_metadata
        )

    def combine(
        self, output: torch.Tensor, dispatch: MoeDispatch | None = None
    ) -> torch.Tensor:
        return self._reduce(output)


class AllGatherReduceScatterMoeCommunication(MoeCommunication):
    """Gather DP tokens per TP lane, scatter partials, then complete the TP sum.

    The runner agrees on capacity once per step, outside the layer loop. Equal
    padded rank segments make NCCL AG/RS capturable; slicing restores the local
    token order. Padding never enters attention or persistent request state.
    """

    def __init__(
        self, parallel_context: ParallelContext, *, max_rows=None, hidden_size=None,
        dtype=None, reduce=None, reduction_dtype=None,
    ):
        if parallel_context.moe_tp_size != 1:
            raise ValueError("AG/RS currently requires MoE TP=1 (EP=world size).")
        self.group = parallel_context.attn_dp
        self._reduce = reduce or parallel_context.attn_tp.all_reduce
        self.max_rows = max_rows
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.reduction_dtype = reduction_dtype or dtype
        self.op = None
        self.reduction_op = None
        self.closed = False

    def _finish(self, output):
        # RS sums experts on one TP lane; complete the other lanes locally.
        # Idle replicas have no output rows and need no TP collective.
        return self._reduce(output) if len(output) else output

    @property
    def name(self):
        return "torch_distributed_agrs" if self.op is None else self.op.name

    def prepare(self, *, device_index, cuda_graph):
        if self.closed or self.op is not None:
            raise RuntimeError("AG/RS transport can only be prepared once.")
        self.op = prepare_parallel_agrs(
            self.group, max_rows=self.max_rows, hidden_size=self.hidden_size,
            dtype=self.dtype, device_index=device_index,
        )
        if self.reduction_dtype != self.dtype:
            self.reduction_op = prepare_parallel_agrs(
                self.group, max_rows=self.max_rows, hidden_size=self.hidden_size,
                dtype=self.reduction_dtype, device_index=device_index,
            )

    def close(self):
        if self.reduction_op is not None:
            self.reduction_op.close()
            self.reduction_op = None
        if self.op is not None:
            self.op.close()
            self.op = None
        self.closed = True

    def _use_prepared(self, capacity):
        if self.closed:
            raise RuntimeError("AG/RS transport is closed.")
        if self.max_rows is not None and self.op is None:
            raise RuntimeError("AG/RS transport is not prepared.")
        # Only the agreed capacity can select the collective. Local phases may
        # differ (e.g. a one-token prefill opposite a decode or idle rank).
        return (
            self.op is not None
            and capacity <= self.max_rows
        )

    def dispatch(self, hidden_states: torch.Tensor, *, capacity: int, routing_metadata=None) -> MoeDispatch:
        rows, hidden = hidden_states.shape
        if capacity < rows or capacity <= 0:
            raise ValueError(
                f"MoE capacity {capacity} cannot hold {rows} local tokens."
            )
        if rows == capacity:
            local = hidden_states.contiguous()
        else:
            local = hidden_states.new_zeros((capacity, hidden))
            local[:rows].copy_(hidden_states)
        gathered = hidden_states.new_empty((capacity * self.group.size, hidden))
        if self._use_prepared(capacity):
            self.op.all_gather(gathered, local)
        elif self.group.size == 1:
            gathered.copy_(local)
        else:
            dist.all_gather_into_tensor(gathered, local, group=self.group.process_group)
        gathered_metadata = None
        if routing_metadata is not None:
            local_metadata = routing_metadata.new_full((capacity, routing_metadata.shape[1]), -1)
            local_metadata[:rows].copy_(routing_metadata)
            gathered_metadata = routing_metadata.new_empty((capacity * self.group.size, routing_metadata.shape[1]))
            if self.group.size == 1:
                gathered_metadata.copy_(local_metadata)
            else:
                dist.all_gather_into_tensor(gathered_metadata, local_metadata, group=self.group.process_group)
        return MoeDispatch(gathered, rows, capacity, gathered_metadata)

    def combine(self, output: torch.Tensor, dispatch: MoeDispatch) -> torch.Tensor:
        if output.shape[0] != dispatch.capacity * self.group.size:
            raise ValueError(
                "Local expert output does not match the dispatched token layout."
            )
        local = output.new_empty((dispatch.capacity, output.shape[1]))
        if self._use_prepared(dispatch.capacity):
            (self.reduction_op or self.op).reduce_scatter(local, output.contiguous())
        elif self.group.size == 1:
            local.copy_(output)
        else:
            dist.reduce_scatter_tensor(
                local, output.contiguous(), group=self.group.process_group
            )
        return local[: dispatch.local_rows]


def prepare_moe_communication(parallel_context: ParallelContext, collectives=None):
    if parallel_context.attn_dp_size > 1:
        if collectives is not None:
            return collectives.moe_transport
        return AllGatherReduceScatterMoeCommunication(parallel_context)
    return AllReduceMoeCommunication(
        collectives.moe.run
        if collectives is not None
        else parallel_context.world.all_reduce
    )
