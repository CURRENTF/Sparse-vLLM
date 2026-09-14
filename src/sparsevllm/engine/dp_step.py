"""One host coordination point per DP forward, outside captured model work."""

from __future__ import annotations

import torch.distributed as dist

from sparsevllm.utils.context import get_context


def coordinate_dp_step(runner, seqs, is_prefill: bool) -> bool:
    graph_runner = runner.decode_graph_runner
    rows = sum(seq.current_chunk_size for seq in seqs) if is_prefill else len(seqs)
    decode_capacity = (
        graph_runner._select_graph_batch_size(len(seqs))
        if seqs and not is_prefill
        else 0
    )
    force_eager = getattr(runner.cache_manager, "decode_graph_force_eager", None)
    control = runner.dp_control_buffer
    control[0] = max(rows, decode_capacity)
    control[1] = bool(seqs and (is_prefill or (force_eager and force_eager())))
    dist.all_reduce(control, op=dist.ReduceOp.MAX, group=runner.dp_control_group)
    capacity, eager = int(control[0]), bool(control[1])
    get_context().moe_token_capacity = capacity
    graph_runner.dp_batch_capacity = None if eager else capacity
    return eager
