from sparsevllm.distributed.parallel_context import ParallelContext


def engine_process_title(parallel_context: ParallelContext) -> str:
    if parallel_context.world_size == 1:
        return "SVLLM_Engine"

    title = f"SVLLM_TP{parallel_context.tp_rank}_EP{parallel_context.ep_rank}"
    if parallel_context.dp_size > 1:
        title += f"_DP{parallel_context.dp_rank}"
    return title


def set_engine_process_title(parallel_context: ParallelContext) -> None:
    import setproctitle

    setproctitle.setproctitle(engine_process_title(parallel_context))
