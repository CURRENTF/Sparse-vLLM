"""Validate the MoE token transport selected for the parallel topology."""


def validate_moe_backend(config):
    dp = config.data_parallel_size > 1
    if config.moe_backend is None:
        config.moe_backend = "agrs" if dp else "all-reduce"
    backend = config.moe_backend
    if backend not in ("all-reduce", "agrs", "deepepv1"):
        raise ValueError("moe_backend must be all-reduce, agrs or deepepv1.")
    if dp and backend == "all-reduce":
        raise ValueError("DP attention requires moe_backend=agrs or deepepv1.")
    if not dp and backend != "all-reduce":
        raise ValueError("AG/RS and DeepEP require DP attention (TP=1, EP=DP).")
    if backend == "deepepv1":
        if config.expert_parallel_size not in (2, 4, 8):
            raise ValueError("DeepEP V1 normal NVLink requires EP=DP in {2, 4, 8}.")
        # Only the selected transport imports/checks the optional dependency.
        from sparsevllm.operators.all2all import check_all2all_dependency

        check_all2all_dependency()
