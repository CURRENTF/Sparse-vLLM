"""Resolve token transport independently of expert compute providers."""


def normalize_moe_communication(config):
    backend = str(config.moe_communication_backend).strip().lower()
    if backend not in ("auto", "agrs", "all2all"):
        raise ValueError("moe_communication_backend must be auto, agrs or all2all.")
    dp = config.data_parallel_size > 1
    if backend != "auto" and not dp:
        raise ValueError(
            "Explicit MoE token transport requires DP attention (TP=1, EP=DP)."
        )
    config.moe_communication_backend = backend
    config.resolved_moe_communication_backend = (
        ("agrs" if dp else "allreduce") if backend == "auto" else backend
    )
    if backend == "all2all":
        if config.expert_parallel_size not in (2, 4, 8):
            raise ValueError("DeepEP V1 normal NVLink requires EP=DP in {2, 4, 8}.")
        # Only the selected transport imports/checks the optional dependency.
        from sparsevllm.operators.all2all import check_all2all_dependency

        check_all2all_dependency()
