"""Model-path, parallelism, and platform-alias normalization."""

import os

from sparsevllm.configs.common import _coerce_bool_config


def _normalize_platform_aliases(config) -> None:
    if config.device_memory_utilization is not None:
        config.gpu_memory_utilization = float(config.device_memory_utilization)
    config.device_memory_utilization = float(config.gpu_memory_utilization)

    if config.decode_graph is not None:
        config.decode_cuda_graph = _coerce_bool_config("decode_graph", config.decode_graph)
    else:
        config.decode_cuda_graph = _coerce_bool_config(
            "decode_cuda_graph",
            config.decode_cuda_graph,
        )
    config.decode_graph = bool(config.decode_cuda_graph)

    if config.decode_graph_capture_sampling is not None:
        config.decode_cuda_graph_capture_sampling = _coerce_bool_config(
            "decode_graph_capture_sampling",
            config.decode_graph_capture_sampling,
        )
    else:
        config.decode_cuda_graph_capture_sampling = _coerce_bool_config(
            "decode_cuda_graph_capture_sampling",
            config.decode_cuda_graph_capture_sampling,
        )
    config.decode_graph_capture_sampling = bool(config.decode_cuda_graph_capture_sampling)

    if config.decode_graph_capture_sizes is not None:
        config.decode_cuda_graph_capture_sizes = config.decode_graph_capture_sizes


def normalize_platform(config) -> None:
    if not os.path.isdir(config.model):
        raise FileNotFoundError(f"Model directory does not exist: {config.model}")
    config.tensor_parallel_size = int(config.tensor_parallel_size)
    config.expert_parallel_size = int(config.expert_parallel_size)
    config.data_parallel_size = int(config.data_parallel_size)
    config.weight_loading_workers = int(config.weight_loading_workers)
    if not 1 <= config.tensor_parallel_size <= 8:
        raise ValueError(f"tensor_parallel_size must be in [1, 8], got {config.tensor_parallel_size}.")
    if config.expert_parallel_size <= 0:
        raise ValueError(
            f"expert_parallel_size must be positive, got {config.expert_parallel_size}."
        )
    if config.data_parallel_size <= 0:
        raise ValueError(
            f"data_parallel_size must be positive, got {config.data_parallel_size}."
        )
    if config.weight_loading_workers <= 0:
        raise ValueError(
            "weight_loading_workers must be positive, "
            f"got {config.weight_loading_workers}."
        )
    _normalize_platform_aliases(config)
