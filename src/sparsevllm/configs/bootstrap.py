"""Environment-controlled bootstrap configuration."""

import os

def normalize_bootstrap(config) -> None:
    if os.getenv("PROFILER_SVLLM"):
        config.enable_profiler = True
    if config.tiny_random or os.getenv("SPARSEVLLM_TINY_RANDOM") is not None:
        from sparsevllm.debug.tiny_random import resolve_tiny_random_settings

        (
            config.tiny_random,
            config.tiny_random_config,
            config.tiny_random_seed,
        ) = resolve_tiny_random_settings(
            enabled=config.tiny_random,
            config_path=config.tiny_random_config,
            seed=config.tiny_random_seed,
        )
