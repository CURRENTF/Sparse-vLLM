from typing import Any


def config_get(config: Any, name: str, default: Any = None) -> Any:
    if config is None:
        return default
    return config.get(name, default) if isinstance(config, dict) else getattr(config, name, default)
