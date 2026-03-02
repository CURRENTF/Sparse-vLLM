from __future__ import annotations

from sparsevllm.config import Config

from .standard import StandardCacheManager


class OmniKVCacheManager(StandardCacheManager):
    def __init__(self, config: Config, rank: int, world_size: int):
        super().__init__(config, rank, world_size)


