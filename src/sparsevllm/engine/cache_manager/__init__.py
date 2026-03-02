from __future__ import annotations

from .base import CacheManager, LayerBatchStates

__all__ = [
    "CacheManager",
    "LayerBatchStates",
    "StandardCacheManager",
    "SnapKVCacheManager",
    "OmniKVCacheManager",
    "DeepSeekMLACacheManager",
    "DeltaKVCacheManager",
    "DeltaKVCacheTritonManager",
    "DeltaKVCacheTritonManagerV2",
    "DeltaKVCacheTritonManagerV3",
    "DeltaKVCacheTritonManagerV4",
    "DeltaKVCacheTritonManagerV3WithOffload",
    "DeltaKVCacheTritonManagerV3WithCUDAOffload",
]


def __getattr__(name: str):
    if name == "StandardCacheManager":
        from .standard import StandardCacheManager

        return StandardCacheManager
    if name == "SnapKVCacheManager":
        from .snapkv import SnapKVCacheManager

        return SnapKVCacheManager
    if name == "OmniKVCacheManager":
        from .omnikv import OmniKVCacheManager

        return OmniKVCacheManager
    if name == "DeepSeekMLACacheManager":
        from .deepseek_mla import DeepSeekMLACacheManager

        return DeepSeekMLACacheManager

    if name in {
        "DeltaKVCacheManager",
        "DeltaKVCacheTritonManager",
        "DeltaKVCacheTritonManagerV2",
        "DeltaKVCacheTritonManagerV3",
        "DeltaKVCacheTritonManagerV4",
        "DeltaKVCacheTritonManagerV3WithOffload",
        "DeltaKVCacheTritonManagerV3WithCUDAOffload",
    }:
        from . import deltakv as _deltakv

        return getattr(_deltakv, name)

    raise AttributeError(name)
