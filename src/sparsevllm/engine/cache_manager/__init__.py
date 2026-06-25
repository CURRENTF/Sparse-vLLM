from __future__ import annotations

from .base import CacheManager, DecodeComputeView, LayerBatchStates, PrefillComputeView, SparseSelection

__all__ = [
    "CacheManager",
    "DecodeComputeView",
    "LayerBatchStates",
    "PrefillComputeView",
    "SparseSelection",
    "StandardCacheManager",
    "StreamingLLMCacheManager",
    "SnapKVCacheManager",
    "RKVCacheManager",
    "SkipKVCacheManager",
    "QuestCacheManager",
    "OmniKVCacheManager",
    "DeltaKVCacheManager",
    "DeltaKVCacheTritonManagerV4",
    "DeltaKVLessMemoryCacheManager",
    "DeltaKVLessMemoryCudaGraphCacheManager",
]


def __getattr__(name: str):
    if name == "StandardCacheManager":
        from .standard import StandardCacheManager

        return StandardCacheManager
    if name == "StreamingLLMCacheManager":
        from .streamingllm import StreamingLLMCacheManager

        return StreamingLLMCacheManager
    if name == "SnapKVCacheManager":
        from .snapkv import SnapKVCacheManager

        return SnapKVCacheManager
    if name == "RKVCacheManager":
        from .rkv import RKVCacheManager

        return RKVCacheManager
    if name == "SkipKVCacheManager":
        from .skipkv import SkipKVCacheManager

        return SkipKVCacheManager
    if name == "QuestCacheManager":
        from .quest import QuestCacheManager

        return QuestCacheManager
    if name == "OmniKVCacheManager":
        from .omnikv import OmniKVCacheManager

        return OmniKVCacheManager
    if name == "DeltaKVCacheManager":
        from .deltakv_runtime import DeltaKVCacheManager

        return DeltaKVCacheManager
    if name == "DeltaKVCacheTritonManagerV4":
        from .deltakv_base import DeltaKVCacheTritonManagerV4

        return DeltaKVCacheTritonManagerV4
    if name == "DeltaKVLessMemoryCacheManager":
        from .deltakv_less_memory import DeltaKVLessMemoryCacheManager

        return DeltaKVLessMemoryCacheManager
    if name == "DeltaKVLessMemoryCudaGraphCacheManager":
        from .deltakv_less_memory_cuda_graph import DeltaKVLessMemoryCudaGraphCacheManager

        return DeltaKVLessMemoryCudaGraphCacheManager

    raise AttributeError(name)
