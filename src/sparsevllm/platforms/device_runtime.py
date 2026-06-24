from __future__ import annotations

import torch

import sparsevllm.platforms as platforms


def get_device(local_rank: int = 0) -> torch.device:
    return platforms.current_platform.get_device(local_rank)


def set_device(device: torch.device | int | str) -> None:
    platforms.current_platform.set_device(device)


def synchronize() -> None:
    platforms.current_platform.synchronize()


def empty_cache() -> None:
    platforms.current_platform.empty_cache()
