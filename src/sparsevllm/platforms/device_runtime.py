from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch

import sparsevllm.platforms as platforms
from sparsevllm.platforms.interface import Platform


def get_device(local_rank: int = 0) -> torch.device:
    return platforms.current_platform.get_device(local_rank)


def set_device(device: torch.device | int | str) -> None:
    platforms.current_platform.set_device(device)


def synchronize() -> None:
    platforms.current_platform.synchronize()


def empty_cache() -> None:
    platforms.current_platform.empty_cache()


def _optional_platform() -> Platform | None:
    try:
        return platforms.get_current_platform()
    except RuntimeError:
        return None


def supports_pin_memory() -> bool:
    platform = _optional_platform()
    return bool(platform is not None and platform.supports_pin_memory())


def supports_streams(device: torch.device | str | int | None = None) -> bool:
    platform = _optional_platform()
    if platform is None or not platform.is_cuda_alike():
        return False
    if not torch.cuda.is_available():
        return False
    if device is None:
        return True
    return torch.device(device).type == "cuda"


def optional_device_name(device_id: int = 0) -> str:
    platform = _optional_platform()
    if platform is None or not platform.is_cuda_alike() or not torch.cuda.is_available():
        return ""
    try:
        return str(torch.cuda.get_device_name(int(device_id)))
    except RuntimeError:
        return ""


def new_event(device: torch.device | str | int | None = None) -> Any | None:
    if not supports_streams(device):
        return None
    return torch.cuda.Event()


def record_event(event: Any, device: torch.device | str | int | None = None) -> None:
    if event is None:
        return
    event.record(torch.cuda.current_stream(device=device))


def wait_event(event: Any, device: torch.device | str | int | None = None) -> None:
    if event is None:
        return
    if supports_streams(device):
        torch.cuda.current_stream(device=device).wait_event(event)
    else:
        event.synchronize()


def synchronize_event(event: Any) -> None:
    if event is not None:
        event.synchronize()


def new_stream(device: torch.device | str | int | None = None) -> Any | None:
    if not supports_streams(device):
        return None
    return torch.cuda.Stream(device=device)


def stream_context(stream: Any):
    if stream is None:
        return nullcontext()
    return torch.cuda.stream(stream)


def stream_wait_event(stream: Any, event: Any) -> None:
    if stream is None:
        wait_event(event)
        return
    stream.wait_event(event)
