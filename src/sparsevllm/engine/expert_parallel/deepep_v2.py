from __future__ import annotations

from dataclasses import dataclass
import importlib.metadata as importlib_metadata
import re
from typing import Any

import torch
import torch.distributed as dist

from sparsevllm.utils.parallel_context import get_ep_group


_MIN_NCCL_VERSION = (2, 30, 4)
_BUFFER_CACHE: dict[tuple[int, int, int, int, bool], Any] = {}
_NCCL_DIST_NAMES = ("nvidia-nccl-cu13", "nvidia-nccl-cu12", "nvidia-nccl")


@dataclass(frozen=True)
class DeepEPV2Dispatch:
    recv_x: torch.Tensor
    recv_topk_idx: torch.Tensor
    recv_topk_weights: torch.Tensor | None
    handle: Any
    num_max_tokens_per_rank: int
    num_recv_tokens: int


def _nccl_version_tuple() -> tuple[int, ...]:
    version = torch.cuda.nccl.version()
    if isinstance(version, tuple):
        return tuple(int(x) for x in version)
    if isinstance(version, int):
        major = version // 1000
        minor = (version % 1000) // 100
        patch = version % 100
        return major, minor, patch
    raise RuntimeError(f"Unexpected torch.cuda.nccl.version() result: {version!r}")


def _parse_version(version: str) -> tuple[int, ...] | None:
    match = re.match(r"^(\d+)(?:\.(\d+))?(?:\.(\d+))?", version)
    if match is None:
        return None
    return tuple(int(part or 0) for part in match.groups())


def _nccl_package_version_tuple() -> tuple[str, tuple[int, ...]] | None:
    for dist_name in _NCCL_DIST_NAMES:
        try:
            version = importlib_metadata.version(dist_name)
        except importlib_metadata.PackageNotFoundError:
            continue
        parsed = _parse_version(version)
        if parsed is not None:
            return dist_name, parsed
    return None


def _check_nccl_version(deep_ep: Any) -> None:
    package_version = _nccl_package_version_tuple()
    if package_version is not None:
        _, version = package_version
        if version >= _MIN_NCCL_VERSION:
            return

    torch_version = _nccl_version_tuple()
    if torch_version >= _MIN_NCCL_VERSION:
        return

    nccl_root = "unknown"
    try:
        nccl_root = str(deep_ep.utils.find_nccl_root())
    except Exception:
        pass
    package_detail = (
        f"{package_version[0]}=={'.'.join(str(x) for x in package_version[1])}"
        if package_version is not None
        else "no nvidia-nccl wheel found"
    )
    raise RuntimeError(
        "DeepEP v2 requires NCCL >= 2.30.4. "
        f"torch.cuda.nccl.version() returned {torch_version}; "
        f"NCCL package check returned {package_detail}; "
        f"DeepEP NCCL root is {nccl_root}."
    )


def _import_deepep():
    try:
        import deep_ep  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "expert_parallel_backend='deepep_v2' requires the DeepEP package. "
            "Install DeepEP v2 in the active environment before enabling this backend."
        ) from exc
    if not hasattr(deep_ep, "ElasticBuffer"):
        raise RuntimeError(
            "DeepEP package does not expose ElasticBuffer; install DeepEP v2, not the legacy API."
        )
    return deep_ep


def _require_ep_group() -> object:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("DeepEP v2 expert parallelism requires initialized torch.distributed.")
    group = get_ep_group()
    if group is None:
        raise RuntimeError("DeepEP v2 expert parallelism requires an EP process group.")
    return group


def _is_cuda_graph_capturing() -> bool:
    return bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())


def _resolve_num_max_tokens_per_rank(num_local_tokens: int, device: torch.device, group: object) -> int:
    if _is_cuda_graph_capturing():
        return max(1, int(num_local_tokens))
    local_tokens = torch.tensor([int(num_local_tokens)], device=device, dtype=torch.int32)
    dist.all_reduce(local_tokens, op=dist.ReduceOp.MAX, group=group)
    return max(1, int(local_tokens.item()))


def get_buffer(
    *,
    num_max_tokens_per_rank: int,
    hidden_size: int,
    num_topk: int,
    num_experts: int,
    use_fp8_dispatch: bool = False,
):
    deep_ep = _import_deepep()
    _check_nccl_version(deep_ep)
    group = _require_ep_group()
    key = (
        id(group),
        int(num_max_tokens_per_rank),
        int(hidden_size),
        int(num_topk),
        bool(use_fp8_dispatch),
    )
    required_bytes = deep_ep.ElasticBuffer.get_buffer_size_hint(
        group,
        int(num_max_tokens_per_rank),
        int(hidden_size),
        num_topk=int(num_topk),
        use_fp8_dispatch=bool(use_fp8_dispatch),
    )
    buffer = _BUFFER_CACHE.get(key)
    if buffer is not None and getattr(buffer, "num_bytes", 0) >= required_bytes:
        return buffer
    for cached_key, cached_buffer in _BUFFER_CACHE.items():
        (
            cached_group_id,
            cached_num_max_tokens_per_rank,
            cached_hidden_size,
            cached_num_topk,
            cached_use_fp8_dispatch,
        ) = cached_key
        if (
            cached_group_id == id(group)
            and cached_num_max_tokens_per_rank >= int(num_max_tokens_per_rank)
            and cached_hidden_size == int(hidden_size)
            and cached_num_topk == int(num_topk)
            and cached_use_fp8_dispatch == bool(use_fp8_dispatch)
            and getattr(cached_buffer, "num_bytes", 0) >= required_bytes
        ):
            return cached_buffer
    buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=int(num_max_tokens_per_rank),
        hidden=int(hidden_size),
        num_topk=int(num_topk),
        use_fp8_dispatch=bool(use_fp8_dispatch),
        explicitly_destroy=True,
    )
    _BUFFER_CACHE[key] = buffer
    return buffer


def destroy_cached_buffers() -> None:
    """Destroy DeepEP buffers while the process group is still valid."""
    if not _BUFFER_CACHE:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    errors: list[BaseException] = []
    for buffer in list(_BUFFER_CACHE.values()):
        try:
            destroy = getattr(buffer, "destroy", None)
            if callable(destroy):
                destroy()
        except BaseException as exc:
            errors.append(exc)
    _BUFFER_CACHE.clear()
    if errors:
        raise RuntimeError("Failed to destroy DeepEP v2 cached buffers.") from errors[0]


def dispatch(
    *,
    hidden_states: torch.Tensor,
    selected_experts: torch.Tensor,
    routing_weights: torch.Tensor,
    num_experts: int,
) -> DeepEPV2Dispatch:
    if hidden_states.dim() != 2:
        raise ValueError(
            "DeepEP v2 MoE dispatch expects hidden_states with shape [tokens, hidden], "
            f"got {tuple(hidden_states.shape)}."
        )
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "DeepEP v2 dispatch currently requires bfloat16 hidden states, "
            f"got {hidden_states.dtype}."
        )
    if selected_experts.dim() != 2:
        raise ValueError(
            "DeepEP v2 MoE dispatch expects selected_experts with shape [tokens, top_k], "
            f"got {tuple(selected_experts.shape)}."
        )
    if routing_weights.shape != selected_experts.shape:
        raise ValueError(
            "DeepEP v2 routing_weights shape must match selected_experts: "
            f"routing_weights={tuple(routing_weights.shape)} "
            f"selected_experts={tuple(selected_experts.shape)}."
        )

    group = _require_ep_group()
    num_max_tokens_per_rank = _resolve_num_max_tokens_per_rank(
        int(hidden_states.shape[0]),
        hidden_states.device,
        group,
    )
    buffer = get_buffer(
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden_size=int(hidden_states.shape[-1]),
        num_topk=int(selected_experts.shape[-1]),
        num_experts=int(num_experts),
    )
    graph_capturing = _is_cuda_graph_capturing()
    topk_idx = selected_experts if selected_experts.dtype == torch.int64 else selected_experts.to(torch.int64)
    topk_weights = routing_weights if routing_weights.dtype == torch.float32 else routing_weights.float()
    recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
        hidden_states,
        topk_idx=topk_idx,
        topk_weights=topk_weights,
        num_experts=int(num_experts),
        num_max_tokens_per_rank=int(num_max_tokens_per_rank),
        expert_alignment=1,
        async_with_compute_stream=True,
        do_handle_copy=not graph_capturing,
        do_cpu_sync=not graph_capturing,
        do_expand=False,
    )
    event.current_stream_wait()
    num_recv_tokens = int(recv_x.shape[0]) if graph_capturing else int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    return DeepEPV2Dispatch(
        recv_x=recv_x,
        recv_topk_idx=recv_topk_idx,
        recv_topk_weights=recv_topk_weights,
        handle=handle,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_recv_tokens=num_recv_tokens,
    )


def combine(local_output: torch.Tensor, handle: Any) -> torch.Tensor:
    deep_ep = _import_deepep()
    _check_nccl_version(deep_ep)
    buffer = get_buffer(
        num_max_tokens_per_rank=int(handle.num_max_tokens_per_rank),
        hidden_size=int(local_output.shape[-1]),
        num_topk=int(handle.topk_idx.shape[-1]),
        num_experts=int(handle.num_experts),
    )
    combined_x, _, event = buffer.combine(
        local_output,
        handle=handle,
        async_with_compute_stream=True,
    )
    event.current_stream_wait()
    return combined_x
