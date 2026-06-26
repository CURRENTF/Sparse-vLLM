from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist

from sparsevllm.utils.parallel_context import get_ep_group


_MIN_NCCL_VERSION = (2, 30, 4)
_BUFFER_CACHE: dict[tuple[int, int, int, int, bool], Any] = {}


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


def _check_nccl_version() -> None:
    version = _nccl_version_tuple()
    if version < _MIN_NCCL_VERSION:
        raise RuntimeError(
            "DeepEP v2 requires NCCL >= 2.30.4. "
            f"torch.cuda.nccl.version() returned {version}."
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


def _resolve_num_max_tokens_per_rank(num_local_tokens: int, device: torch.device, group: object) -> int:
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
    _check_nccl_version()
    deep_ep = _import_deepep()
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
    buffer = deep_ep.ElasticBuffer(
        group,
        num_max_tokens_per_rank=int(num_max_tokens_per_rank),
        hidden=int(hidden_size),
        num_topk=int(num_topk),
        use_fp8_dispatch=bool(use_fp8_dispatch),
    )
    _BUFFER_CACHE[key] = buffer
    return buffer


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
    recv_x, recv_topk_idx, recv_topk_weights, handle, event = buffer.dispatch(
        hidden_states,
        topk_idx=selected_experts.to(torch.int64),
        topk_weights=routing_weights.float(),
        num_experts=int(num_experts),
        num_max_tokens_per_rank=int(num_max_tokens_per_rank),
        expert_alignment=1,
        async_with_compute_stream=True,
        do_cpu_sync=True,
        do_expand=False,
    )
    event.current_stream_wait()
    num_recv_tokens = int(handle.psum_num_recv_tokens_per_scaleup_rank[-1].item())
    return DeepEPV2Dispatch(
        recv_x=recv_x,
        recv_topk_idx=recv_topk_idx,
        recv_topk_weights=recv_topk_weights,
        handle=handle,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        num_recv_tokens=num_recv_tokens,
    )


def combine(local_output: torch.Tensor, handle: Any) -> torch.Tensor:
    _check_nccl_version()
    deep_ep = _import_deepep()
    del deep_ep
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
