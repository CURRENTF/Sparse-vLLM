from __future__ import annotations

import importlib.metadata
import importlib.util
import re

import torch
import triton

from sparsevllm.triton_kernel.moe import MoeAlignment


_SGL_KERNEL_DISTRIBUTIONS = ("sglang-kernel", "sgl-kernel")
_MIN_VALIDATED_VERSIONS = {
    (0, 3): (0, 3, 14),
    (0, 4): (0, 4, 5),
}


def _parse_version(version: str) -> tuple[int, int, int] | None:
    parts = str(version).split(".")
    if len(parts) < 3:
        return None
    parsed: list[int] = []
    for part in parts[:3]:
        match = re.match(r"(\d+)", part)
        if match is None:
            return None
        parsed.append(int(match.group(1)))
    return tuple(parsed)  # type: ignore[return-value]


def _installed_sgl_kernel_version() -> tuple[str, tuple[int, int, int]] | None:
    for distribution in _SGL_KERNEL_DISTRIBUTIONS:
        try:
            version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
        parsed = _parse_version(version)
        if parsed is None:
            return version, (-1, -1, -1)
        return version, parsed
    return None


def _sgl_moe_support(feature: str) -> tuple[bool, str]:
    try:
        module_spec = importlib.util.find_spec("sgl_kernel")
    except (ImportError, ValueError):
        module_spec = None
    if module_spec is None:
        return False, "sgl-kernel is not installed"
    installed = _installed_sgl_kernel_version()
    if installed is None:
        return False, "sgl-kernel package metadata is unavailable"
    version, parsed = installed
    minimum = _MIN_VALIDATED_VERSIONS.get(parsed[:2])
    if minimum is None:
        return (
            False,
            "requires validated sgl-kernel 0.3.x or sglang-kernel 0.4.x "
            f"API, got {version}",
        )
    if parsed < minimum:
        package = "sglang-kernel" if parsed[:2] == (0, 4) else "sgl-kernel"
        required = ".".join(str(value) for value in minimum)
        return False, f"requires {package} >= {required}, got {version}"
    return True, f"SGL kernel {version} {feature} is available"


def sgl_fused_moe_gate_support() -> tuple[bool, str]:
    """Check the SGL fused-gate API used by the GLM router."""

    return _sgl_moe_support("MoE gate")


def sgl_moe_alignment_support() -> tuple[bool, str]:
    """Check the SGL expert-alignment API used by the Triton MoE provider."""

    return _sgl_moe_support("MoE alignment")


def sgl_moe_ep_alignment_support() -> tuple[bool, str]:
    """Check the invalid-route filtering required by expert parallelism."""

    return _sgl_moe_support("MoE EP alignment")


class SglGlmFusedMoeGate:
    """One-kernel sigmoid, biased top-k, normalization, and route scaling."""

    def __init__(self, *, num_experts: int, top_k: int) -> None:
        supported, reason = sgl_fused_moe_gate_support()
        if not supported:
            raise RuntimeError(reason)
        if (int(num_experts), int(top_k)) != (64, 4):
            raise ValueError(
                "SGL GLM fused gate is validated only for 64 experts and "
                f"top-k 4, got {num_experts} and {top_k}."
            )
        installed = _installed_sgl_kernel_version()
        if installed is None:
            raise RuntimeError("sgl-kernel package metadata is unavailable")
        _, parsed = installed
        self._api_series = parsed[:2]
        if self._api_series == (0, 3):
            from sgl_kernel import moe_fused_gate

            self._op = moe_fused_gate
        else:
            from sgl_kernel import topk_sigmoid

            self._op = topk_sigmoid

    def __call__(
        self,
        router_logits: torch.Tensor,
        correction_bias: torch.Tensor,
        *,
        top_k: int,
        routed_scaling_factor: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(router_logits.shape[1:]) != (64,) or int(top_k) != 4:
            raise ValueError(
                "SGL GLM fused gate expects logits [tokens, 64] and top-k 4, "
                f"got {tuple(router_logits.shape)} and {top_k}."
            )
        if router_logits.dtype != torch.float32:
            raise TypeError(
                "SGL GLM fused gate requires FP32 logits, got "
                f"{router_logits.dtype}."
            )
        if tuple(correction_bias.shape) != (64,):
            raise ValueError(
                "SGL GLM fused gate expects correction_bias [64], got "
                f"{tuple(correction_bias.shape)}."
            )

        # The 0.3.x kernel supports at most 32 experts per group. Splitting
        # GLM's single 64-expert group into two virtual groups and selecting
        # both preserves the original global top-k candidate set.
        if self._api_series == (0, 3):
            weights, ids = self._op(
                router_logits,
                correction_bias,
                2,
                2,
                4,
                0,
                float(routed_scaling_factor),
                True,
            )
        else:
            weights = torch.empty(
                (int(router_logits.shape[0]), 4),
                dtype=torch.float32,
                device=router_logits.device,
            )
            ids = torch.empty(
                (int(router_logits.shape[0]), 4),
                dtype=torch.int32,
                device=router_logits.device,
            )
            self._op(weights, ids, router_logits, True, correction_bias)
            weights.mul_(float(routed_scaling_factor))
        return weights, ids


def sgl_moe_align_block_size(
    topk_ids: torch.Tensor,
    *,
    block_size: int,
    num_experts: int,
    local_expert_start: int,
    local_expert_end: int,
) -> MoeAlignment:
    """Build a local MoE assignment with the SGL CUDA kernel."""

    local_expert_start = int(local_expert_start)
    local_expert_end = int(local_expert_end)
    num_local_experts = local_expert_end - local_expert_start
    if not 0 <= local_expert_start < local_expert_end <= int(num_experts):
        raise ValueError(
            "Invalid local expert range "
            f"[{local_expert_start}, {local_expert_end}) for {num_experts} experts."
        )
    supported, reason = sgl_moe_alignment_support()
    if not supported:
        raise RuntimeError(reason)
    installed = _installed_sgl_kernel_version()
    if installed is None:
        raise RuntimeError("sgl-kernel package metadata is unavailable")
    _, parsed = installed

    local_topk_ids = topk_ids
    has_remote_experts = num_local_experts != int(num_experts)
    ignore_invalid_expert = has_remote_experts and parsed[:2] == (0, 4)
    if num_local_experts != int(num_experts):
        from sparsevllm.triton_kernel.moe import localize_expert_ids

        local_topk_ids = localize_expert_ids(
            topk_ids,
            local_expert_start=local_expert_start,
            local_expert_end=local_expert_end,
            remote_expert_id=-1 if ignore_invalid_expert else num_local_experts,
        )
    num_assignments = int(local_topk_ids.numel())
    padding_experts = num_local_experts + int(
        has_remote_experts and parsed[:2] == (0, 3)
    )
    max_num_tokens_padded = triton.cdiv(
        num_assignments + padding_experts * (int(block_size) - 1),
        int(block_size),
    ) * int(block_size)
    sorted_token_ids = torch.empty(
        max_num_tokens_padded,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    expert_ids = torch.empty(
        max_num_tokens_padded // int(block_size),
        dtype=torch.int32,
        device=topk_ids.device,
    )
    num_tokens_post_padded = torch.empty(
        1,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    cumsum_buffer = torch.empty(
        num_local_experts + 1,
        dtype=torch.int32,
        device=topk_ids.device,
    )
    from sgl_kernel import moe_align_block_size

    # Both validated APIs iterate to num_experts - 1. The extra empty logical
    # expert makes the complete [0, num_experts) range participate.
    if parsed[:2] == (0, 3):
        moe_align_block_size(
            local_topk_ids,
            num_local_experts + 1,
            int(block_size),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            cumsum_buffer,
            True,
        )
    else:
        moe_align_block_size(
            local_topk_ids,
            num_local_experts + 1,
            int(block_size),
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            cumsum_buffer,
            True,
            ignore_invalid_expert,
        )
    return MoeAlignment(
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
        block_size=int(block_size),
        naive=False,
    )


__all__ = [
    "SglGlmFusedMoeGate",
    "sgl_fused_moe_gate_support",
    "sgl_moe_alignment_support",
    "sgl_moe_ep_alignment_support",
    "sgl_moe_align_block_size",
]
