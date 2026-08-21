"""Experimental fixed-split Gemma 4 attention backend."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sparsevllm import platforms
from sparsevllm.kernels.triton.context_independent_gemma4_decode_attention import (
    context_independent_gemma4_decode,
)
from sparsevllm.layers.attention_backend import _require_explicit_payload
from sparsevllm.operators.gemma4_attention import Gemma4AttentionBackend


@dataclass
class _Gemma4DecodeWorkspace:
    mid_output: torch.Tensor
    mid_lse: torch.Tensor

    @property
    def nbytes(self) -> int:
        return (
            self.mid_output.numel() * self.mid_output.element_size()
            + self.mid_lse.numel() * self.mid_lse.element_size()
        )


class ContextIndependentGemma4AttentionBackend(Gemma4AttentionBackend):
    name = "triton_gemma4_context_independent"
    cuda_graph_context_independent = True

    def __init__(
        self,
        *,
        baseline: Gemma4AttentionBackend,
        workspace: _Gemma4DecodeWorkspace,
        target_tokens_per_split: int = 256,
        use_grouped_no_score: bool = True,
        safe_fp32_reduction: bool = False,
        stage1_num_warps: int = 8,
    ) -> None:
        super().__init__(
            sliding_window=baseline.sliding_window,
            flashinfer_prefill=baseline.flashinfer_prefill,
            use_window_decode=False,
            global_decode_heads_per_program=None,
        )
        self.workspace = workspace
        self.target_tokens_per_split = int(target_tokens_per_split)
        self.use_grouped_no_score = bool(use_grouped_no_score)
        self.safe_fp32_reduction = bool(safe_fp32_reduction)
        self.stage1_num_warps = int(stage1_num_warps)
        if self.sliding_window is None and self.stage1_num_warps == 4:
            self.name = "triton_gemma4_context_independent_h20_tuned"

    def get_decode_workspace(
        self,
        *,
        batch_size: int,
        num_heads: int,
        head_dim: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mid_output = self.workspace.mid_output
        mid_lse = self.workspace.mid_lse
        if (
            batch_size > int(mid_output.shape[0])
            or num_heads != int(mid_output.shape[1])
            or head_dim != int(mid_output.shape[3])
            or device != mid_output.device
        ):
            raise RuntimeError(
                "context-independent Gemma 4 workspace mismatch: "
                f"requested={(batch_size, num_heads, head_dim, device)} "
                f"prepared={tuple(mid_output.shape)}/{mid_output.device}"
            )
        return mid_output[:batch_size], mid_lse[:batch_size]

    def run_decode(
        self,
        q: torch.Tensor,
        view,
        *,
        mid_o: torch.Tensor,
        mid_o_logexpsum: torch.Tensor,
        max_len_in_batch: int,
        block_seq: int,
        num_heads: int,
        num_kv_heads: int,
        gqa_block_n: int = 16,
        gqa_num_warps: int = 2,
    ) -> torch.Tensor:
        del (
            max_len_in_batch,
            block_seq,
            num_heads,
            num_kv_heads,
            gqa_block_n,
            gqa_num_warps,
        )
        payload = _require_explicit_payload(
            view,
            operation="context-independent Gemma 4 decode",
        )
        if payload.backend != "dense":
            raise RuntimeError(
                "context-independent Gemma 4 decode requires dense explicit KV, "
                f"got backend={payload.backend!r}"
            )
        return context_independent_gemma4_decode(
            q,
            payload.k_cache,
            payload.v_cache,
            view.meta.active_slots,
            view.meta.req_indices,
            view.meta.context_lens,
            mid_o,
            mid_o_logexpsum,
            sliding_window=self.sliding_window,
            attn_score=view.meta.attn_score,
            target_tokens_per_split=self.target_tokens_per_split,
            use_grouped_no_score=self.use_grouped_no_score,
            safe_fp32_reduction=self.safe_fp32_reduction,
            stage1_num_warps=self.stage1_num_warps,
        )


def bind_context_independent_gemma4_attention(
    model: torch.nn.Module,
    *,
    max_batch_size: int,
    device: torch.device,
    global_max_kv_splits: int = 64,
    window_max_kv_splits: int = 16,
) -> tuple[int, int]:
    device_index = 0 if device.index is None else int(device.index)
    device_name = platforms.current_platform.get_device_caps(device_index).device_name
    use_h20_global_safe_path = "H20" in device_name
    workspaces: dict[tuple[int, int, int], _Gemma4DecodeWorkspace] = {}
    bound = 0
    for module in model.modules():
        baseline = getattr(module, "attention_backend", None)
        if type(baseline) is not Gemma4AttentionBackend:
            continue
        num_heads = int(getattr(module, "num_heads"))
        head_dim = int(getattr(module, "head_dim"))
        max_kv_splits = (
            global_max_kv_splits
            if baseline.sliding_window is None
            else window_max_kv_splits
        )
        signature = (num_heads, head_dim, max_kv_splits)
        workspace = workspaces.get(signature)
        if workspace is None:
            workspace = _Gemma4DecodeWorkspace(
                mid_output=torch.empty(
                    (max_batch_size, num_heads, max_kv_splits, head_dim),
                    dtype=torch.float32,
                    device=device,
                ),
                mid_lse=torch.empty(
                    (max_batch_size, num_heads, max_kv_splits),
                    dtype=torch.float32,
                    device=device,
                ),
            )
            workspaces[signature] = workspace
        module.attention_backend = ContextIndependentGemma4AttentionBackend(
            baseline=baseline,
            workspace=workspace,
            # H20 global HD=512 needs the per-head four-warp specialization.
            # Window attention is unaffected and keeps the tuned grouped kernel.
            use_grouped_no_score=(
                baseline.sliding_window is not None or not use_h20_global_safe_path
            ),
            stage1_num_warps=(
                4
                if baseline.sliding_window is None and use_h20_global_safe_path
                else 8
            ),
        )
        bound += 1
    return bound, sum(workspace.nbytes for workspace in workspaces.values())


__all__ = [
    "ContextIndependentGemma4AttentionBackend",
    "bind_context_independent_gemma4_attention",
]
