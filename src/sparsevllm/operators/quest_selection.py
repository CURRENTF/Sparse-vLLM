from __future__ import annotations

from dataclasses import dataclass

import torch

import sparsevllm.platforms as platforms
from sparsevllm.kernels.external.flashinfer.topk import (
    flashinfer_top_k_page_table_transform,
    flashinfer_top_k_page_table_transform_support,
)
from sparsevllm.operators.registry import (
    OpRegistry,
    OpResolver,
    PortfolioPolicy,
    ProviderRole,
    SupportResult,
)
from sparsevllm.platforms.interface import DeviceCaps, PlatformEnum


@dataclass(frozen=True)
class QuestPageSelectionOpSpec:
    score_dtype: torch.dtype
    cuda_graph: bool

    def __post_init__(self) -> None:
        if self.score_dtype not in {torch.float32, torch.float16, torch.bfloat16}:
            raise TypeError(
                "QuEST page selection requires FP32, FP16, or BF16 scores, got "
                f"{self.score_dtype}."
            )


def _validate_inputs(
    spec: QuestPageSelectionOpSpec,
    scores: torch.Tensor,
    page_table: torch.Tensor,
    lengths: torch.Tensor,
    k: int,
) -> int:
    if scores.ndim != 2 or not scores.is_contiguous():
        raise ValueError("QuEST page scores must be contiguous and rank 2.")
    if scores.dtype != spec.score_dtype:
        raise TypeError(
            f"QuEST page selector expected {spec.score_dtype}, got {scores.dtype}."
        )
    if page_table.shape != scores.shape or page_table.dtype != torch.int32:
        raise TypeError(
            "QuEST page table must be int32 and match page scores, got "
            f"scores={tuple(scores.shape)} pages={tuple(page_table.shape)}/"
            f"{page_table.dtype}."
        )
    if not page_table.is_contiguous():
        raise ValueError("QuEST page table must be contiguous.")
    if lengths.shape != (int(scores.shape[0]),) or lengths.dtype != torch.int32:
        raise TypeError("QuEST page selection requires one int32 length per row.")
    if not lengths.is_contiguous():
        raise ValueError("QuEST page-selection lengths must be contiguous.")
    if page_table.device != scores.device or lengths.device != scores.device:
        raise ValueError("QuEST page-selection inputs must share a device.")
    k = int(k)
    if not 0 < k <= int(scores.shape[1]):
        raise ValueError(
            f"QuEST page selection requires 0 < k <= {scores.shape[1]}, got {k}."
        )
    return k


class QuestPageSelectionProvider:
    name = ""

    def __init__(self, *, op_spec: QuestPageSelectionOpSpec) -> None:
        self.spec = op_spec

    def select(
        self,
        scores: torch.Tensor,
        page_table: torch.Tensor,
        lengths: torch.Tensor,
        k: int,
    ) -> torch.Tensor:
        raise NotImplementedError


QUEST_PAGE_SELECTION_REGISTRY: OpRegistry[
    QuestPageSelectionOpSpec,
    QuestPageSelectionProvider,
] = OpRegistry(
    "QuEST page selection",
    portfolio=PortfolioPolicy(
        upstream_standard=("flashinfer",),
        repo_portable=("torch",),
    ),
)


@QUEST_PAGE_SELECTION_REGISTRY.register_atomic(ProviderRole.UPSTREAM_STANDARD)
class FlashInferQuestPageSelectionProvider(QuestPageSelectionProvider):
    name = "flashinfer"

    @classmethod
    def supports(
        cls,
        spec: QuestPageSelectionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        if caps.platform != PlatformEnum.CUDA:
            return SupportResult.unsupported(f"requires CUDA, got {caps.platform.name}")
        if spec.cuda_graph and not caps.supports_graph_capture:
            return SupportResult.unsupported("device does not support CUDA Graph capture")
        supported, reason = flashinfer_top_k_page_table_transform_support()
        return SupportResult.yes(reason) if supported else SupportResult.unsupported(reason)

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "flashinfer-python",
            "kernel_path": "flashinfer.top_k_page_table_transform",
            "deterministic": True,
            "tie_break": "small",
            "cuda_graph": self.spec.cuda_graph,
        }

    def select(self, scores, page_table, lengths, k):
        k = _validate_inputs(self.spec, scores, page_table, lengths, k)
        if not scores.is_cuda:
            raise ValueError("FlashInfer QuEST page selection requires CUDA scores.")
        return flashinfer_top_k_page_table_transform(
            scores,
            page_table,
            lengths,
            k,
            cuda_graph=self.spec.cuda_graph,
        )


@QUEST_PAGE_SELECTION_REGISTRY.register_atomic(ProviderRole.REPO_PORTABLE)
class TorchQuestPageSelectionProvider(QuestPageSelectionProvider):
    name = "torch"

    @classmethod
    def supports(
        cls,
        spec: QuestPageSelectionOpSpec,
        caps: DeviceCaps,
    ) -> SupportResult:
        del spec, caps
        return SupportResult.yes("Torch top-k/gather baseline")

    def binding_metadata(self) -> dict[str, object]:
        return {
            "implementation_kind": "atomic_provider",
            "implementation_source": "torch",
            "kernel_path": "torch.topk+torch.gather",
        }

    def select(self, scores, page_table, lengths, k):
        k = _validate_inputs(self.spec, scores, page_table, lengths, k)
        columns = torch.arange(
            int(scores.shape[1]),
            dtype=torch.int32,
            device=scores.device,
        )
        valid = columns[None, :] < lengths[:, None]
        top_indices = scores.masked_fill(~valid, -float("inf")).topk(
            k,
            dim=-1,
            sorted=False,
        ).indices
        selected = page_table.gather(1, top_indices)
        return selected.masked_fill(
            top_indices >= lengths[:, None].to(torch.long),
            -1,
        )


def resolve_quest_page_selection_provider(
    spec: QuestPageSelectionOpSpec,
    *,
    device_index: int | None = None,
) -> QuestPageSelectionProvider:
    platform = platforms.current_platform
    if device_index is None:
        device_index = torch.cuda.current_device() if platform.is_cuda_alike() else 0
    caps = platform.get_device_caps(int(device_index))
    return OpResolver(QUEST_PAGE_SELECTION_REGISTRY).resolve(
        spec,
        caps,
        op_spec=spec,
    ).provider


__all__ = [
    "FlashInferQuestPageSelectionProvider",
    "QUEST_PAGE_SELECTION_REGISTRY",
    "QuestPageSelectionOpSpec",
    "QuestPageSelectionProvider",
    "TorchQuestPageSelectionProvider",
    "resolve_quest_page_selection_provider",
]
