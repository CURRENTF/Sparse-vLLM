from __future__ import annotations

import torch


def select_h2o_heads(
    cumulative: torch.Tensor,
    *,
    selection_groups: int,
    budget: int,
    recent_ratio: float,
    reduction: str,
) -> torch.Tensor:
    """Rank cumulative [Hq, L] probabilities, returning [groups, min(B,L)].

    A group is one native KV head for explicit KV, or the whole layer for
    shared MLA latent storage. Reduction happens only after time accumulation.
    Equal scores prefer older positions, independently of the device top-k.
    """
    if cumulative.ndim != 2 or cumulative.shape[0] == 0:
        raise ValueError("H2O cumulative scores must have shape [query_heads, length].")
    heads, length = cumulative.shape
    if selection_groups <= 0 or heads % selection_groups:
        raise ValueError("H2O query heads must divide into complete selection groups.")
    if budget <= 0 or not 0 < recent_ratio < 1:
        raise ValueError("H2O requires a positive budget and recent_ratio in (0, 1).")
    if reduction not in {"max", "mean"}:
        raise ValueError("H2O head reduction must be 'max' or 'mean'.")
    if length <= budget:
        return torch.arange(length, device=cumulative.device).expand(selection_groups, -1)

    grouped = cumulative.reshape(selection_groups, heads // selection_groups, length)
    ranks = grouped.amax(dim=1) if reduction == "max" else grouped.mean(dim=1)
    recent_count = min(budget, max(1, int(budget * recent_ratio)))
    recent_start = length - recent_count
    heavy = torch.argsort(
        ranks[:, :recent_start], dim=-1, descending=True, stable=True,
    )[:, :budget - recent_count]
    recent = torch.arange(recent_start, length, device=cumulative.device).expand(
        selection_groups, -1,
    )
    return torch.cat((heavy, recent), dim=-1).sort(dim=-1).values
