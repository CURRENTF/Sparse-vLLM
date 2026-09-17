"""R-KV scores matching the official vLLM port's per-head algorithm.

Reference: Zefan-Cai/R-KV 6715468b, vLLM/rkv/algo.py. Pairwise rows are
streamed when necessary; head and request batching stay within a byte budget.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def rkv_head_scores(
    keys: torch.Tensor,
    queries: torch.Tensor,
    *,
    window: int,
    kernel_size: int,
    alpha: float,
    workspace_bytes: int,
) -> torch.Tensor:
    """Return [batch, kv_heads, length-window] joint scores from BHLD keys/Q.

    GQA max is before softmax. Redundancy uses the complete resident domain,
    including the trailing window, and a column mean after representative
    removal. The threshold finds the last representative; it is not a filter.
    """
    if keys.ndim != 4 or queries.ndim != 4:
        raise ValueError("R-KV expects keys/queries shaped [batch, heads, tokens, dim].")
    batch, heads, length, dim = keys.shape
    if (min(batch, heads, length, dim, window) <= 0
            or queries.shape[1] <= 0
            or queries.shape[0] != batch or queries.shape[-1] != dim
            or queries.shape[1] % heads or queries.shape[2] != window
            or length <= window or kernel_size <= 0 or kernel_size % 2 != 1):
        raise ValueError("R-KV incompatible key/query shapes, window, or pooling kernel.")
    if keys.dtype != queries.dtype or keys.device != queries.device:
        raise ValueError("R-KV keys and queries must share dtype and device.")
    groups = queries.shape[1] // heads
    elt = keys.element_size()
    # Include normalized/gathered keys, GQA logits, probabilities, and output.
    fixed_per_head = (4 * length * dim + 4 * groups * window * length) * max(elt, 4)
    # Inputs are gathered by the cache manager; noncontiguous layouts may need
    # a second copy when flattening batch/head dimensions. Account for both.
    resident_bytes = 2 * (keys.numel() + queries.numel()) * elt
    resident_bytes += batch * heads * (length - window) * elt
    available_bytes = int(workspace_bytes) - resident_bytes
    pair_row_bytes = 2 * (2 * elt + 5) * length
    if available_bytes < fixed_per_head + pair_row_bytes:
        raise RuntimeError("R-KV scoring workspace is too small for one head/row; "
                           "increase rkv_score_chunk_mb or reduce the context.")
    units = max(1, min(batch * heads, available_bytes //
                      (fixed_per_head + pair_row_bytes * length)))
    row_tile = min(length, (available_bytes // units - fixed_per_head) // pair_row_bytes)
    flat_keys = keys.reshape(batch * heads, length, dim)
    flat_queries = queries.reshape(batch * heads, groups, window, dim)
    result = torch.empty((batch * heads, length - window), dtype=keys.dtype, device=keys.device)
    columns = torch.arange(length, device=keys.device)
    for start in range(0, batch * heads, units):
        end = min(start + units, batch * heads)
        k = flat_keys[start:end]
        q = flat_queries[start:end]
        logits = torch.matmul(q, k[:, None].transpose(-1, -2)) / math.sqrt(dim)
        logits = logits.max(dim=1).values
        importance = logits[..., :-window].softmax(dim=-1, dtype=torch.float32).mean(dim=-2).to(q.dtype)
        importance = F.max_pool1d(importance[:, None], kernel_size,
                                  stride=1, padding=kernel_size // 2).squeeze(1)
        if k.dtype == torch.float16:
            norm = k.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
            normalized = k / norm.to(k.dtype)
        else:
            normalized = k / (k.norm(dim=-1, keepdim=True) + 1e-8)
        column_sum = None
        for row_start in range(0, length, row_tile):
            row_end = min(row_start + row_tile, length)
            sim = torch.matmul(normalized[:, row_start:row_end], normalized.transpose(-1, -2))
            rows = torch.arange(row_start, row_end, device=k.device)
            sim[:, torch.arange(row_end - row_start, device=k.device), rows] = 0
            representatives = torch.where(sim > 0.5, columns, 0).amax(dim=-1)
            sim.scatter_(-1, representatives.unsqueeze(-1), 0)
            if row_tile == length:
                redundancy_mean = sim.mean(dim=-2)
            else:
                partial = sim.float().sum(dim=-2)
                column_sum = partial if column_sum is None else column_sum + partial
        if row_tile != length:
            redundancy_mean = (column_sum / length).to(k.dtype)
        redundancy = redundancy_mean.softmax(dim=-1)[..., :-window]
        result[start:end] = alpha * importance - (1.0 - alpha) * redundancy
    return result.view(batch, heads, length - window)
