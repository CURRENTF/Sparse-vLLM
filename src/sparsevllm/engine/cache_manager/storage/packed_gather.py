"""Cache-owned active-page materialization shared by sequential native layers."""

from dataclasses import dataclass

import numpy as np
import torch

from .packed_shared_kv import HEAD_DIM, PAGE_SIZE


@dataclass
class PackedGatherPlan:
    metadata: torch.Tensor
    page_capacity: int
    active_pages: int = 0

    @property
    def block_table(self):
        return self.metadata[:self.active_pages][None]

    @property
    def page_map(self):
        return self.metadata[self.page_capacity:2 * self.page_capacity]

    @property
    def length(self):
        return self.metadata[-1:]


class PackedSharedKVGather:
    def __init__(self, page_capacities, *, device):
        if not page_capacities or min(page_capacities.values()) <= 0:
            raise ValueError("Packed gather requires positive page capacities.")
        self.values = torch.empty((1, max(page_capacities.values()) * PAGE_SIZE, HEAD_DIM),
                                  dtype=torch.bfloat16, device=device)
        self.plans = {ratio: PackedGatherPlan(torch.full((2 * pages + 1,), -1, dtype=torch.int32,
                                                       device=device), pages)
                      for ratio, pages in page_capacities.items()}

    def accounting_tensors(self):
        return (self.values, *(plan.metadata for plan in self.plans.values()))

    def prepare(self, ratio, pages):
        """Publish a host-planned page superset once per ratio and prefill batch."""
        plan = self.plans[ratio]
        pages = np.asarray(pages)
        if pages.ndim != 1 or pages.dtype.kind not in "iu":
            raise ValueError("Packed gather pages must be an integer vector.")
        if np.any(pages < 0) or np.any(pages >= plan.page_capacity):
            raise ValueError("Packed gather page is outside the physical domain.")
        pages = np.unique(pages).astype(np.int32)
        host = torch.full((2 * plan.page_capacity + 1,), -1, dtype=torch.int32)
        host[:len(pages)] = torch.from_numpy(pages)
        host[plan.page_capacity + pages.astype(np.int64)] = torch.arange(len(pages), dtype=torch.int32)
        host[-1] = len(pages) * PAGE_SIZE
        plan.metadata.copy_(host)
        plan.active_pages = len(pages)
        return plan

    def materialize(self, storage, ratio, *, layer_idx=0):
        plan = self.plans[ratio]
        if storage.layer_payload(layer_idx).cache.shape[0] != plan.page_capacity:
            raise ValueError("Packed gather plan differs from the storage page domain.")
        out = self.values[:, :plan.active_pages * PAGE_SIZE]
        if plan.active_pages:
            storage.gather(layer_idx, out, plan.block_table, plan.length)
        return out[0, :, None], plan.page_map
