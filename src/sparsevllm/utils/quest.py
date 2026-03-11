import torch


@torch.no_grad()
def build_quest_decode_view(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    req_to_tokens: torch.Tensor,
    req_indices: torch.Tensor,
    context_lens: torch.Tensor,
    *,
    num_kv_heads: int,
    page_size: int,
    token_budget: int,
    layer_idx: int,
    skip_layers: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
    if layer_idx < skip_layers:
        return None

    if q.dim() != 3:
        raise ValueError(f"QuEST expects q to be [B, H, D], got shape={tuple(q.shape)}")
    if req_to_tokens.dim() != 2:
        raise ValueError(f"QuEST expects req_to_tokens to be 2D, got shape={tuple(req_to_tokens.shape)}")

    page_size = int(page_size)
    token_budget = int(token_budget)
    if page_size <= 0 or token_budget <= 0:
        return None

    batch_size, num_heads, _ = q.shape
    if num_heads % num_kv_heads != 0:
        raise ValueError(f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}")
    gqa_group_size = num_heads // num_kv_heads

    local_slots: list[torch.Tensor] = []
    local_lens: list[int] = []
    max_keep = 0

    for b_idx in range(batch_size):
        kv_len = int(context_lens[b_idx].item())
        req_idx = int(req_indices[b_idx].item())
        seq_slots = req_to_tokens[req_idx, :kv_len].to(torch.long)

        if kv_len <= token_budget or kv_len <= page_size:
            keep_slots = seq_slots
        else:
            num_pages = (kv_len + page_size - 1) // page_size
            page_budget = min(num_pages, max(3, token_budget // page_size))
            if page_budget >= num_pages:
                keep_slots = seq_slots
            else:
                seq_k = k_cache.index_select(0, seq_slots).float()  # [L, H_kv, D]
                pad_len = num_pages * page_size - kv_len
                if pad_len > 0:
                    seq_k = torch.cat(
                        [
                            seq_k,
                            torch.zeros(
                                pad_len,
                                seq_k.shape[1],
                                seq_k.shape[2],
                                dtype=seq_k.dtype,
                                device=seq_k.device,
                            ),
                        ],
                        dim=0,
                    )

                paged_k = seq_k.view(num_pages, page_size, num_kv_heads, seq_k.shape[-1]).permute(2, 0, 1, 3)
                page_max = paged_k.amax(dim=2)
                page_min = paged_k.amin(dim=2)

                if gqa_group_size > 1:
                    page_max = page_max.repeat_interleave(gqa_group_size, dim=0)
                    page_min = page_min.repeat_interleave(gqa_group_size, dim=0)

                q_b = q[b_idx].float()
                bound = torch.where(q_b[:, None, :] >= 0, page_max, page_min)
                page_scores = (q_b[:, None, :] * bound).sum(dim=-1).amax(dim=0)

                if num_pages == 1:
                    selected_pages = torch.zeros((1,), dtype=torch.long, device=q.device)
                else:
                    prev_budget = min(page_budget - 1, num_pages - 1)
                    if prev_budget > 0:
                        top_prev = page_scores[:-1].topk(prev_budget, dim=-1).indices
                        selected_pages = torch.cat(
                            [
                                top_prev,
                                torch.tensor([num_pages - 1], dtype=torch.long, device=q.device),
                            ]
                        )
                    else:
                        selected_pages = torch.tensor([num_pages - 1], dtype=torch.long, device=q.device)
                selected_pages = selected_pages.unique(sorted=True)

                token_offsets = torch.arange(page_size, dtype=torch.long, device=q.device)
                token_indices = (selected_pages[:, None] * page_size + token_offsets[None, :]).reshape(-1)
                token_indices = token_indices[token_indices < kv_len]
                keep_slots = seq_slots.index_select(0, token_indices)

        local_slots.append(keep_slots.to(torch.int32))
        local_lens.append(int(keep_slots.numel()))
        max_keep = max(max_keep, int(keep_slots.numel()))

    packed_slots = torch.zeros((batch_size, max_keep), dtype=torch.int32, device=q.device)
    for b_idx, keep_slots in enumerate(local_slots):
        if keep_slots.numel() > 0:
            packed_slots[b_idx, : keep_slots.numel()] = keep_slots

    local_req_indices = torch.arange(batch_size, dtype=torch.int32, device=q.device)
    local_context_lens = torch.tensor(local_lens, dtype=torch.int32, device=q.device)
    return packed_slots, local_req_indices, local_context_lens
