from torch import nn
import torch
import torch.nn.functional as F
from typing import Optional, Union, Tuple
from transformers.models.llama.modeling_llama import repeat_kv


def get_qk_score(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    scaling: float,
    no_softmax=False,
):
    # query: (bs, num_heads, q_len, head_dim)
    # key: (bs, num_kv_heads, k_len, head_dim)
    key_states = repeat_kv(key, module.num_key_value_groups)
    attn_weights = torch.matmul(query, key_states.transpose(2, 3))
    if no_softmax: return attn_weights

    attn_weights = nn.functional.softmax(attn_weights * scaling, dim=-1, dtype=torch.float32).to(query.dtype)
    return attn_weights


def snapkv_token_selection(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    scaling: float,
    num_top_tokens: Union[int, float],
    pool_kernel_size: int = 5,
    output_2d: bool = True,
):
    # candidate_weights shape: (bs, heads, q_len, n_candidates)
    candidate_weights = get_qk_score(module, query_states, key_states, scaling)
    
    bs, num_heads, q_len, n_candidates = candidate_weights.shape
    
    if output_2d:
        # Global selection (across all heads)
        # Voting: mean over queries
        vote_score = candidate_weights.mean(dim=2) # (bs, heads, n_candidates)
        # Aggregation: max over heads
        vote_score = vote_score.max(dim=1).values # (bs, n_candidates)
        
        # Pooling
        pool_padding = pool_kernel_size // 2
        score_in = vote_score.unsqueeze(1)
        pooled_score = F.max_pool1d(
            score_in,
            kernel_size=pool_kernel_size,
            padding=pool_padding,
            stride=1
        ).squeeze(1)
        
        # Selection
        if isinstance(num_top_tokens, float) and num_top_tokens <= 1.0:
            k = int(n_candidates * num_top_tokens)
        else:
            k = int(num_top_tokens)
        k = min(max(k, 1), n_candidates)
        
        _, top_token_idx = torch.topk(pooled_score, dim=-1, k=k)
        return top_token_idx # (bs, k)
    else:
        # Per-KV-head selection (compatible with GQA)
        num_kv_heads = module.num_key_value_heads
        num_groups = num_heads // num_kv_heads
        
        # Reshape to (bs, num_kv_heads, num_groups, q_len, n_candidates)
        vote_score = candidate_weights.view(bs, num_kv_heads, num_groups, q_len, n_candidates)
        # Mean over groups and queries
        vote_score = vote_score.mean(dim=(2, 3)) # (bs, num_kv_heads, n_candidates)
        
        # Pooling
        pool_padding = pool_kernel_size // 2
        score_in = vote_score.view(-1, 1, n_candidates)
        pooled_score = F.max_pool1d(
            score_in,
            kernel_size=pool_kernel_size,
            padding=pool_padding,
            stride=1
        ).view(bs, num_kv_heads, n_candidates)
        
        # Selection
        if isinstance(num_top_tokens, float) and num_top_tokens <= 1.0:
            k = int(n_candidates * num_top_tokens)
        else:
            k = int(num_top_tokens)
        k = min(max(k, 1), n_candidates)
        
        _, top_token_idx = torch.topk(pooled_score, dim=-1, k=k)
        return top_token_idx # (bs, num_kv_heads, k)


def omnikv_token_selection(
    module: nn.Module,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    scaling: float,
    num_top_tokens: Union[int, float],
    pool_kernel_size: int = 5,
    last_token_scores: Optional[torch.Tensor] = None,
    score_method: str = 'last',
) -> Tuple[torch.Tensor, torch.Tensor]:
    # candidate_weights shape: (bs, heads, q_len, n_candidates)
    if query_states.shape[-2] > 1:
        candidate_weights = get_qk_score(module, query_states, key_states, scaling, no_softmax=True)
    else:
        candidate_weights = get_qk_score(module, query_states, key_states, scaling)
    
    # Voting & Aggregation
    # Mean over queries, max over heads
    token_scores = candidate_weights.mean(dim=2).max(dim=1).values # (bs, n_candidates)
    
    # Max pooling to smooth scores
    if pool_kernel_size > 1:
        pool_padding = pool_kernel_size // 2
        token_scores = F.max_pool1d(
            token_scores.unsqueeze(1),
            kernel_size=pool_kernel_size,
            padding=pool_padding,
            stride=1
        ).squeeze(1)
        
    # Score EMA (exp logic)
    if last_token_scores is not None and score_method == 'exp':
        prev_len = last_token_scores.shape[-1]
        token_scores[:, :prev_len] = last_token_scores * 0.5 + token_scores[:, :prev_len] * 0.5
        
    # Selection
    if isinstance(num_top_tokens, float) and num_top_tokens <= 1.0:
        k = int(token_scores.shape[-1] * num_top_tokens)
    else:
        k = int(num_top_tokens)
    k = min(max(k, 1), token_scores.shape[-1])
    
    _, top_token_idx = torch.topk(token_scores, dim=-1, k=k)
    return top_token_idx, token_scores