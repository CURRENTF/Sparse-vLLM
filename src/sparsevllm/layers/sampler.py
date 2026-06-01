import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def _sample(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_ps: torch.Tensor,
        top_ks: torch.Tensor,
    ):
        logits = logits.float()
        greedy_mask = temperatures <= 1e-10
        safe_temperatures = torch.where(greedy_mask, torch.ones_like(temperatures), temperatures)
        sampled_logits = logits.div(safe_temperatures.unsqueeze(dim=1))

        sorted_logits, sorted_indices = torch.sort(sampled_logits, dim=-1, descending=True)
        vocab_size = sorted_logits.shape[-1]
        top_k_limits = torch.where(
            top_ks <= 0,
            torch.full_like(top_ks, vocab_size),
            top_ks.clamp(max=vocab_size),
        )
        top_k_positions = torch.arange(vocab_size, device=logits.device).unsqueeze(0)
        top_k_mask = top_k_positions >= top_k_limits.unsqueeze(1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        top_p_mask = cumulative_probs > top_ps.unsqueeze(dim=1)
        top_p_mask[:, 1:] = top_p_mask[:, :-1].clone()
        top_p_mask[:, 0] = False
        remove_mask = top_p_mask | top_k_mask
        sampled_logits = sorted_logits.masked_fill(remove_mask, -torch.inf)

        probs = torch.softmax(sampled_logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        sample_tokens = sorted_indices.gather(1, sample_tokens.unsqueeze(1)).squeeze(1)
        greedy_tokens = logits.argmax(dim=-1)
        return torch.where(greedy_mask, greedy_tokens, sample_tokens)

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor | None,
        top_ps: torch.Tensor | None = None,
        top_ks: torch.Tensor | None = None,
        all_greedy: bool = False,
    ):
        if all_greedy:
            return logits.argmax(dim=-1)
        if temperatures is None:
            raise ValueError("temperatures must be provided when all_greedy=False")
        if top_ps is None:
            raise ValueError("top_ps must be provided when all_greedy=False")
        if top_ks is None:
            top_ks = torch.zeros_like(temperatures, dtype=torch.int64)
        return self._sample(logits, temperatures, top_ps, top_ks)
