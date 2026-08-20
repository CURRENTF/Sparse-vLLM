"""Shared final-token budgeting for LongBench runners."""

from __future__ import annotations

from typing import Any


def encode_prompt_with_generation_budget(
    tokenizer: Any,
    prompt: str,
    *,
    max_model_len: int,
    max_gen: int,
) -> list[int]:
    """Encode the rendered prompt and retain its head/tail within the runtime budget."""
    max_model_len = int(max_model_len)
    max_gen = int(max_gen)
    if max_model_len <= 0:
        raise ValueError(f"max_model_len must be positive, got {max_model_len}.")
    if max_gen <= 0:
        raise ValueError(f"max_gen must be positive, got {max_gen}.")

    prompt_budget = max_model_len - max_gen
    if prompt_budget <= 0:
        raise ValueError(
            "LongBench generation leaves no prompt budget: "
            f"max_model_len={max_model_len}, max_gen={max_gen}."
        )

    add_special_tokens = bool(
        tokenizer.bos_token is not None and not prompt.startswith(tokenizer.bos_token)
    )
    token_ids = [
        int(token_id)
        for token_id in tokenizer.encode(
            prompt,
            add_special_tokens=add_special_tokens,
        )
    ]
    if len(token_ids) > prompt_budget:
        head = prompt_budget // 2
        tail = prompt_budget - head
        token_ids = token_ids[:head] + token_ids[-tail:]

    if not token_ids:
        raise ValueError("LongBench prompt tokenization produced no tokens.")
    if len(token_ids) + max_gen > max_model_len:
        raise RuntimeError(
            "LongBench prompt budget invariant failed: "
            f"prompt_tokens={len(token_ids)}, max_gen={max_gen}, "
            f"max_model_len={max_model_len}."
        )
    return token_ids
