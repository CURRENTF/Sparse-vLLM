from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    max_tokens: int = 64
    ignore_eos: bool = False
    eos_token_ids: int | list[int] | tuple[int, ...] | None = None
    logprobs: int | None = None

    def __post_init__(self):
        if self.temperature < 0.0:
            raise ValueError("temperature must be non-negative")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k < 0:
            raise ValueError("top_k must be non-negative")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        if self.eos_token_ids is None:
            self.eos_token_ids = ()
        elif isinstance(self.eos_token_ids, int):
            self.eos_token_ids = (int(self.eos_token_ids),)
        else:
            self.eos_token_ids = tuple(dict.fromkeys(int(token_id) for token_id in self.eos_token_ids))
        if any(token_id < 0 for token_id in self.eos_token_ids):
            raise ValueError("eos_token_ids must contain non-negative token ids")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError("logprobs must be non-negative")
