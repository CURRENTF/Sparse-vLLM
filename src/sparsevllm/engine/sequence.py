import torch
from copy import copy
from enum import Enum, auto
from itertools import count

from sparsevllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


# 一部分属性会在 scheduler 动态修改
class Sequence:
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self._prompt_token_ids: tuple[int, ...] | None = tuple(
            int(token_id) for token_id in token_ids
        )
        self.last_token = token_ids[-1] if token_ids else None
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_prefilled_tokens = 0
        self.current_chunk_size = None
        self.prefix_cache_enabled = False
        self.prefix_cache_hit_len = 0
        self.prefix_cache_hit_block_count = 0
        self.prefix_cache_hit_last_block_id: bytes | None = None
        self.prefix_cache_block_size = 0
        self.prefix_cache_method = ""
        self.chain_id: str | None = None
        self.chain_status = "disabled"
        self.chain_reused_tokens = 0
        # None means normal generation. During recompute replay the original
        # prompt is prefetched again, then accepted completion tokens before the
        # current last token are replayed through decode without being sampled
        # or returned to the caller.
        self.recompute_replay_cursor: int | None = None
        # Completion count at the last residency handoff/replay rebuild. A
        # sole decode may yield to another replay only after exceeding this
        # checkpoint; otherwise two full-cache requests can replay forever
        # without publishing a new token.
        self.decode_progress_checkpoint = 0

        self.temperature = sampling_params.temperature
        self.top_p = sampling_params.top_p
        self.top_k = sampling_params.top_k
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.eos_token_ids = tuple(sampling_params.eos_token_ids)
        self.logprobs = sampling_params.logprobs
        self.completion_token_logprobs: list[float | None] = []
        self.completion_top_logprobs: list[dict[int, float] | None] = []

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, i):
        return self.token_ids[i]

    @property
    def kv_change_state(self):
        if self.num_prefilled_tokens == 0:
            return 'first_prefill'
        elif self.num_prefilled_tokens < self.num_prompt_tokens:
            return 'prefill'
        elif self.num_prefilled_tokens == self.num_prompt_tokens:
            return 'decode'

        raise ValueError

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        if self._prompt_token_ids is not None:
            return self._prompt_token_ids
        return tuple(self.token_ids[:self.num_prompt_tokens])

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def is_recompute_replay(self) -> bool:
        return self.recompute_replay_cursor is not None

    @property
    def is_recompute_prefill(self) -> bool:
        return self.is_recompute_replay and self.num_prefilled_tokens < self.num_prompt_tokens

    @property
    def is_recompute_decode(self) -> bool:
        return self.is_recompute_replay and self.num_prefilled_tokens >= self.num_prompt_tokens

    @property
    def replay_decode_token_count(self) -> int:
        # The last accepted token remains the normal next decode input. Only
        # earlier completion tokens need their KV reconstructed.
        return max(0, int(self.num_completion_tokens) - 1)

    @property
    def replay_input_token_ids(self) -> list[int]:
        if not self.is_recompute_prefill:
            return []
        start = int(self.num_prefilled_tokens)
        end = start + int(self.current_chunk_size or 0)
        if len(self.token_ids) == end - start:
            # TP workers receive only the active chunk over IPC.
            return [int(token_id) for token_id in self.token_ids]
        return [int(token_id) for token_id in self.token_ids[start:end]]

    @property
    def decode_input_token(self) -> int:
        if not self.is_recompute_decode:
            if self.last_token is None:
                raise RuntimeError(f"Decode input token is missing for seq_id={self.seq_id}.")
            return int(self.last_token)
        cursor = int(self.recompute_replay_cursor or 0)
        if cursor >= self.replay_decode_token_count:
            raise RuntimeError(
                "Recompute replay cursor is past the accepted decode history: "
                f"seq_id={self.seq_id} cursor={cursor} "
                f"replay_decode_tokens={self.replay_decode_token_count}."
            )
        if len(self.token_ids) > self.num_prompt_tokens + cursor:
            return int(self.token_ids[self.num_prompt_tokens + cursor])
        if self.last_token is None:
            raise RuntimeError(f"Replay decode input token is missing for seq_id={self.seq_id}.")
        # TP workers receive only the active replay token over IPC.
        return int(self.last_token)

    @property
    def decode_input_position(self) -> int:
        if self.is_recompute_decode:
            return int(self.num_prompt_tokens + int(self.recompute_replay_cursor or 0))
        return int(self.num_tokens - 1)

    @property
    def should_publish_sample(self) -> bool:
        return not self.is_recompute_replay

    def start_recompute_replay(self) -> None:
        if self.num_completion_tokens <= 0:
            raise RuntimeError(
                "Recompute replay requires at least one accepted completion token: "
                f"seq_id={self.seq_id} completion_tokens={self.num_completion_tokens}."
            )
        self.recompute_replay_cursor = 0
        self.decode_progress_checkpoint = int(self.num_completion_tokens)
        self.num_prefilled_tokens = 0
        self.current_chunk_size = None
        self.status = SequenceStatus.WAITING
        self.clear_prefix_cache_hit()

    def finish_recompute_replay(self) -> None:
        if not self.is_recompute_replay:
            raise RuntimeError(f"Sequence {self.seq_id} is not in recompute replay.")
        self.recompute_replay_cursor = None
        self.decode_progress_checkpoint = int(self.num_completion_tokens)

    def advance_recompute_replay(self) -> None:
        if not self.is_recompute_decode:
            raise RuntimeError(f"Sequence {self.seq_id} is not replaying decode tokens.")
        self.recompute_replay_cursor = int(self.recompute_replay_cursor or 0) + 1
        if int(self.recompute_replay_cursor) >= self.replay_decode_token_count:
            self.finish_recompute_replay()

    def clear_prefix_cache_hit(self):
        self.prefix_cache_enabled = False
        self.prefix_cache_hit_len = 0
        self.prefix_cache_hit_block_count = 0
        self.prefix_cache_hit_last_block_id = None
        self.prefix_cache_block_size = 0
        self.prefix_cache_method = ""

    @property
    def is_last_chunk_prefill(self):
        return (self.num_prefilled_tokens + self.current_chunk_size) >= self.num_prompt_tokens

    def append_token(
        self,
        token_id: int,
        logprob: float | None = None,
        top_logprobs: dict[int, float] | None = None,
    ):
        self.token_ids.append(token_id)
        if self.num_completion_tokens >= 0:
            self.completion_token_logprobs.append(logprob)
            self.completion_top_logprobs.append(top_logprobs)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # 优化 IPC：不发送 slot_mapping，只发送元数据和必要的 token
        if self.num_completion_tokens == 0 or self.is_recompute_prefill:
            chunk_size = self.current_chunk_size if self.current_chunk_size is not None else self.num_prompt_tokens
            data = self.token_ids[self.num_prefilled_tokens : self.num_prefilled_tokens + chunk_size]
        elif self.is_recompute_decode:
            data = self.decode_input_token
        else:
            data = self.last_token

        return (
            self.seq_id,
            self.status,
            self.num_tokens,
            self.num_prompt_tokens,
            self.num_prefilled_tokens,
            self.current_chunk_size,
            self.temperature,
            self.top_p,
            self.top_k,
            self.max_tokens,
            self.ignore_eos,
            self.eos_token_ids,
            self.logprobs,
            data,
            self.prefix_cache_enabled,
            self.prefix_cache_hit_len,
            self.prefix_cache_hit_block_count,
            self.prefix_cache_hit_last_block_id,
            self.prefix_cache_block_size,
            self.prefix_cache_method,
            self.chain_id,
            self.chain_status,
            self.chain_reused_tokens,
            self.recompute_replay_cursor,
            self.decode_progress_checkpoint,
        )

    def __setstate__(self, state):
        (self.seq_id, self.status, self.num_tokens, self.num_prompt_tokens,
         self.num_prefilled_tokens, self.current_chunk_size, self.temperature,
         self.top_p, self.top_k, self.max_tokens, self.ignore_eos, self.eos_token_ids,
         self.logprobs, data,
         self.prefix_cache_enabled, self.prefix_cache_hit_len, self.prefix_cache_hit_block_count,
         self.prefix_cache_hit_last_block_id, self.prefix_cache_block_size, self.prefix_cache_method,
         self.chain_id, self.chain_status, self.chain_reused_tokens,
         self.recompute_replay_cursor, self.decode_progress_checkpoint) = state
        self.completion_token_logprobs = []
        self.completion_top_logprobs = []

        if self.num_completion_tokens == 0 or self.is_recompute_prefill:
            self.token_ids = data
            self.last_token = self.token_ids[-1] if self.token_ids else None
            self._prompt_token_ids = (
                tuple(int(token_id) for token_id in data)
                if self.num_prefilled_tokens == 0
                and len(data) == self.num_prompt_tokens
                else None
            )
        else:
            self.last_token = data
            self.token_ids = []
            self._prompt_token_ids = None
