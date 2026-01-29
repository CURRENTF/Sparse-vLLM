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
        self.last_token = token_ids[-1] if token_ids else None
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_prefilled_tokens = 0
        self.current_chunk_size = None

        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos

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
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def is_last_chunk_prefill(self):
        return (self.num_prefilled_tokens + self.current_chunk_size) >= self.num_prompt_tokens

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def __getstate__(self):
        # 优化 IPC：不发送 slot_mapping，只发送元数据和必要的 token
        if self.num_completion_tokens == 0:
            chunk_size = self.current_chunk_size if self.current_chunk_size is not None else self.num_prompt_tokens
            data = self.token_ids[self.num_prefilled_tokens : self.num_prefilled_tokens + chunk_size]
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
            self.max_tokens,
            self.ignore_eos,
            data,
        )

    def __setstate__(self, state):
        (self.seq_id, self.status, self.num_tokens, self.num_prompt_tokens,
         self.num_prefilled_tokens, self.current_chunk_size, self.temperature,
         self.max_tokens, self.ignore_eos, data) = state

        if self.num_completion_tokens == 0:
            self.token_ids = data
            self.last_token = self.token_ids[-1] if self.token_ids else None
        else:
            self.last_token = data
            self.token_ids = []
