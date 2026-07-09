from typing import Any

from pydantic import BaseModel
from pydantic import ConfigDict

from sparsevllm.entrypoints.openai.protocol.chat import ChatMessage


class PrefixCacheInspectRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ids: list[int] | None = None
    text: str | None = None
    include_subtree: bool = False


class PrefixCacheMatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ids: list[int] | None = None
    text: str | None = None
    messages: list[ChatMessage] | None = None
    response: dict[str, Any] | None = None


class PrefixCacheDeleteSubtreeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ids: list[int] | None = None
    text: str | None = None


class PrefixCacheSetEvictionPriorityRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    token_ids: list[int] | None = None
    text: str | None = None
    priority: int

