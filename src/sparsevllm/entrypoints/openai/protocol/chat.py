from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ChatContentPart(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["text"]
    text: str


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: Literal["developer", "system", "user", "assistant", "tool"]
    content: str | list[ChatContentPart] | None = None
    tool_call_id: str | None = None
    tool_calls: list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    messages: list[ChatMessage]
    max_tokens: int = Field(default=16, ge=1)
    max_completion_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    ignore_eos: bool = False
    stop: str | list[str] | None = None
    logprobs: bool = False
    top_logprobs: int | None = Field(default=None, ge=0, le=20)
    stream_options: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    chat_template_kwargs: Any = None

