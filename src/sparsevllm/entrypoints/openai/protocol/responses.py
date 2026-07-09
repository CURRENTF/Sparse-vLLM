from typing import Any
from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class ResponseReasoning(BaseModel):
    model_config = ConfigDict(extra="forbid")

    effort: Literal["none", "minimal", "low", "medium", "high", "xhigh"] | None = None
    summary: str | None = None


class ResponseRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    input: str | list[dict[str, Any]]
    instructions: str | None = None
    max_output_tokens: int | None = Field(default=None, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    stream: bool = False
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    reasoning: ResponseReasoning | None = None
    chat_template_kwargs: Any = None

