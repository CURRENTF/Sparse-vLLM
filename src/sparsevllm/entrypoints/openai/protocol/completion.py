from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class CompletionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model: str
    prompt: str | list[int] | list[str] | list[list[int]]
    max_tokens: int = Field(default=16, ge=1)
    temperature: float = Field(default=1.0, ge=0.0)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    top_k: int = Field(default=0, ge=0)
    n: int = Field(default=1, ge=1)
    stream: bool = False
    ignore_eos: bool = False
    stop: str | list[str] | None = None
    logprobs: int | None = Field(default=None, ge=0, le=5)

