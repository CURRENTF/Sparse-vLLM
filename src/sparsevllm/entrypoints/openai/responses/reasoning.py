from dataclasses import dataclass
from typing import Callable


class ReasoningParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedReasoning:
    reasoning_text: str | None
    output_text: str
    incomplete_reasoning: bool = False


ReasoningParser = Callable[[str, str | None], ParsedReasoning]


def get_reasoning_parser(name: str | None) -> ReasoningParser | None:
    if name is None:
        return None
    if name == "qwen3":
        return parse_qwen3_reasoning
    raise ValueError(f"Unsupported reasoning parser {name!r}.")


def parse_qwen3_reasoning(text: str, finish_reason: str | None) -> ParsedReasoning:
    stripped = text.lstrip()
    leading_len = len(text) - len(stripped)
    if stripped.startswith("<think>"):
        start = leading_len + len("<think>")
        end = text.find("</think>", start)
        if end >= 0:
            return ParsedReasoning(
                reasoning_text=text[start:end],
                output_text=_clean_qwen3_output_text(text[end + len("</think>"):]),
            )

        if finish_reason == "length":
            return ParsedReasoning(
                reasoning_text=text[start:],
                output_text="",
                incomplete_reasoning=True,
            )
        raise ReasoningParseError("Qwen3 reasoning output opened <think> but did not close </think>.")

    end = text.find("</think>")
    if end >= 0:
        return ParsedReasoning(
            reasoning_text=text[:end],
            output_text=_clean_qwen3_output_text(text[end + len("</think>"):]),
        )
    return ParsedReasoning(reasoning_text=None, output_text=_clean_qwen3_output_text(text))


def _clean_qwen3_output_text(text: str) -> str:
    cleaned = text
    for token in ("<|im_end|>", "<|endoftext|>"):
        if token in cleaned:
            cleaned = cleaned.replace(token, "")
    return cleaned.lstrip()
