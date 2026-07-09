from dataclasses import dataclass
from typing import Callable
from typing import Literal


class ReasoningParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedReasoning:
    reasoning_text: str | None
    output_text: str
    incomplete_reasoning: bool = False


@dataclass(frozen=True)
class ReasoningStreamEvent:
    kind: Literal["reasoning_delta", "answer_delta", "reasoning_done"]
    text: str = ""


ReasoningParser = Callable[[str, str | None], ParsedReasoning]

_THINK_START = "<think>"
_THINK_END = "</think>"
_SPECIAL_OUTPUT_TOKENS = ("<|im_end|>", "<|endoftext|>")


def get_reasoning_parser(name: str | None) -> ReasoningParser | None:
    if name is None:
        return None
    if name == "qwen3":
        return parse_qwen3_reasoning
    raise ValueError(f"Unsupported reasoning parser {name!r}.")


def get_reasoning_stream_parser(name: str | None):
    if name is None:
        return PlainReasoningStreamParser()
    if name == "qwen3":
        return Qwen3ReasoningStreamParser()
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


class PlainReasoningStreamParser:
    incomplete_reasoning = False

    def feed(self, text_delta: str) -> list[ReasoningStreamEvent]:
        if not text_delta:
            return []
        return [ReasoningStreamEvent("answer_delta", text_delta)]

    def finish(self, finish_reason: str | None) -> list[ReasoningStreamEvent]:
        del finish_reason
        return []


class Qwen3ReasoningStreamParser:
    def __init__(self):
        self._state = "content"
        self._buffer = ""
        self._answer_filter = _SpecialTokenFilter(_SPECIAL_OUTPUT_TOKENS)
        self.incomplete_reasoning = False

    def feed(self, text_delta: str) -> list[ReasoningStreamEvent]:
        if not text_delta:
            return []
        if self._state == "content":
            return self._feed_content(text_delta)
        if self._state == "reasoning":
            return self._feed_reasoning(text_delta)
        if self._state == "answer":
            return self._answer_delta(text_delta)
        raise AssertionError(f"unknown Qwen3 reasoning stream state {self._state!r}")

    def finish(self, finish_reason: str | None) -> list[ReasoningStreamEvent]:
        if self._state == "content":
            events = self._answer_delta(self._buffer)
            self._buffer = ""
            return events + self._flush_answer()
        if self._state == "reasoning":
            if finish_reason == "length":
                events = []
                if self._buffer:
                    events.append(ReasoningStreamEvent("reasoning_delta", self._buffer))
                    self._buffer = ""
                self.incomplete_reasoning = True
                events.append(ReasoningStreamEvent("reasoning_done"))
                return events
            raise ReasoningParseError("Qwen3 reasoning output opened <think> but did not close </think>.")
        if self._state == "answer":
            return self._flush_answer()
        raise AssertionError(f"unknown Qwen3 reasoning stream state {self._state!r}")

    def _feed_content(self, text_delta: str) -> list[ReasoningStreamEvent]:
        self._buffer += text_delta
        stripped = self._buffer.lstrip()
        if not stripped:
            return []
        if stripped.startswith(_THINK_START):
            self._buffer = stripped[len(_THINK_START):]
            self._state = "reasoning"
            return self._feed_reasoning("")
        if _THINK_START.startswith(stripped):
            return []
        end = self._buffer.find(_THINK_END)
        if end >= 0:
            reasoning_text = self._buffer[:end]
            answer_text = self._buffer[end + len(_THINK_END):].lstrip()
            self._buffer = ""
            self._state = "answer"
            events = []
            if reasoning_text:
                events.append(ReasoningStreamEvent("reasoning_delta", reasoning_text))
            events.append(ReasoningStreamEvent("reasoning_done"))
            return events + self._answer_delta(answer_text)

        answer_text = self._buffer
        self._buffer = ""
        self._state = "answer"
        return self._answer_delta(answer_text)

    def _feed_reasoning(self, text_delta: str) -> list[ReasoningStreamEvent]:
        combined = self._buffer + text_delta
        end = combined.find(_THINK_END)
        if end >= 0:
            reasoning_text = combined[:end]
            answer_text = combined[end + len(_THINK_END):].lstrip()
            self._buffer = ""
            self._state = "answer"
            events = []
            if reasoning_text:
                events.append(ReasoningStreamEvent("reasoning_delta", reasoning_text))
            events.append(ReasoningStreamEvent("reasoning_done"))
            return events + self._answer_delta(answer_text)

        keep = _suffix_prefix_len(combined, (_THINK_END,))
        if keep:
            emit_text = combined[:-keep]
            self._buffer = combined[-keep:]
        else:
            emit_text = combined
            self._buffer = ""
        if not emit_text:
            return []
        return [ReasoningStreamEvent("reasoning_delta", emit_text)]

    def _answer_delta(self, text: str) -> list[ReasoningStreamEvent]:
        if not text:
            return []
        filtered = self._answer_filter.feed(text)
        if not filtered:
            return []
        return [ReasoningStreamEvent("answer_delta", filtered)]

    def _flush_answer(self) -> list[ReasoningStreamEvent]:
        filtered = self._answer_filter.finish()
        if not filtered:
            return []
        return [ReasoningStreamEvent("answer_delta", filtered)]


class _SpecialTokenFilter:
    def __init__(self, tokens: tuple[str, ...]):
        self._tokens = tokens
        self._buffer = ""

    def feed(self, text: str) -> str:
        combined = self._buffer + text
        keep = _suffix_prefix_len(combined, self._tokens)
        if keep:
            emit_text = combined[:-keep]
            self._buffer = combined[-keep:]
        else:
            emit_text = combined
            self._buffer = ""
        return self._remove_tokens(emit_text)

    def finish(self) -> str:
        emit_text = self._buffer
        self._buffer = ""
        return self._remove_tokens(emit_text)

    def _remove_tokens(self, text: str) -> str:
        for token in self._tokens:
            text = text.replace(token, "")
        return text


def _suffix_prefix_len(text: str, candidates: tuple[str, ...]) -> int:
    best = 0
    for candidate in candidates:
        max_len = min(len(candidate) - 1, len(text))
        for size in range(max_len, 0, -1):
            if text.endswith(candidate[:size]):
                best = max(best, size)
                break
    return best
