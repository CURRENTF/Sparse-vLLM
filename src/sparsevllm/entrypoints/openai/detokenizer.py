from dataclasses import dataclass
from typing import Any

from tokenizers.decoders import DecodeStream


@dataclass(frozen=True)
class DecodedDelta:
    text: str
    raw_text: str


@dataclass(frozen=True)
class DecodedFinal:
    text: str
    raw_text: str
    text_delta: str
    raw_text_delta: str


class IncrementalDetokenizer:
    def __init__(self, tokenizer: Any):
        backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
        if not getattr(tokenizer, "is_fast", False) or backend_tokenizer is None:
            raise TypeError(
                "OpenAI serving requires a fast tokenizer backend with "
                f"DecodeStream support; got {type(tokenizer).__name__}."
            )
        self.tokenizer = tokenizer
        self.backend_tokenizer = backend_tokenizer
        self.visible_stream = DecodeStream(skip_special_tokens=True)
        self.raw_stream = DecodeStream(skip_special_tokens=False)
        self.token_ids: list[int] = []
        self.text = ""
        self.raw_text = ""
        self.finished = False

    def push(self, token_ids: list[int]) -> DecodedDelta:
        if self.finished:
            raise RuntimeError("Cannot push token IDs after incremental detokenization finished.")

        text_parts: list[str] = []
        raw_text_parts: list[str] = []
        for token_id in token_ids:
            token_id = int(token_id)
            self.token_ids.append(token_id)
            text = self.visible_stream.step(self.backend_tokenizer, token_id)
            raw_text = self.raw_stream.step(self.backend_tokenizer, token_id)
            if text:
                text_parts.append(text)
            if raw_text:
                raw_text_parts.append(raw_text)

        text_delta = "".join(text_parts)
        raw_text_delta = "".join(raw_text_parts)
        self.text += text_delta
        self.raw_text += raw_text_delta
        return DecodedDelta(text=text_delta, raw_text=raw_text_delta)

    def finish(self, token_ids: list[int]) -> DecodedFinal:
        if self.finished:
            raise RuntimeError("Incremental detokenization already finished.")

        final_token_ids = [int(token_id) for token_id in token_ids]
        observed = len(self.token_ids)
        if len(final_token_ids) < observed or final_token_ids[:observed] != self.token_ids:
            raise RuntimeError(
                "Incremental detokenization token history mismatch: "
                f"observed={self.token_ids!r} final={final_token_ids!r}."
            )
        if len(final_token_ids) > observed:
            self.push(final_token_ids[observed:])

        final_text = self.tokenizer.decode(final_token_ids, skip_special_tokens=True)
        final_raw_text = self.tokenizer.decode(final_token_ids, skip_special_tokens=False)
        if not final_text.startswith(self.text):
            raise RuntimeError(
                "Incremental visible text is not a prefix of canonical final text: "
                f"incremental={self.text!r} final={final_text!r}."
            )
        if not final_raw_text.startswith(self.raw_text):
            raise RuntimeError(
                "Incremental raw text is not a prefix of canonical final text: "
                f"incremental={self.raw_text!r} final={final_raw_text!r}."
            )

        text_delta = final_text[len(self.text):]
        raw_text_delta = final_raw_text[len(self.raw_text):]
        self.text = final_text
        self.raw_text = final_raw_text
        self.finished = True
        return DecodedFinal(
            text=final_text,
            raw_text=final_raw_text,
            text_delta=text_delta,
            raw_text_delta=raw_text_delta,
        )
