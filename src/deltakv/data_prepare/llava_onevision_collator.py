from __future__ import annotations

import io
import re
from collections import deque
from pathlib import Path
from typing import Any

import torch
from PIL import Image


IMAGE_TOKEN_RE = re.compile(r"<image>", flags=re.IGNORECASE)


def _load_image(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value.convert("RGB")
    if isinstance(value, dict):
        if value.get("bytes") is not None:
            return Image.open(io.BytesIO(value["bytes"])).convert("RGB")
        if value.get("path"):
            path = Path(value["path"])
            if not path.exists() or path.stat().st_size == 0:
                raise FileNotFoundError(f"LLaVA sample references a missing/empty image: {path}")
            return Image.open(path).convert("RGB")
    if isinstance(value, (bytes, bytearray)):
        return Image.open(io.BytesIO(value)).convert("RGB")
    if isinstance(value, str):
        path = Path(value)
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(f"LLaVA sample references a missing/empty image: {path}")
        return Image.open(path).convert("RGB")
    raise TypeError(f"Unsupported image value type for LLaVA sample: {type(value)!r}")


def _extract_images(sample: dict[str, Any]) -> list[Image.Image]:
    if "images" in sample and sample["images"] is not None:
        raw_images = sample["images"]
        if not isinstance(raw_images, list):
            raise TypeError(f"`images` must be a list when present, got {type(raw_images)!r}")
        images = [_load_image(item) for item in raw_images]
    elif "image" in sample and sample["image"] is not None:
        images = [_load_image(sample["image"])]
    else:
        raise ValueError(f"LLaVA sample has no `image` or `images` field; keys={sorted(sample.keys())}")
    if not images:
        raise ValueError("LLaVA sample contains an empty image list.")
    return images


def _normalize_role(role: str) -> str:
    role = str(role).strip().lower()
    if role in {"human", "user"}:
        return "user"
    if role in {"gpt", "assistant"}:
        return "assistant"
    if role == "system":
        return "system"
    raise ValueError(f"Unsupported LLaVA conversation role: {role!r}")


def _conversation_items(sample: dict[str, Any]) -> list[dict[str, Any]]:
    conversations = sample.get("conversations", sample.get("messages"))
    if not isinstance(conversations, list) or not conversations:
        raise ValueError(f"LLaVA sample requires a non-empty `conversations` or `messages` list; keys={sorted(sample.keys())}")
    return conversations


def _build_content(text: str, *, add_missing_image: bool) -> tuple[list[dict[str, str]], int]:
    parts: list[dict[str, str]] = []
    image_count = 0
    cursor = 0
    for match in IMAGE_TOKEN_RE.finditer(text):
        segment = text[cursor : match.start()].strip()
        if segment:
            parts.append({"type": "text", "text": segment})
        parts.append({"type": "image"})
        image_count += 1
        cursor = match.end()
    tail = text[cursor:].strip()
    if tail:
        parts.append({"type": "text", "text": tail})
    if add_missing_image and image_count == 0:
        parts.insert(0, {"type": "image"})
        image_count = 1
    return parts, image_count


def _to_chat(sample: dict[str, Any], num_images: int) -> list[dict[str, Any]]:
    chat = []
    seen_images = 0
    for raw_message in _conversation_items(sample):
        role = _normalize_role(raw_message.get("from", raw_message.get("role", "")))
        text = str(raw_message.get("value", raw_message.get("content", ""))).strip()
        if not text:
            raise ValueError(f"LLaVA sample has an empty message for role={role!r}; id={sample.get('id')!r}")
        if role == "user":
            content, image_count = _build_content(text, add_missing_image=seen_images == 0 and num_images > 0)
            seen_images += image_count
        else:
            content = [{"type": "text", "text": text}]
        chat.append({"role": role, "content": content})
    if seen_images != num_images:
        raise ValueError(
            f"LLaVA sample image-token count does not match images: image_tokens={seen_images}, "
            f"images={num_images}, id={sample.get('id')!r}, source={sample.get('data_source')!r}"
        )
    return chat


def _find_subsequence(sequence: list[int], subsequence: list[int], *, start: int = 0) -> int:
    if not subsequence:
        raise ValueError("Cannot search for an empty token subsequence.")
    stop = len(sequence) - len(subsequence) + 1
    for idx in range(start, stop):
        if sequence[idx : idx + len(subsequence)] == subsequence:
            return idx
    return -1


class LlavaOnevisionOnlineCollator:
    def __init__(
        self,
        processor,
        *,
        max_length: int = -1,
        image_processor_use_fast: bool = False,
        label_pad_token_id: int = -100,
        replacement_buffer_size: int = 64,
    ):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = int(max_length)
        self.image_processor_use_fast = bool(image_processor_use_fast)
        self.label_pad_token_id = int(label_pad_token_id)
        self.replacement_buffer_size = int(replacement_buffer_size)
        if self.replacement_buffer_size < 1:
            raise ValueError(f"replacement_buffer_size must be >= 1, got {self.replacement_buffer_size}")
        self.tokenizer.padding_side = "right"
        if self.tokenizer.pad_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._assistant_header_id_variants = [
            self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False),
            self.tokenizer.encode("<|im_start|>assistant \n", add_special_tokens=False),
        ]
        self._message_end_ids = self.tokenizer.encode("<|im_end|>", add_special_tokens=False)
        if not all(self._assistant_header_id_variants) or not self._message_end_ids:
            raise ValueError("Failed to tokenize LLaVA assistant header or message end marker for label masking.")
        self._seen_samples = 0
        self._skipped_by_length = 0
        self._replacement_buffer = deque(maxlen=self.replacement_buffer_size)
        self._all_skipped_batches = 0
        self._replacement_samples = 0

    def _build_assistant_labels(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
        labels = torch.full_like(input_ids, self.label_pad_token_id)
        for row_idx in range(input_ids.shape[0]):
            if attention_mask is not None:
                valid_len = int(attention_mask[row_idx].sum().item())
            else:
                valid_len = input_ids.shape[1]
            token_ids = input_ids[row_idx, :valid_len].tolist()
            cursor = 0
            span_count = 0
            while True:
                header_start = -1
                header_ids = None
                for candidate in self._assistant_header_id_variants:
                    candidate_start = _find_subsequence(token_ids, candidate, start=cursor)
                    if candidate_start >= 0 and (header_start < 0 or candidate_start < header_start):
                        header_start = candidate_start
                        header_ids = candidate
                if header_start < 0:
                    break
                content_start = header_start + len(header_ids)
                end_start = _find_subsequence(token_ids, self._message_end_ids, start=content_start)
                if end_start < 0:
                    raise ValueError("Could not find `<|im_end|>` after an assistant header while building LLaVA labels.")
                content_end = end_start + len(self._message_end_ids)
                if content_start < content_end:
                    labels[row_idx, content_start:content_end] = input_ids[row_idx, content_start:content_end]
                    span_count += 1
                cursor = content_end
            if span_count == 0:
                raise ValueError("No assistant span found while building LLaVA labels.")
        return labels

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        if not batch:
            raise ValueError("LLaVA collator received an empty batch.")
        prompts = []
        batch_images = []
        valid_entries = []
        for sample in batch:
            self._seen_samples += 1
            images = _extract_images(sample)
            chat = _to_chat(sample, len(images))
            prompt = self.processor.apply_chat_template(chat, add_generation_prompt=False)
            seq_len = -1
            if self.max_length > 0:
                probe = self.processor(text=[prompt], images=[images], padding=True, return_tensors="pt")
                seq_len = int(probe["input_ids"].shape[1])
                if seq_len > self.max_length:
                    self._skipped_by_length += 1
                    if self._skipped_by_length <= 5 or self._skipped_by_length % 100 == 0:
                        print(
                            "[LLaVA collator] skipped sample longer than max_length: "
                            f"seq_len={seq_len} max_length={self.max_length} "
                            f"skipped={self._skipped_by_length} seen={self._seen_samples} "
                            f"id={sample.get('id')!r} source={sample.get('data_source')!r}",
                            flush=True,
                        )
                    continue
            entry = {
                "prompt": prompt,
                "images": images,
                "seq_len": seq_len,
                "id": sample.get("id"),
                "source": sample.get("data_source"),
            }
            valid_entries.append(entry)
            self._replacement_buffer.append(entry)
        if not prompts:
            if valid_entries:
                prompts = [entry["prompt"] for entry in valid_entries]
                batch_images = [entry["images"] for entry in valid_entries]
            elif self._replacement_buffer:
                self._all_skipped_batches += 1
                replacement_count = min(len(batch), len(self._replacement_buffer))
                replacements = list(self._replacement_buffer)[-replacement_count:]
                prompts = [entry["prompt"] for entry in replacements]
                batch_images = [entry["images"] for entry in replacements]
                self._replacement_samples += replacement_count
                if self._all_skipped_batches <= 5 or self._all_skipped_batches % 100 == 0:
                    replacement_ids = [entry["id"] for entry in replacements]
                    print(
                        "[LLaVA collator] replaced all-skipped batch from buffer: "
                        f"batch_size={len(batch)} replacement_count={replacement_count} "
                        f"max_length={self.max_length} all_skipped_batches={self._all_skipped_batches} "
                        f"replacement_samples={self._replacement_samples} skipped={self._skipped_by_length} "
                        f"seen={self._seen_samples} replacement_ids={replacement_ids!r}",
                        flush=True,
                    )
            else:
                raise ValueError(
                    "All samples in this LLaVA batch exceeded max_length and no replacement buffer is available; "
                    f"max_length={self.max_length}, skipped={self._skipped_by_length}, seen={self._seen_samples}."
                )

        kwargs: dict[str, Any] = {
            "text": prompts,
            "images": batch_images,
            "padding": True,
            "return_tensors": "pt",
        }
        features = self.processor(**kwargs)
        if self.max_length > 0 and features["input_ids"].shape[1] > self.max_length:
            raise ValueError(
                "LLaVA multimodal token truncation is unsafe because it can break image-token alignment; "
                f"got sequence length {features['input_ids'].shape[1]} > max_length {self.max_length}. "
                "Use a larger max_length, set data_max_len=-1, or prefilter long samples explicitly."
            )
        attention_mask = features.get("attention_mask")
        labels = self._build_assistant_labels(features["input_ids"], attention_mask)
        features["labels"] = labels
        return features
