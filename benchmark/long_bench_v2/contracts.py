"""Pure-data contracts for the LongBench v2 regression runner."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable


REQUIRED_FIELDS = (
    "_id",
    "domain",
    "sub_domain",
    "difficulty",
    "length",
    "question",
    "choice_A",
    "choice_B",
    "choice_C",
    "choice_D",
    "answer",
    "context",
)
OFFICIAL_DIFFICULTIES = frozenset({"easy", "hard"})
OFFICIAL_LENGTHS = frozenset({"short", "medium", "long"})
OFFICIAL_ANSWERS = frozenset({"A", "B", "C", "D"})
ANSWER_RE = re.compile(r"The correct answer is \(?([A-D])\)?")


@dataclass(frozen=True)
class TokenBucket:
    name: str
    min_prompt_tokens: int
    max_prompt_tokens: int
    samples: int

    def contains(self, prompt_tokens: int) -> bool:
        return self.min_prompt_tokens <= prompt_tokens <= self.max_prompt_tokens


def parse_token_buckets(value: str | Iterable[dict[str, Any]]) -> tuple[TokenBucket, ...]:
    if isinstance(value, str):
        try:
            raw = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"token buckets are not valid JSON: {exc}") from exc
    else:
        raw = list(value)
    if not isinstance(raw, list) or not raw:
        raise ValueError("token buckets must be a non-empty JSON list.")

    buckets: list[TokenBucket] = []
    names: set[str] = set()
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"token bucket {index} must be a JSON object.")
        name = item.get("name")
        minimum = item.get("min_prompt_tokens")
        maximum = item.get("max_prompt_tokens")
        samples = item.get("samples")
        if not isinstance(name, str) or not name:
            raise ValueError(f"token bucket {index} name must be a non-empty string.")
        if name in names:
            raise ValueError(f"duplicate token bucket name: {name!r}.")
        for key, number in (
            ("min_prompt_tokens", minimum),
            ("max_prompt_tokens", maximum),
            ("samples", samples),
        ):
            if not isinstance(number, int) or isinstance(number, bool) or number <= 0:
                raise ValueError(f"token bucket {name!r} {key} must be a positive integer.")
        if minimum > maximum:
            raise ValueError(
                f"token bucket {name!r} min_prompt_tokens exceeds max_prompt_tokens."
            )
        bucket = TokenBucket(name, minimum, maximum, samples)
        for previous in buckets:
            if not (
                bucket.max_prompt_tokens < previous.min_prompt_tokens
                or bucket.min_prompt_tokens > previous.max_prompt_tokens
            ):
                raise ValueError(
                    f"token buckets {previous.name!r} and {bucket.name!r} overlap."
                )
        buckets.append(bucket)
        names.add(name)
    return tuple(buckets)


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"LongBench v2 data file does not exist: {source}")
    if source.suffix.lower() == ".jsonl":
        rows: list[Any] = []
        with source.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"invalid JSONL at {source}:{line_number}: {exc}") from exc
    elif source.suffix.lower() == ".json":
        with source.open("r", encoding="utf-8") as handle:
            rows = json.load(handle)
        if isinstance(rows, dict) and isinstance(rows.get("data"), list):
            rows = rows["data"]
    else:
        raise ValueError(
            f"LongBench v2 data must be .json or .jsonl, got {source.suffix!r}."
        )
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"LongBench v2 data must contain a non-empty list: {source}")

    validated: list[dict[str, Any]] = []
    observed_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"LongBench v2 row {index} must be a JSON object.")
        missing = [field for field in REQUIRED_FIELDS if field not in row]
        if missing:
            raise ValueError(f"LongBench v2 row {index} is missing fields: {missing}.")
        sample_id = row["_id"]
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"LongBench v2 row {index} has invalid _id={sample_id!r}.")
        if sample_id in observed_ids:
            raise ValueError(f"LongBench v2 contains duplicate _id={sample_id!r}.")
        if row["difficulty"] not in OFFICIAL_DIFFICULTIES:
            raise ValueError(
                f"LongBench v2 row {sample_id!r} has invalid difficulty={row['difficulty']!r}."
            )
        if row["length"] not in OFFICIAL_LENGTHS:
            raise ValueError(
                f"LongBench v2 row {sample_id!r} has invalid length={row['length']!r}."
            )
        if row["answer"] not in OFFICIAL_ANSWERS:
            raise ValueError(
                f"LongBench v2 row {sample_id!r} has invalid answer={row['answer']!r}."
            )
        for field in REQUIRED_FIELDS[1:]:
            if not isinstance(row[field], str):
                raise ValueError(
                    f"LongBench v2 row {sample_id!r} field {field!r} must be a string."
                )
        observed_ids.add(sample_id)
        validated.append(dict(row))
    return validated


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def render_prompt(template: str, sample: dict[str, Any]) -> str:
    replacements = {
        "$DOC$": sample["context"].strip(),
        "$Q$": sample["question"].strip(),
        "$C_A$": sample["choice_A"].strip(),
        "$C_B$": sample["choice_B"].strip(),
        "$C_C$": sample["choice_C"].strip(),
        "$C_D$": sample["choice_D"].strip(),
    }
    for placeholder in replacements:
        if placeholder not in template:
            raise ValueError(f"LongBench v2 prompt template is missing {placeholder}.")
    placeholders = set(re.findall(r"\$[A-Z_]+\$", template))
    unknown = sorted(placeholders - set(replacements))
    if unknown:
        raise ValueError(f"LongBench v2 prompt has unknown placeholders: {unknown}.")
    return re.sub(
        r"\$[A-Z_]+\$",
        lambda match: replacements[match.group(0)],
        template,
    )


def extract_answer(response: str) -> str | None:
    if not isinstance(response, str):
        raise TypeError(f"LongBench v2 response must be a string, got {type(response).__name__}.")
    match = ANSWER_RE.search(response.replace("*", ""))
    return match.group(1) if match else None


def select_samples(
    rows: list[dict[str, Any]],
    *,
    buckets: tuple[TokenBucket, ...],
    seed: int,
    prepare_prompt: Callable[[dict[str, Any]], tuple[str, list[int]]],
    max_prompt_tokens: int,
) -> list[dict[str, Any]]:
    """Select a deterministic, token-stratified subset without truncating prompts."""
    candidates: dict[str, list[dict[str, Any]]] = {bucket.name: [] for bucket in buckets}
    candidate_counts = {bucket.name: 0 for bucket in buckets}
    for source_index, row in enumerate(rows):
        prompt, prompt_token_ids = prepare_prompt(row)
        prompt_tokens = len(prompt_token_ids)
        if prompt_tokens <= 0 or prompt_tokens > max_prompt_tokens:
            continue
        matching = [bucket for bucket in buckets if bucket.contains(prompt_tokens)]
        if not matching:
            continue
        bucket = matching[0]
        candidate_counts[bucket.name] += 1
        candidates[bucket.name].append(
            {
                "source_index": source_index,
                "sample": row,
                "prompt": prompt,
                "prompt_token_ids": prompt_token_ids,
                "prompt_tokens": prompt_tokens,
                "token_bucket": bucket.name,
                "selection_key": hashlib.sha256(
                    f"{int(seed)}:{row['_id']}".encode("utf-8")
                ).hexdigest(),
            }
        )
        candidates[bucket.name].sort(
            key=lambda item: (item["selection_key"], item["sample"]["_id"])
        )
        if len(candidates[bucket.name]) > bucket.samples:
            candidates[bucket.name].pop()

    selected: list[dict[str, Any]] = []
    for bucket in buckets:
        available = candidates[bucket.name]
        if candidate_counts[bucket.name] < bucket.samples:
            raise ValueError(
                f"LongBench v2 token bucket {bucket.name!r} has insufficient untruncated "
                f"samples: required={bucket.samples} available={candidate_counts[bucket.name]} "
                f"range=[{bucket.min_prompt_tokens}, {bucket.max_prompt_tokens}]."
            )
        selected.extend(available)
    for index, item in enumerate(selected):
        item["index"] = index
    return selected


def aggregate_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise ValueError("LongBench v2 aggregate requires at least one sample.")
    parsed = [row for row in rows if row.get("status") == "success"]
    parse_failed = [row for row in rows if row.get("status") == "parse_failed"]
    execution_failed = [
        row
        for row in rows
        if row.get("status") not in {"success", "parse_failed"}
    ]

    def group(field: str) -> dict[str, dict[str, Any]]:
        values: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            values.setdefault(str(row[field]), []).append(row)
        return {
            value: {
                "samples": len(group_rows),
                "correct": sum(bool(item.get("correct")) for item in group_rows),
                "accuracy": 100.0
                * sum(bool(item.get("correct")) for item in group_rows)
                / len(group_rows),
            }
            for value, group_rows in sorted(values.items())
        }

    correct = sum(bool(row.get("correct")) for row in rows)
    return {
        "status": "success" if not execution_failed else "failed",
        "samples": len(rows),
        "evaluated_samples": len(parsed) + len(parse_failed),
        "successful_samples": len(parsed),
        "parse_failed_samples": len(parse_failed),
        "failed_samples": len(execution_failed),
        "correct": correct,
        "accuracy": 100.0 * correct / len(rows),
        "by_difficulty": group("difficulty"),
        "by_official_length": group("official_length"),
        "by_token_bucket": group("token_bucket"),
        "by_domain": group("domain"),
    }
