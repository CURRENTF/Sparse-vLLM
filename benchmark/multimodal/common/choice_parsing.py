from __future__ import annotations

import re


CHOICE_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def first_choice_letter(text: str, *, valid: str = "ABCD") -> str:
    match = re.search(r"\b([A-Z])\b", str(text).upper())
    if match and match.group(1) in set(valid):
        return match.group(1)
    return ""


def status_for_choice(choice: str) -> str:
    return "success" if choice else "parse_failed"
