import argparse
import json
import os
import re
from typing import Optional


def _find_last_boxed(text: str) -> Optional[str]:
    # Supports "\boxed{...}" and "\boxed ...".
    if not text:
        return None
    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    after = text[idx + len("\\boxed") :].lstrip()
    if after.startswith("{"):
        # Parse balanced braces.
        depth = 0
        buf = []
        for ch in after:
            if ch == "{":
                depth += 1
                if depth == 1:
                    continue
            if ch == "}":
                depth -= 1
                if depth == 0:
                    return "".join(buf).strip()
            if depth >= 1:
                buf.append(ch)
        return None

    # "\boxed 123" style: take until end of line.
    line = after.splitlines()[0]
    return line.strip() if line.strip() else None


def _normalize_answer(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("$", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mathrm\{([^}]*)\}", r"\1", s)
    s = s.replace(",", "").strip()

    if re.fullmatch(r"[-+]?\d+", s):
        return str(int(s))

    if re.fullmatch(r"[-+]?\d+\.\d+", s):
        try:
            x = float(s)
        except Exception:
            return s
        if x.is_integer():
            return str(int(x))
        s2 = f"{x:.12f}".rstrip("0").rstrip(".")
        return s2

    return s


def _extract_last_number(text: str) -> Optional[str]:
    if not text:
        return None
    matches = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    return matches[-1] if matches else None


def extract_pred_answer(text: str, allow_unboxed: bool) -> Optional[str]:
    boxed = _find_last_boxed(text)
    if boxed is not None:
        return _normalize_answer(boxed)
    if not allow_unboxed:
        return None
    last_num = _extract_last_number(text)
    if last_num is None:
        return None
    return _normalize_answer(last_num)


def extract_gold_answer(dataset: str, gold: dict) -> Optional[str]:
    # Support both lowercase/uppercase dataset column names.
    gold_norm = {}
    if isinstance(gold, dict):
        for k, v in gold.items():
            gold_norm[str(k).lower()] = v

    # Try common fields first.
    for key in ("final_answer", "answer", "target", "label"):
        val = gold_norm.get(key, None)
        if isinstance(val, (int, float)):
            return _normalize_answer(str(val))
        if isinstance(val, str) and val.strip():
            # GSM8K uses ".... #### 18"
            if "####" in val:
                tail = val.split("####")[-1]
                num = _extract_last_number(tail)
                if num is not None:
                    return _normalize_answer(num)
            boxed = _find_last_boxed(val)
            if boxed is not None:
                return _normalize_answer(boxed)
            num = _extract_last_number(val)
            if num is not None:
                return _normalize_answer(num)

    # Dataset-specific fallbacks.
    if dataset == "gsm8k":
        for key in ("solution", "rationale"):
            val = gold_norm.get(key, None)
            if isinstance(val, str) and "####" in val:
                tail = val.split("####")[-1]
                num = _extract_last_number(tail)
                if num is not None:
                    return _normalize_answer(num)

    if dataset == "aime2024":
        for key in ("output", "result"):
            val = gold_norm.get(key, None)
            if isinstance(val, str):
                num = _extract_last_number(val)
                if num is not None:
                    return _normalize_answer(num)

    return None


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Prediction folder containing *.jsonl")
    parser.add_argument("--allow_unboxed", action="store_true", help="Fallback to last number when no \\\\boxed{} found")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.path
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    scores = {}
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".jsonl"):
            continue
        dataset = filename.split(".")[0]
        total = 0
        correct = 0
        missing = 0

        with open(os.path.join(path, filename), "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                pred = rec.get("pred", "")
                gold = rec.get("gold", {})
                pred_ans = extract_pred_answer(pred, allow_unboxed=args.allow_unboxed)
                gold_ans = extract_gold_answer(dataset, gold if isinstance(gold, dict) else {})
                if pred_ans is None or gold_ans is None:
                    missing += 1
                    total += 1
                    continue
                total += 1
                if pred_ans == gold_ans:
                    correct += 1

        acc = 0.0 if total == 0 else (correct / total * 100.0)
        scores[dataset] = {
            "pass@1": round(acc, 2),
            "correct": correct,
            "total": total,
            "missing_extracted": missing,
        }

    out_path = os.path.join(path, "result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
