import argparse
import importlib.metadata
import json
import os
from dataclasses import dataclass
from typing import Any, Optional


MATH_VERIFY_METRIC = "math_verify"


@dataclass
class ParsedAnswer:
    parsed: list[Any]
    extraction_target: str
    source_field: Optional[str] = None
    source_text: Optional[str] = None
    error: Optional[str] = None


def _load_math_verify():
    try:
        from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
    except ModuleNotFoundError as e:
        raise RuntimeError(
            "benchmark/math_bench/eval.py now requires `math-verify`. "
            "Install it in the active environment with `pip install math-verify`."
        ) from e
    return parse, verify, LatexExtractionConfig, ExprExtractionConfig


def _math_verify_version() -> str:
    try:
        return importlib.metadata.version("math-verify")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _prediction_extraction_config() -> tuple[list[Any], str]:
    _, _, LatexExtractionConfig, ExprExtractionConfig = _load_math_verify()
    return (
        [ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)],
        "ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)",
    )


def _gold_latex_config() -> tuple[list[Any], str]:
    _, _, LatexExtractionConfig, _ = _load_math_verify()
    return [LatexExtractionConfig()], "LatexExtractionConfig()"


def _gold_expr_latex_config() -> tuple[list[Any], str]:
    _, _, LatexExtractionConfig, ExprExtractionConfig = _load_math_verify()
    return [ExprExtractionConfig(), LatexExtractionConfig()], "ExprExtractionConfig(), LatexExtractionConfig()"


def _serialize_parsed(values: Optional[list[Any]]) -> list[dict[str, str]]:
    if not values:
        return []
    return [{"type": type(value).__name__, "str": str(value), "repr": repr(value)} for value in values]


def _primary_parsed_str(values: Optional[list[Any]]) -> Optional[str]:
    if not values:
        return None
    return str(values[0])


def _parse_with_math_verify(
    text: str,
    *,
    extraction_config: list[Any],
    fallback_mode: str,
    extraction_mode: str,
    parsing_timeout: float,
    raise_on_error: bool,
) -> list[Any]:
    parse, _, _, _ = _load_math_verify()
    return parse(
        text,
        extraction_config=extraction_config,
        fallback_mode=fallback_mode,
        extraction_mode=extraction_mode,
        parsing_timeout=int(parsing_timeout),
        raise_on_error=raise_on_error,
    )


def _field_value(gold: dict[str, Any], wanted_key: str) -> tuple[Optional[str], Any]:
    for key, value in gold.items():
        if str(key).lower() == wanted_key:
            return str(key), value
    return None, None


def _gold_candidate_keys(dataset: str) -> tuple[str, ...]:
    if dataset == "math500":
        return ("solution", "answer", "final_answer", "target", "label")
    if dataset == "gsm8k":
        return ("answer", "final_answer", "target", "label", "solution", "rationale")
    if dataset == "aime2024":
        return ("answer", "final_answer", "target", "label", "output", "result")
    if dataset == "hmmt_nov":
        return ("answer", "final_answer", "target", "label", "solution")
    return ("solution", "answer", "final_answer", "target", "label", "output", "result")


def _gold_text_candidates(dataset: str, gold: Any) -> list[tuple[str, str]]:
    if isinstance(gold, (str, int, float)):
        return [("gold", str(gold))]
    if not isinstance(gold, dict):
        return []

    candidates: list[tuple[str, str]] = []
    for wanted_key in _gold_candidate_keys(dataset):
        source_field, value = _field_value(gold, wanted_key)
        if source_field is None:
            continue
        if isinstance(value, (int, float)):
            candidates.append((source_field, str(value)))
        elif isinstance(value, str) and value.strip():
            candidates.append((source_field, value.strip()))
    return candidates


def _gold_config_for_field(dataset: str, source_field: str) -> tuple[list[Any], str]:
    if dataset == "math500" and source_field.lower() == "solution":
        return _gold_latex_config()
    return _gold_expr_latex_config()


def parse_prediction(text: str, *, parsing_timeout: float = 5.0) -> ParsedAnswer:
    config, config_desc = _prediction_extraction_config()
    parsed = _parse_with_math_verify(
        text,
        extraction_config=config,
        fallback_mode="first_match",
        extraction_mode="any_match",
        parsing_timeout=parsing_timeout,
        raise_on_error=False,
    )
    return ParsedAnswer(parsed=parsed, extraction_target=config_desc)


def parse_gold_answer(dataset: str, gold: Any, *, parsing_timeout: float = 5.0) -> ParsedAnswer:
    candidates = _gold_text_candidates(dataset, gold)
    if not candidates:
        return ParsedAnswer(parsed=[], extraction_target="", error="missing_gold_candidate")

    errors: list[str] = []
    for source_field, source_text in candidates:
        config, config_desc = _gold_config_for_field(dataset, source_field)
        try:
            parsed = _parse_with_math_verify(
                source_text,
                extraction_config=config,
                fallback_mode="first_match",
                extraction_mode="any_match",
                parsing_timeout=parsing_timeout,
                raise_on_error=True,
            )
        except Exception as e:  # math-verify can raise on malformed gold latex.
            errors.append(f"{source_field}: {type(e).__name__}: {e}")
            continue
        if parsed:
            return ParsedAnswer(
                parsed=parsed,
                extraction_target=config_desc,
                source_field=source_field,
                source_text=source_text,
            )
        errors.append(f"{source_field}: no parsed value")

    return ParsedAnswer(parsed=[], extraction_target="", error="; ".join(errors))


def verify_math_answer(
    gold_parsed: list[Any],
    pred_parsed: list[Any],
    *,
    verify_timeout: float = 5.0,
) -> bool:
    _, verify, _, _ = _load_math_verify()
    return bool(
        verify(
            gold_parsed,
            pred_parsed,
            timeout_seconds=int(verify_timeout),
            raise_on_error=True,
        )
    )


def extract_pred_answer(text: str) -> Optional[str]:
    return _primary_parsed_str(parse_prediction(text).parsed)


def extract_gold_answer(dataset: str, gold: Any) -> Optional[str]:
    return _primary_parsed_str(parse_gold_answer(dataset, gold).parsed)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True, help="Prediction folder containing *.jsonl")
    parser.add_argument("--parse_timeout", type=int, default=5, help="Per-row math-verify parse timeout in seconds")
    parser.add_argument("--verify_timeout", type=int, default=5, help="Per-row math-verify verify timeout in seconds")
    return parser.parse_args()


def _should_skip_jsonl(filename: str) -> bool:
    return (
        filename.endswith("_per_sample_results.jsonl")
        or filename.endswith("_parsed_outputs.jsonl")
        or filename.startswith("raw_outputs")
    )


def _record_invalid_input(rec: Any, dataset: str) -> tuple[dict[str, Any], dict[str, Any]]:
    parsed_record = {
        "id": rec.get("id") if isinstance(rec, dict) else None,
        "dataset": dataset,
        "status": "invalid_input",
        "pred_parsed": [],
        "gold_parsed": [],
        "error": "prediction row must be a JSON object with string field `pred`",
    }
    sample_record = {
        "id": parsed_record["id"],
        "dataset": dataset,
        "status": "invalid_input",
        "pred_answer": None,
        "gold_answer": None,
        "correct": False,
        "error": parsed_record["error"],
    }
    return parsed_record, sample_record


def evaluate_record(
    rec: Any,
    *,
    dataset: str,
    parse_timeout: float,
    verify_timeout: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not isinstance(rec, dict) or not isinstance(rec.get("pred"), str):
        return _record_invalid_input(rec, dataset)

    pred_text = rec["pred"]
    gold = rec.get("gold", {})
    pred_answer = parse_prediction(pred_text, parsing_timeout=parse_timeout)
    gold_answer = parse_gold_answer(dataset, gold, parsing_timeout=parse_timeout)

    status = "success"
    correct = False
    error = None
    if not gold_answer.parsed:
        status = "metric_failed"
        error = gold_answer.error or "gold answer could not be parsed"
    elif not pred_answer.parsed:
        status = "parse_failed"
        error = "prediction answer could not be parsed"
    else:
        try:
            correct = verify_math_answer(gold_answer.parsed, pred_answer.parsed, verify_timeout=verify_timeout)
        except Exception as e:
            status = "metric_failed"
            error = f"{type(e).__name__}: {e}"

    parsed_record = {
        "id": rec.get("id"),
        "dataset": dataset,
        "status": status,
        "pred_parsed": _serialize_parsed(pred_answer.parsed),
        "gold_parsed": _serialize_parsed(gold_answer.parsed),
        "pred_extraction_target": pred_answer.extraction_target,
        "gold_extraction_target": gold_answer.extraction_target,
        "gold_source_field": gold_answer.source_field,
    }
    sample_record = {
        "id": rec.get("id"),
        "dataset": dataset,
        "status": status,
        "pred_answer": _primary_parsed_str(pred_answer.parsed),
        "gold_answer": _primary_parsed_str(gold_answer.parsed),
        "correct": correct,
        "metric": MATH_VERIFY_METRIC,
        "gold_source_field": gold_answer.source_field,
    }
    if error:
        parsed_record["error"] = error
        sample_record["error"] = error
    return parsed_record, sample_record


def main() -> None:
    args = parse_args()
    path = args.path
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    # Fail fast before truncating any existing per-sample files.
    _load_math_verify()

    scores = {}
    for filename in sorted(os.listdir(path)):
        if not filename.endswith(".jsonl") or _should_skip_jsonl(filename):
            continue
        dataset = filename.split(".")[0]
        total = 0
        correct = 0
        missing = 0
        status_counts: dict[str, int] = {}
        input_path = os.path.join(path, filename)
        parsed_path = os.path.join(path, f"{dataset}_parsed_outputs.jsonl")
        per_sample_path = os.path.join(path, f"{dataset}_per_sample_results.jsonl")

        with open(input_path, "r", encoding="utf-8") as f, open(
            parsed_path, "w", encoding="utf-8"
        ) as parsed_out, open(per_sample_path, "w", encoding="utf-8") as per_sample:
            for line_no, line in enumerate(f, start=1):
                if not line.strip():
                    continue
                rec = json.loads(line)
                parsed_record, sample_record = evaluate_record(
                    rec,
                    dataset=dataset,
                    parse_timeout=args.parse_timeout,
                    verify_timeout=args.verify_timeout,
                )
                parsed_record["line_no"] = line_no
                sample_record["line_no"] = line_no

                status = str(sample_record["status"])
                total += 1
                if status != "success":
                    missing += 1
                if sample_record["correct"]:
                    correct += 1
                status_counts[status] = status_counts.get(status, 0) + 1

                json.dump(parsed_record, parsed_out, ensure_ascii=False)
                parsed_out.write("\n")
                json.dump(sample_record, per_sample, ensure_ascii=False)
                per_sample.write("\n")

        acc = 0.0 if total == 0 else (correct / total * 100.0)
        pred_config_desc = _prediction_extraction_config()[1]
        scores[dataset] = {
            "pass@1": round(acc, 2),
            "correct": correct,
            "total": total,
            "missing_extracted": missing,
            "status_counts": status_counts,
            "metric": MATH_VERIFY_METRIC,
            "math_verify_package": f"math-verify=={_math_verify_version()}",
            "pred_extraction_target": pred_config_desc,
            "gold_source": "dataset-specific candidates; math500 uses solution with answer fallback",
            "parsed_outputs": parsed_path,
            "per_sample_results": per_sample_path,
        }

    out_path = os.path.join(path, "result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)
    print(json.dumps(scores, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
