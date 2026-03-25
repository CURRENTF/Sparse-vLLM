from __future__ import annotations

import json
import os
import sys
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE_PATH = "/root/autodl-fs/deltakv_outputs"

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent

sys.path.append(str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "baselines" / "kvzip"))

from datasets import load_dataset
from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS, dump_jsonl
from tqdm import tqdm

from data.wrapper import get_query
from model import ModelKVzip
from results.metric import evaluate_answer


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument(
        "--data_root",
        type=str,
        default="/root/autodl-fs/datasets/SCBench-preprocessed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=f"{BASE_PATH}/benchmark/scbench_preprocessed",
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--num_eval_examples", type=int, default=-1)
    parser.add_argument("--start_example_id", type=int, default=0)
    parser.add_argument("--max_seq_length", type=int, default=131072)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--level", type=str, default="pair")
    parser.add_argument("--kv_type", type=str, default="retain")
    parser.add_argument("--prefill_chunk_size", type=int, default=16000)
    return parser.parse_args()


def kvzip_template(model_name: str, task_name: str) -> tuple[str, str]:
    model_name = Path(model_name.rstrip("/")).name.lower()

    if "llama" in model_name or model_name == "duo":
        prefix = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        prefix += "You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        postfix = "\n\n<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    elif model_name.startswith("qwen"):
        prefix = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
        prefix += "<|im_start|>user\n"
        postfix = "<|im_end|>\n<|im_start|>assistant\n"
        if "qwen3-" in model_name:
            postfix += "<think>\n\n</think>\n\n"
    elif model_name.startswith("gemma3") or model_name.startswith("gemma-3"):
        prefix = "<bos><start_of_turn>user\n"
        prefix += "You are a helpful assistant.\n\n"
        postfix = "<end_of_turn>\n<start_of_turn>model\n"
    else:
        prefix = "<|begin_of_text|>"
        postfix = "\n\nAnswer: "

    if task_name.startswith("gsm"):
        prefix += "Given the context, answer to the following reasoning question.\n\n"
    else:
        prefix += "Given the context, answer to the following question or request without explanation.\n\n"

    return prefix, postfix


def truncate_tensor(input_ids, max_length: int):
    if max_length < 0 or input_ids.shape[1] <= max_length:
        return input_ids

    split = max_length // 2
    if split == 0:
        return input_ids[:, -max_length:]
    return input_ids[:, :split].contiguous().clone(), input_ids[:, -split:].contiguous().clone()


def truncate_context_ids(input_ids, max_length: int):
    truncated = truncate_tensor(input_ids, max_length)
    if isinstance(truncated, tuple):
        return torch.cat(truncated, dim=1)
    return truncated


def build_refs(row: dict[str, Any], data_name: str):
    subtask = row.get("task")

    if "many_shot" in data_name:
        refs = []
        for prompt, gt in zip(row["prompts"][1:], row["ground_truth"]):
            candidates = [line.strip() for line in prompt.split("\n") if f"({gt})" in line]
            refs.append(candidates[0] if candidates else str(gt))
        return refs, subtask

    if "repoqa" in data_name:
        refs = defaultdict(list)
        refs["lang"] = row["lang"]
        refs["repo"] = row["repo"]
        refs["func_name"] = row["func_name"]
        refs["ground_truth"] = row["ground_truth"]
        return refs, subtask

    return row["ground_truth"], subtask


def mean_nested(perfs: list[Any]) -> float:
    def _mean_one(perf: Any) -> float:
        if isinstance(perf, (int, float, bool)):
            return float(perf)
        if not perf:
            return 0.0
        return sum(float(v) for v in perf) / len(perf)

    if not perfs:
        return 0.0
    return sum(_mean_one(perf) for perf in perfs) / len(perfs)


def shorten_val(value: Any) -> str:
    value = str(value)
    if "/" in value:
        value = os.path.basename(value.rstrip("/"))
    if len(value) > 40:
        value = value[:20] + ".." + value[-15:]
    return value


def get_max_new_tokens(max_new_tokens, subtask, idx: int):
    if isinstance(max_new_tokens, dict):
        task_name = subtask[idx]
        return max_new_tokens[task_name]
    return max_new_tokens


def get_context_ids(model: ModelKVzip, row: dict[str, Any], data_name: str, max_input_length: int):
    prefix, _ = kvzip_template(model.name, data_name)
    prefix_ids = model.encode(prefix)
    ctx_ids = model.encode(row["prompts"][0])
    max_context_length = max_input_length - prefix_ids.shape[1]
    return truncate_context_ids(ctx_ids, max_context_length)


if __name__ == "__main__":
    import torch

    args = parse_args()
    data_names = args.task.split(",") if "," in args.task else [args.task]

    model = ModelKVzip(args.model_name_or_path, kv_type=args.kv_type)
    real_model_name = Path(args.model_name_or_path.rstrip("/")).name

    verbalize_hyper_param = shorten_val(
        f"_ratio={args.ratio}-level={args.level}-kv_type={args.kv_type}"
    )
    result_dir = Path(
        args.output_dir,
        f"{real_model_name}_kvzip_scdq_preprocessed{verbalize_hyper_param}",
    )
    result_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    for data_name in data_names:
        model.set_chat_template(data_name)
        dataset = load_dataset(
            "parquet",
            data_files=str(Path(args.data_root) / f"{data_name}.parquet"),
            split="train",
        )
        if args.num_eval_examples != -1:
            dataset = dataset.select(range(min(args.num_eval_examples, len(dataset))))

        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        output_path = result_dir / f"prediction_{data_name}_scdq.jsonl"
        max_turn_size = len(dataset[0]["ground_truth"])
        if data_name in ["scbench_summary_with_needles", "scbench_repoqa_and_kv"]:
            tokens_to_sum = (
                sum(list(max_new_tokens.values()))
                if isinstance(max_new_tokens, dict)
                else max_new_tokens
            )
            max_input_length = args.max_seq_length - (tokens_to_sum * max_turn_size // 2)
        else:
            max_input_length = args.max_seq_length - max_new_tokens * max_turn_size

        print(f"==== Evaluation {data_name} ====")
        print(f"# examples: {len(dataset)}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Max input length: {max_input_length}")

        preds = []
        perfs = []
        ratio_trues = []
        for idx in tqdm(range(len(dataset))):
            if idx < args.start_example_id:
                continue

            row = dataset[idx]
            ctx_ids = get_context_ids(model, row, data_name, max_input_length)
            do_score = args.ratio < 1.0
            kv = model.prefill(
                ctx_ids,
                prefill_chunk_size=args.prefill_chunk_size,
                do_score=do_score,
            )

            thres = None
            ratio_true = 1.0
            if args.ratio < 1.0:
                thres, ratio_true = kv.prune(args.ratio, args.level)
            ratio_trues.append(float(ratio_true))

            answers = []
            subtask = row.get("task")
            for q_idx, query in enumerate(row["prompts"][1:]):
                model.gen_kwargs["max_new_tokens"] = get_max_new_tokens(
                    max_new_tokens, subtask, q_idx
                )
                q_ids = model.apply_template(get_query("qa", query))
                answers.append(model.generate(q_ids, kv=kv, update_cache=False))

            refs, eval_subtask = build_refs(row, data_name)
            perf = evaluate_answer(
                answers,
                refs,
                data_name,
                "qa",
                subtask=eval_subtask,
            )
            perfs.append(perf)

            record = {
                "id": idx,
                "answers": answers,
                "gt": row["ground_truth"],
                "score_raw": perf,
                "ratio_true": ratio_true,
                "threshold": thres,
            }
            if "task" in row:
                record["task"] = row["task"]
            preds.append(record)

            sample_score = mean_nested([perf]) * 100
            if thres is None:
                print(
                    f"[{idx + 1}/{len(dataset)}] sample_score={sample_score:.2f}",
                    flush=True,
                )
            else:
                print(
                    f"[{idx + 1}/{len(dataset)}] sample_score={sample_score:.2f} "
                    f"ratio_true={ratio_true:.4f} thres={thres:.4f}",
                    flush=True,
                )

            del kv

        dump_jsonl(preds, output_path)
        score = round(mean_nested(perfs) * 100, 2)
        results[data_name] = score
        print(f"{data_name}: {score}")
        if args.ratio < 1.0:
            print(f"{data_name} avg_ratio_true: {sum(ratio_trues) / len(ratio_trues):.4f}")

    with open(result_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))
