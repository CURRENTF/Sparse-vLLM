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

from datasets import load_dataset
from eval_utils import DATA_NAME_TO_MAX_NEW_TOKENS, dump_jsonl
from run_scbench import load_model
from tqdm import tqdm

sys.path.insert(0, str(REPO_ROOT / "baselines" / "kvzip"))

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
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--use_sparq", action="store_true")
    parser.add_argument("--topk", type=int, default=-1)
    parser.add_argument("--starting_layer", type=int, default=-1)
    parser.add_argument("--topk_dims_file_path", type=str, default=None)
    parser.add_argument("--kv_cache_cpu", action="store_true")
    parser.add_argument("--kv_cache_cpu_device", type=str, default="cpu")
    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--copy_on_gpu", action="store_true")
    parser.add_argument("--is_search", action="store_true")
    parser.add_argument("--attn_type", type=str, default="hf")
    parser.add_argument("--kv_type", type=str, default="dense")
    parser.add_argument("--hyper_param", type=json.loads, default={})
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


def truncate_input(input_ids: list[int], max_length: int) -> list[int]:
    if max_length < 0 or len(input_ids) <= max_length:
        return input_ids
    split = max_length // 2
    return input_ids[:split] + input_ids[-split:]


def build_kvzip_scdq_example(
    row: dict[str, Any],
    model_name: str,
    data_name: str,
    tokenizer,
    max_input_length: int,
):
    prefix, postfix = kvzip_template(model_name, data_name)
    context_prompt = prefix + row["prompts"][0]
    context_ids = tokenizer.encode(context_prompt, add_special_tokens=False)
    prompts = [truncate_input(context_ids, max_input_length)]
    prompts.extend(f"\n\nQ: {query.strip()}{postfix}" for query in row["prompts"][1:])

    example = {
        "prompts": prompts,
        "ground_truth": row["ground_truth"],
    }
    if "task" in row:
        example["task"] = row["task"]
    return example


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


if __name__ == "__main__":
    args = parse_args()

    if "," in args.task:
        data_names = args.task.split(",")
    else:
        data_names = [args.task]

    model_name = args.model_name_or_path
    real_model_name = model_name.split("/")[-1]

    model, tok = load_model(
        model_name,
        args.topk,
        args.starting_layer,
        args.topk_dims_file_path,
        args.use_sparq,
        attn_type=args.attn_type,
        max_seq_length=args.max_seq_length,
        is_search=args.is_search,
        kv_type=args.kv_type,
        trust_remote_code=args.trust_remote_code,
        kv_cache_cpu=args.kv_cache_cpu,
        kv_cache_cpu_device=args.kv_cache_cpu_device,
        tensor_parallel_size=args.tensor_parallel_size,
        hyper_param=args.hyper_param.copy(),
        copy_on_gpu=args.copy_on_gpu,
    )

    verbalize_hyper_param = (
        f"_{'-'.join([f'{k}={v}' for k, v in args.hyper_param.items() if k != 'best_pattern'])}"
        if args.hyper_param
        else ""
    )
    verbalize_hyper_param = shorten_val(verbalize_hyper_param)
    result_dir = Path(
        args.output_dir,
        f"{real_model_name}_{args.attn_type}_scdq_preprocessed{verbalize_hyper_param}",
    )
    result_dir.mkdir(exist_ok=True, parents=True)

    results = {}
    for data_name in data_names:
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
            if isinstance(max_new_tokens, dict):
                tokens_to_sum = sum(list(max_new_tokens.values()))
            else:
                tokens_to_sum = max_new_tokens
            max_input_length = args.max_seq_length - (tokens_to_sum * max_turn_size // 2)
        else:
            max_input_length = args.max_seq_length - max_new_tokens * max_turn_size

        print(f"==== Evaluation {data_name} ====")
        print(f"# examples: {len(dataset)}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"Max input length: {max_input_length}")

        preds = []
        perfs = []
        for idx in tqdm(range(len(dataset))):
            if idx < args.start_example_id:
                continue

            row = dataset[idx]
            example = build_kvzip_scdq_example(
                row,
                model_name,
                data_name,
                tok,
                max_input_length,
            )
            output = model.test_scdq(example, max_length=max_new_tokens)

            refs, subtask = build_refs(row, data_name)
            perf = evaluate_answer(
                output["answers"],
                refs,
                data_name,
                "qa",
                subtask=subtask,
            )
            perfs.append(perf)

            record = {
                "id": idx,
                "answers": output["answers"],
                "gt": row["ground_truth"],
                "score_raw": perf,
            }
            if "task" in row:
                record["task"] = row["task"]
            preds.append(record)

        dump_jsonl(preds, output_path)
        score = round(mean_nested(perfs) * 100, 2)
        results[data_name] = score
        print(f"{data_name}: {score}")

    with open(result_dir / "result.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(json.dumps(results, indent=2, ensure_ascii=False))
