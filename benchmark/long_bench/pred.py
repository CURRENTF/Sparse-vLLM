import os
import json
import sys
import subprocess
import re
import traceback
from typing import Any, Union
from pathlib import Path

from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.multiprocessing as mp
import torch
from transformers import AutoTokenizer, GenerationConfig
import torch.distributed as dist
from deltakv.get_chat_api import get_generate_api
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = os.getenv("DELTAKV_OUTPUT_DIR", str(REPO_ROOT / "outputs"))
DATA_PREFIX_PATH = os.getenv("DELTAKV_LONGBENCH_DATA_DIR") or os.getenv("DELTAKV_DATA_DIR")
NO_CHAT_TEMPLATE_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}
SAMPLE_STATUSES = {
    "success",
    "invalid_input",
    "model_failed",
    "parse_failed",
    "metric_failed",
    "skipped_by_policy",
}


def get_longbench_data_path(dataset, use_longbench_e):
    if not DATA_PREFIX_PATH:
        raise FileNotFoundError(
            "LongBench data root is not configured.\n"
            "Set DELTAKV_LONGBENCH_DATA_DIR or DELTAKV_DATA_DIR to the LongBench "
            "root directory that contains data/*.jsonl."
        )
    suffix = "_e" if use_longbench_e else ""
    return os.path.join(DATA_PREFIX_PATH, "data", f"{dataset}{suffix}.jsonl")


def validate_longbench_data_paths(datasets, use_longbench_e):
    if not DATA_PREFIX_PATH:
        raise FileNotFoundError(
            "LongBench data root is not configured.\n"
            "Set DELTAKV_LONGBENCH_DATA_DIR or DELTAKV_DATA_DIR to the LongBench "
            "root directory that contains data/*.jsonl."
        )
    if not os.path.isdir(DATA_PREFIX_PATH):
        raise FileNotFoundError(
            "LongBench data root does not exist: "
            f"{DATA_PREFIX_PATH}\n"
            "Set DELTAKV_LONGBENCH_DATA_DIR or DELTAKV_DATA_DIR to the LongBench root "
            "directory that contains data/*.jsonl."
        )

    data_dir = os.path.join(DATA_PREFIX_PATH, "data")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            "LongBench data directory does not exist: "
            f"{data_dir}\n"
            "Set DELTAKV_LONGBENCH_DATA_DIR or DELTAKV_DATA_DIR to the LongBench root "
            "directory that contains a data/ subdirectory."
        )

    missing_paths = [
        get_longbench_data_path(dataset, use_longbench_e)
        for dataset in datasets
        if not os.path.isfile(get_longbench_data_path(dataset, use_longbench_e))
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Missing LongBench dataset files:\n"
            + "\n".join(missing_paths)
            + "\nCheck DELTAKV_LONGBENCH_DATA_DIR / DELTAKV_DATA_DIR."
        )

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def should_use_chat_template(dataset, no_chat_template=False, thinking_mode="off"):
    return not no_chat_template and dataset not in NO_CHAT_TEMPLATE_DATASETS


def build_chat(tokenizer, prompt, dataset, no_chat_template=False, thinking_mode="off"):
    if not should_use_chat_template(dataset, no_chat_template, thinking_mode):
        return prompt
    if hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        msgs = [
            # {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]
        enable_thinking = thinking_mode != "off"
        prompt = tokenizer.apply_chat_template(
            msgs,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        # Some local Qwen3 tokenizer templates still end with an open `<think>` block
        # even when `enable_thinking=False`. Close it explicitly to force empty-thinking mode.
        if thinking_mode == "off" and prompt.endswith("<think>\n"):
            prompt += "</think>\n"
    if os.getenv('DEBUG'):
        print('input prompt:', prompt)
    return prompt


def strip_thinking_content(text: str) -> str:
    closing_tag = "</think>"
    if closing_tag not in text:
        raise ValueError(
            "Thinking output ended before </think>; increase max_new_tokens instead "
            "of scoring truncated reasoning."
        )
    return text.split(closing_tag, 1)[1].lstrip()


def _append_jsonl(path: str | os.PathLike[str], record: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


def _artifact_paths(out_root: str) -> dict[str, str]:
    return {
        "raw": os.path.join(out_root, "raw_outputs.jsonl"),
        "parsed": os.path.join(out_root, "parsed_outputs.jsonl"),
        "sample": os.path.join(out_root, "sample_results.jsonl"),
    }


def _sample_base_record(
    *,
    dataset: str,
    batch_offset: int,
    json_obj: dict[str, Any],
    prompt_tokens: int | None = None,
) -> dict[str, Any]:
    source_idx = json_obj.get("_longbench_source_idx")
    if source_idx is None:
        source_idx = json_obj.get("_source_idx", batch_offset)
    return {
        "dataset": dataset,
        "sample_idx": int(batch_offset),
        "source_idx": int(source_idx),
        "prompt_tokens": None if prompt_tokens is None else int(prompt_tokens),
        "answers": json_obj.get("answers"),
        "all_classes": json_obj.get("all_classes"),
        "length": json_obj.get("length"),
    }


def _write_sample_record(
    *,
    out_root: str,
    task_out_path: str,
    record: dict[str, Any],
) -> None:
    status = record.get("status")
    if status not in SAMPLE_STATUSES:
        raise ValueError(f"Invalid sample status {status!r}; expected one of {sorted(SAMPLE_STATUSES)}.")

    paths = _artifact_paths(out_root)
    raw_record = {
        key: record.get(key)
        for key in (
            "dataset",
            "sample_idx",
            "source_idx",
            "status",
            "prompt_tokens",
            "raw_pred",
            "error",
            "traceback",
        )
        if key in record
    }
    parsed_record = {
        key: record.get(key)
        for key in (
            "dataset",
            "sample_idx",
            "source_idx",
            "status",
            "prompt_tokens",
            "pred",
            "error",
        )
        if key in record
    }
    _append_jsonl(paths["raw"], raw_record)
    _append_jsonl(paths["parsed"], parsed_record)
    _append_jsonl(paths["sample"], record)

    # Keep the historical per-task files for benchmark/long_bench/eval.py.
    task_record = {
        "status": record["status"],
        "pred": record.get("pred", ""),
        "raw_pred": record.get("raw_pred", record.get("pred", "")),
        "answers": record.get("answers"),
        "all_classes": record.get("all_classes"),
        "length": record.get("length"),
        "prompt_tokens": record.get("prompt_tokens"),
        "source_idx": record.get("source_idx"),
    }
    if "error" in record:
        task_record["error"] = record["error"]
    _append_jsonl(task_out_path, task_record)


def build_kvzip_prompt_parts(prompt_format, json_obj, use_kvzip_template=True):
    if "{context}" not in prompt_format:
        raise ValueError("KVzip LongBench adapter requires '{context}' in the prompt template.")

    pre_context, post_context = prompt_format.split("{context}", 1)
    format_fields = dict(json_obj)
    context_text = format_fields.pop("context")

    context_prefix = pre_context.format(**format_fields)
    query_suffix = post_context.format(**format_fields)
    return {
        "prefill_text": context_prefix + context_text,
        "query_text": query_suffix,
        "use_kvzip_template": use_kvzip_template,
    }


def load_model_and_tokenizer(rank, args):
    infer_config = {
        'max_model_len': args.max_model_len,
    }

    if args.hyper_param:
        if os.path.exists(args.hyper_param):
            with open(args.hyper_param, 'r') as f:
                extra_config = json.load(f)
            infer_config.update(extra_config)
            print(f"Loaded hyper-parameters from {args.hyper_param}: {extra_config}")
        else:
            # Try to parse as JSON string
            try:
                extra_config = json.loads(args.hyper_param)
                infer_config.update(extra_config)
                print(f"Parsed hyper-parameters from string: \n{extra_config}")
            except json.JSONDecodeError as e:
                raise ValueError(f"Failed to parse --hyper_param '{args.hyper_param}'. "
                                 f"It is neither a valid file path nor a valid JSON string. Error: {e}")

    generate_fn = get_generate_api(
        model_path=args.model_path,
        infer_config=infer_config,
        deltakv_checkpoint_path=args.deltakv_checkpoint_path,
        tokenizer_path=args.tokenizer_path,
        sparse_method=args.sparse_method,
        cuda_device=rank,
        backend=args.backend
    )
    
    # 我们还需要 tokenizer 来进行长度检查和截断
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        max_length = int(config.max_position_embeddings)
    except Exception as exc:
        raise RuntimeError(
            "Failed to resolve max_position_embeddings from the model config: "
            f"{args.model_path}."
        ) from exc

    generation_config = GenerationConfig.from_pretrained(args.model_path, trust_remote_code=True)
    eos_token_ids = generation_config.eos_token_id
    if eos_token_ids is None:
        eos_token_ids = []
    elif isinstance(eos_token_ids, int):
        eos_token_ids = [eos_token_ids]
    else:
        eos_token_ids = list(eos_token_ids)
    if tokenizer.eos_token_id is not None:
        eos_token_ids.append(int(tokenizer.eos_token_id))
    if getattr(tokenizer, "eot_token_id", None) is not None:
        eos_token_ids.append(int(tokenizer.eot_token_id))
    eos_token_ids = list(dict.fromkeys(int(token_id) for token_id in eos_token_ids))

    return generate_fn, tokenizer, max_length, eos_token_ids


def get_pred(rank, data, dataset_info, args, model, tokenizer, model_max_length, eos_token_ids):
    dataset = dataset_info['dataset']
    prompt_format = dataset_info['prompt_format']
    max_gen = args.max_new_tokens_override if args.max_new_tokens_override is not None else dataset_info['max_gen']
    max_length = model_max_length if model_max_length else dataset_info['max_length']
    out_path = dataset_info['out_path']
    out_root = dataset_info['out_root']

    batch_size = args.batch_size
    failures: list[dict[str, Any]] = []
    for i in tqdm(range(0, len(data), batch_size), desc=f'[Rank {rank}] {dataset}'):
        batch_data = data[i:i + batch_size]
        prompts = []
        prepared_records: list[dict[str, Any]] = []
        for json_obj in batch_data:
            selected_idx = int(json_obj.get("_longbench_selected_idx", i + len(prepared_records)))
            prompt_tokens = json_obj.get("_longbench_prompt_tokens")
            try:
                if "answers" not in json_obj or "all_classes" not in json_obj:
                    raise ValueError("LongBench sample must contain answers and all_classes fields.")

                if args.sparse_method == "kvzip" and args.backend == "hf":
                    use_kvzip_template = should_use_chat_template(
                        dataset,
                        args.no_chat_template,
                        args.thinking_mode,
                    )
                    prompt_parts = build_kvzip_prompt_parts(
                        prompt_format,
                        json_obj,
                        use_kvzip_template=use_kvzip_template,
                    )
                    query_ids = tokenizer(
                        prompt_parts["query_text"],
                        truncation=False,
                        return_tensors="pt",
                    ).input_ids[0]
                    max_context_length = max(max_length - len(query_ids), 1)
                    context_ids = tokenizer(
                        prompt_parts["prefill_text"],
                        truncation=False,
                        return_tensors="pt",
                    ).input_ids[0]
                    if prompt_tokens is None:
                        prompt_tokens = int(len(context_ids) + len(query_ids))
                    if len(context_ids) > max_context_length:
                        half = int(max_context_length / 2)
                        if half == 0:
                            prompt_parts["prefill_text"] = tokenizer.decode(
                                context_ids[-max_context_length:],
                                skip_special_tokens=True,
                            )
                        else:
                            prompt_parts["prefill_text"] = (
                                tokenizer.decode(context_ids[:half], skip_special_tokens=True) +
                                tokenizer.decode(context_ids[-half:], skip_special_tokens=True)
                            )
                    prompt = prompt_parts
                else:
                    prompt = prompt_format.format(**json_obj)
                    tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                    if len(tokenized_prompt) > max_length:
                        half = int(max_length / 2)
                        prompt = (
                                tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                                tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                        )
                    prompt = build_chat(tokenizer, prompt, dataset, args.no_chat_template, args.thinking_mode)
                    if prompt_tokens is None:
                        add_special_tokens = True
                        if tokenizer.bos_token is None or prompt.startswith(tokenizer.bos_token):
                            add_special_tokens = False
                        prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=add_special_tokens))
                prompts.append(prompt)
                prepared_records.append(
                    _sample_base_record(
                        dataset=dataset,
                        batch_offset=selected_idx,
                        json_obj=json_obj,
                        prompt_tokens=prompt_tokens,
                    )
                )
            except Exception as exc:
                record = _sample_base_record(
                    dataset=dataset,
                    batch_offset=selected_idx,
                    json_obj=json_obj,
                    prompt_tokens=prompt_tokens,
                )
                record.update(
                    {
                        "status": "invalid_input",
                        "pred": "",
                        "raw_pred": "",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                _write_sample_record(out_root=out_root, task_out_path=out_path, record=record)
                failures.append(record)

        if failures:
            break

        try:
            preds = model(
                prompts,
                max_new_tokens=max_gen,
                num_beams=1,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                eos_token_id=eos_token_ids,
            )
        except Exception as exc:
            for record in prepared_records:
                failed = dict(record)
                failed.update(
                    {
                        "status": "model_failed",
                        "pred": "",
                        "raw_pred": "",
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                _write_sample_record(out_root=out_root, task_out_path=out_path, record=failed)
                failures.append(failed)
            break

        if isinstance(preds, str): preds = [preds]
        if len(preds) != len(prepared_records):
            error = (
                f"Model returned {len(preds)} predictions for "
                f"{len(prepared_records)} prompts in dataset={dataset}."
            )
            for record in prepared_records:
                failed = dict(record)
                failed.update(
                    {
                        "status": "parse_failed",
                        "pred": "",
                        "raw_pred": "",
                        "error": error,
                    }
                )
                _write_sample_record(out_root=out_root, task_out_path=out_path, record=failed)
                failures.append(failed)
            break

        for record, pred in zip(prepared_records, preds):
            raw_pred = pred
            try:
                if not isinstance(raw_pred, str):
                    raise TypeError(f"Model prediction must be str, got {type(raw_pred).__name__}.")
                should_strip_thinking = (
                    args.thinking_mode == "on_strip"
                    and not args.no_chat_template
                    and dataset not in NO_CHAT_TEMPLATE_DATASETS
                )
                parsed_pred = strip_thinking_content(raw_pred) if should_strip_thinking else raw_pred
            except Exception as exc:
                failed = dict(record)
                failed.update(
                    {
                        "status": "parse_failed",
                        "pred": "",
                        "raw_pred": raw_pred if isinstance(raw_pred, str) else repr(raw_pred),
                        "error": repr(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                _write_sample_record(out_root=out_root, task_out_path=out_path, record=failed)
                failures.append(failed)
                continue

            ok = dict(record)
            ok.update(
                {
                    "status": "success",
                    "pred": parsed_pred,
                    "raw_pred": raw_pred,
                }
            )
            _write_sample_record(out_root=out_root, task_out_path=out_path, record=ok)

        if failures:
            break

    if failures:
        first = failures[0]
        raise RuntimeError(
            f"LongBench prediction failed for dataset={dataset}, rank={rank}, "
            f"status={first.get('status')}, source_idx={first.get('source_idx')}: {first.get('error')}"
        )


def worker(rank, world_size, datasets, dataset2prompt, dataset2maxlen, args, out_root, max_length_limit):
    seed_everything(42)
    model, tokenizer, model_max_length, eos_token_ids = load_model_and_tokenizer(rank, args)
    
    for dataset in datasets:
        data_path = get_longbench_data_path(dataset, args.e)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"LongBench dataset file not found for dataset '{dataset}': {data_path}"
            )
        
        data = [json.loads(line) for line in open(data_path, 'r', encoding="utf-8")]
        if args.num_samples: data = data[:args.num_samples]

        if args.min_prompt_tokens is not None:
            from benchmark.sparsevllm_regression.longbench_mini import select_longbench_mini_samples

            selected, selection_meta = select_longbench_mini_samples(
                data=data,
                tokenizer=tokenizer,
                dataset=dataset,
                prompt_format=dataset2prompt[dataset],
                min_prompt_tokens=int(args.min_prompt_tokens),
                samples_per_task=int(args.samples_per_task),
                min_required_samples=int(args.min_required_samples),
                no_chat_template=bool(args.no_chat_template),
                thinking_mode=args.thinking_mode,
            )
            if rank == 0:
                _append_jsonl(os.path.join(out_root, "longbench_mini_selection.jsonl"), selection_meta)
            if selection_meta["status"] == "skipped_by_policy":
                if rank == 0:
                    skipped = {
                        "dataset": dataset,
                        "sample_idx": -1,
                        "source_idx": -1,
                        "prompt_tokens": None,
                        "answers": None,
                        "all_classes": None,
                        "length": None,
                        "status": "skipped_by_policy",
                        "pred": "",
                        "raw_pred": "",
                        "error": (
                            f"Only {selection_meta['selected_rows']} samples reached "
                            f"min_prompt_tokens={selection_meta['min_prompt_tokens']}; "
                            f"min_required_samples={selection_meta['min_required_samples']}."
                        ),
                        "selection": selection_meta,
                    }
                    _write_sample_record(
                        out_root=out_root,
                        task_out_path=os.path.join(out_root, f"{dataset}.jsonl"),
                        record=skipped,
                    )
                continue

            selected_data: list[dict[str, Any]] = []
            for selected_idx, item in enumerate(selected):
                row = dict(item.row)
                row["_longbench_source_idx"] = int(item.source_idx)
                row["_longbench_selected_idx"] = int(selected_idx)
                row["_longbench_prompt_tokens"] = int(item.prompt_tokens)
                selected_data.append(row)
            data = selected_data
        
        data_subset = data[rank::world_size]
        if not data_subset: continue
        
        dataset_info = {
            'dataset': dataset,
            'prompt_format': dataset2prompt[dataset],
            'max_gen': dataset2maxlen[dataset],
            'max_length': max_length_limit,
            'out_path': os.path.join(out_root, f"{dataset}.jsonl"),
            'out_root': out_root,
        }
        
        get_pred(
            rank,
            data_subset,
            dataset_info,
            args,
            model,
            tokenizer,
            model_max_length,
            eos_token_ids,
        )
        torch.cuda.empty_cache()


def launch_single_gpu_workers(args, out_root):
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        gpu_ids = [gpu.strip() for gpu in visible.split(",") if gpu.strip()]
    else:
        gpu_ids = [str(i) for i in range(torch.cuda.device_count())]

    if len(gpu_ids) < args.ws:
        raise ValueError(
            f"Requested ws={args.ws}, but only {len(gpu_ids)} visible GPUs are available: {gpu_ids}"
        )

    script_path = Path(__file__).resolve()
    child_argv = sys.argv[1:]
    procs = []
    for rank in range(args.ws):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[rank]
        cmd = [
            sys.executable,
            "-u",
            str(script_path),
            *child_argv,
            "--worker_rank",
            str(rank),
            "--worker_world_size",
            str(args.ws),
            "--output_root",
            out_root,
        ]
        print(f"[Parent] launch rank={rank} gpu={gpu_ids[rank]} cmd={' '.join(cmd)}", flush=True)
        procs.append(subprocess.Popen(cmd, env=env, cwd=str(script_path.parent.parent.parent)))

    failed_ranks = []
    for rank, proc in enumerate(procs):
        ret = proc.wait()
        if ret != 0:
            failed_ranks.append((rank, ret))
    if failed_ranks:
        raise RuntimeError(
            "LongBench worker failed; aborting evaluation. "
            + ", ".join(f"rank={rank}, exitcode={ret}" for rank, ret in failed_ranks)
        )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="my_model")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--ws", default=1, type=int, help='world size')
    parser.add_argument("--task_start_id", default=0, type=int)
    parser.add_argument("--task", default=None, type=str)

    # DeltaKV related arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--deltakv_checkpoint_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--sparse_method", type=str, default='deltakv')
    parser.add_argument("--backend", type=str, default='hf', choices=['hf', 'sparsevllm'])
    parser.add_argument("--num_samples", type=int, default=None, help="Limit the number of samples to process per task")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--no_chat_template", action='store_true', help="Do not use chat template")
    parser.add_argument("--hyper_param", type=str, default=None, help="Path to a JSON file or a JSON string containing hyper-parameters")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--thinking_mode", type=str, default="off", choices=["off", "on_strip"])
    parser.add_argument("--max_new_tokens_override", type=int, default=None)
    parser.add_argument("--min_prompt_tokens", type=int, default=None)
    parser.add_argument("--samples_per_task", type=int, default=20)
    parser.add_argument("--min_required_samples", type=int, default=5)
    parser.add_argument("--worker_rank", type=int, default=-1)
    parser.add_argument("--worker_world_size", type=int, default=1)
    parser.add_argument("--output_root", type=str, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    
    model_name = args.model
    compressor_name = os.path.basename(args.deltakv_checkpoint_path.rstrip('/')) if args.deltakv_checkpoint_path else "None"
    
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        # en + zh
        # datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]
        # en
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa", "samsum", "passage_count",
                    "passage_retrieval_en", "lcc", "repobench-p"]
    
    datasets = datasets[args.task_start_id:]
    if args.task: datasets = args.task.split(',')

    dataset2prompt = json.load(open("benchmark/long_bench/config/dataset2prompt.json", "r"))
    dataset2maxlen = json.load(open("benchmark/long_bench/config/dataset2maxlen.json", "r"))
    validate_longbench_data_paths(datasets, args.e)
    
    if args.output_root:
        out_root = args.output_root
    else:
        time_tag = datetime.now().strftime("%m%d_%H%M")
        out_root = os.path.join(BASE_PATH, f"benchmark/long_bench/{'pred_e' if args.e else 'pred'}/{model_name}/{compressor_name}_{time_tag}")
    os.makedirs(out_root, exist_ok=True)
    print(f"Results will be saved in: {out_root}")

    if args.worker_rank < 0:
        for dataset in datasets:
            with open(os.path.join(out_root, f"{dataset}.jsonl"), 'w') as f:
                pass
        for artifact in ("raw_outputs.jsonl", "parsed_outputs.jsonl", "sample_results.jsonl", "longbench_mini_selection.jsonl"):
            with open(os.path.join(out_root, artifact), "w", encoding="utf-8") as f:
                pass

    max_length_limit = 120_000 + 1000
    args.max_model_len = max_length_limit

    if args.worker_rank < 0:
        resolved_config = {
            "model": args.model,
            "model_path": args.model_path,
            "tokenizer_path": args.tokenizer_path or args.model_path,
            "backend": args.backend,
            "sparse_method": args.sparse_method,
            "deltakv_checkpoint_path": args.deltakv_checkpoint_path,
            "datasets": datasets,
            "longbench_data_root": DATA_PREFIX_PATH,
            "max_model_len": args.max_model_len,
            "decoding": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens_override": args.max_new_tokens_override,
            },
            "selection": {
                "min_prompt_tokens": args.min_prompt_tokens,
                "samples_per_task": args.samples_per_task,
                "min_required_samples": args.min_required_samples,
            },
            "args": vars(args),
        }
        with open(os.path.join(out_root, "resolved_config.json"), "w", encoding="utf-8") as f:
            json.dump(resolved_config, f, ensure_ascii=False, indent=2)
            f.write("\n")

    if args.worker_rank >= 0:
        worker(args.worker_rank, args.worker_world_size, datasets, dataset2prompt, dataset2maxlen, args, out_root, max_length_limit)
    elif args.ws > 1 and (args.sparse_method == "kvzip" or args.backend == "sparsevllm"):
        launch_single_gpu_workers(args, out_root)
    elif args.ws > 1:
        processes = []
        for rank in range(args.ws):
            p = mp.Process(target=worker, args=(rank, args.ws, datasets, dataset2prompt, dataset2maxlen, args, out_root, max_length_limit))
            p.start()
            processes.append(p)
        failed_ranks = []
        for rank, p in enumerate(processes):
            p.join()
            if p.exitcode != 0:
                failed_ranks.append((rank, p.exitcode))
        if failed_ranks:
            raise RuntimeError(
                "LongBench worker failed; aborting evaluation. "
                + ", ".join(f"rank={rank}, exitcode={exitcode}" for rank, exitcode in failed_ranks)
            )
    else:
        worker(0, 1, datasets, dataset2prompt, dataset2maxlen, args, out_root, max_length_limit)

    if args.worker_rank < 0:
        # 记录评测信息到日志文件
        log_path = os.path.join(out_root, "longbench_eval.log")
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Command: python {' '.join(sys.argv)}\n")
            f.write(f"Output Root: {out_root}\n")
            f.write(f"Args: {json.dumps(vars(args), indent=2)}\n")
            f.write("-" * 80 + "\n")

        # 自动运行评测并记录日志
        print(f"正在对 {out_root} 进行自动评测...")
        eval_cmd = [
            sys.executable,
            "benchmark/long_bench/eval.py",
            "--path", out_root
        ]
        if args.e:
            eval_cmd.append("--e")

        subprocess.run(eval_cmd, check=True)

        # 读取评测结果并写入日志
        result_path = os.path.join(out_root, "result.json")
        if not os.path.exists(result_path):
            raise FileNotFoundError(f"LongBench evaluation did not write result.json: {result_path}")
        with open(result_path, "r", encoding="utf-8") as f:
            scores = json.load(f)

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"Evaluation Results ({'LongBench-E' if args.e else 'LongBench'}):\n")
            f.write(json.dumps(scores, indent=4, ensure_ascii=False))
            f.write("\n" + "="*80 + "\n\n")
        print(f"评测结果已成功写入日志: {log_path}")
