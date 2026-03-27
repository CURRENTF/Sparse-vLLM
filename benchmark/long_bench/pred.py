import os
import json
import sys
import subprocess
import re
from typing import Union

from tqdm import tqdm
import numpy as np
import random
import argparse
import torch.multiprocessing as mp
import torch
from transformers import AutoTokenizer
import torch.distributed as dist
from deltakv.get_chat_api import get_generate_api
from datetime import datetime

BASE_PATH = os.getenv("DELTAKV_OUTPUT_DIR", "/root/autodl-fs/deltakv_outputs")
DATA_PREFIX_PATH = os.getenv(
    "DELTAKV_LONGBENCH_DATA_DIR",
    os.getenv("DELTAKV_DATA_DIR", "/root/autodl-fs/datasets/LongBench/"),
)
NO_CHAT_TEMPLATE_DATASETS = {"trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"}


def get_longbench_data_path(dataset, use_longbench_e):
    suffix = "_e" if use_longbench_e else ""
    return os.path.join(DATA_PREFIX_PATH, "data", f"{dataset}{suffix}.jsonl")


def validate_longbench_data_paths(datasets, use_longbench_e):
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


def build_chat(tokenizer, prompt, dataset, no_chat_template=False, thinking_mode="off"):
    if dataset in NO_CHAT_TEMPLATE_DATASETS:
        return prompt
    if not no_chat_template and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
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
    # When the prompt already ends with an open `<think>`, the generated text is
    # often just the continuation of that block and only emits the closing tag.
    if "</think>" in text:
        text = text.split("</think>", 1)[1]
    text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    text = text.lstrip()

    # Some generations never emit `</think>` and instead end with a natural-
    # language answer cue. Recover the short answer conservatively.
    if "\n" in text and len(text) > 200:
        for pattern in [
            r"(?is)i['’]ll say\s+[\"“]?([^\"”\n\.\?!]+)",
            r"(?is)i['’]ll go with\s+[\"“]?([^\"”\n\.\?!]+)",
            r"(?is)answer would be\s+[\"“]?([^\"”\n\.\?!]+)",
            r"(?is)so the answer is\s+[\"“]?([^\"”\n\.\?!]+)",
            r"(?is)i think the answer is\s+[\"“]?([^\"”\n\.\?!]+)",
            r"(?is)the answer is\s+[\"“]?([^\"”\n\.\?!]+)",
        ]:
            matches = re.findall(pattern, text)
            for match in reversed(matches):
                candidate = match.strip().strip('\'"“”')
                lower = candidate.lower()
                if len(candidate) > 80:
                    continue
                if any(
                    phrase in lower
                    for phrase in ["not explicitly", "not provided", "not enough information"]
                ):
                    continue
                return candidate.rstrip(" .")

    return text


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
        compressor_path=args.compressor_path,
        tokenizer_path=args.tokenizer_path,
        model_cls=args.model_cls,
        cuda_device=rank,
        backend=args.backend
    )
    
    # 我们还需要 tokenizer 来进行长度检查和截断
    tokenizer_path = args.tokenizer_path if args.tokenizer_path else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    
    # 尝试从模型配置中获取 max_position_embeddings
    try:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        max_length = getattr(config, "max_position_embeddings", 32000)
    except:
        max_length = 32000

    return generate_fn, tokenizer, max_length


def get_pred(rank, data, dataset_info, args, model, tokenizer, model_max_length):
    dataset = dataset_info['dataset']
    prompt_format = dataset_info['prompt_format']
    max_gen = args.max_new_tokens_override if args.max_new_tokens_override is not None else dataset_info['max_gen']
    max_length = model_max_length if model_max_length else dataset_info['max_length']
    out_path = dataset_info['out_path']

    batch_size = args.batch_size
    for i in tqdm(range(0, len(data), batch_size), desc=f'[Rank {rank}] {dataset}'):
        batch_data = data[i:i + batch_size]
        prompts = []
        for json_obj in batch_data:
            if args.model_cls == "kvzip" and args.backend == "hf":
                use_kvzip_template = not args.no_chat_template and dataset not in NO_CHAT_TEMPLATE_DATASETS
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
                prompts.append(prompt_parts)
            else:
                prompt = prompt_format.format(**json_obj)
                tokenized_prompt = tokenizer(prompt, truncation=False, return_tensors="pt").input_ids[0]
                if len(tokenized_prompt) > max_length:
                    half = int(max_length / 2)
                    prompt = (
                            tokenizer.decode(tokenized_prompt[:half], skip_special_tokens=True) +
                            tokenizer.decode(tokenized_prompt[-half:], skip_special_tokens=True)
                    )
                prompts.append(build_chat(tokenizer, prompt, dataset, args.no_chat_template, args.thinking_mode))

        eos_token_id = [tokenizer.eos_token_id]
        if hasattr(tokenizer, 'eot_token_id'):
            eos_token_id.append(tokenizer.eot_token_id)

        preds = model(
            prompts,
            max_new_tokens=max_gen,
            num_beams=1,
            do_sample=args.temperature > 0,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            eos_token_id=eos_token_id,
        )

        if isinstance(preds, str): preds = [preds]

        for json_obj, pred in zip(batch_data, preds):
            if args.thinking_mode == "on_strip":
                pred = strip_thinking_content(pred)
            with open(out_path, "a", encoding="utf-8") as f:
                json.dump({
                    "pred": pred, "answers": json_obj["answers"],
                    "all_classes": json_obj["all_classes"],
                    "length": json_obj["length"]
                }, f, ensure_ascii=False)
                f.write('\n')


def worker(rank, world_size, datasets, dataset2prompt, dataset2maxlen, args, out_root, max_length_limit):
    seed_everything(42)
    model, tokenizer, model_max_length = load_model_and_tokenizer(rank, args)
    
    for dataset in datasets:
        data_path = get_longbench_data_path(dataset, args.e)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(
                f"LongBench dataset file not found for dataset '{dataset}': {data_path}"
            )
        
        data = [json.loads(line) for line in open(data_path, 'r', encoding="utf-8")]
        if args.num_samples: data = data[:args.num_samples]
        
        data_subset = data[rank::world_size]
        if not data_subset: continue
        
        dataset_info = {
            'dataset': dataset,
            'prompt_format': dataset2prompt[dataset],
            'max_gen': dataset2maxlen[dataset],
            'max_length': max_length_limit,
            'out_path': os.path.join(out_root, f"{dataset}.jsonl")
        }
        
        get_pred(rank, data_subset, dataset_info, args, model, tokenizer, model_max_length)
        torch.cuda.empty_cache()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default="my_model")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument("--ws", default=1, type=int, help='world size')
    parser.add_argument("--task_start_id", default=0, type=int)
    parser.add_argument("--task", default=None, type=str)

    # DeltaKV related arguments
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--compressor_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, default=None)
    parser.add_argument("--model_cls", type=str, default='deltakv')
    parser.add_argument("--backend", type=str, default='deltakv', choices=['hf', 'sparsevllm'])
    parser.add_argument("--num_samples", type=int, default=None, help="Limit the number of samples to process per task")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--no_chat_template", action='store_true', help="Do not use chat template")
    parser.add_argument("--hyper_param", type=str, default=None, help="Path to a JSON file or a JSON string containing hyper-parameters")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_p", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=20)
    parser.add_argument("--thinking_mode", type=str, default="off", choices=["off", "on_strip"])
    parser.add_argument("--max_new_tokens_override", type=int, default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    mp.set_start_method('spawn', force=True)
    
    model_name = args.model
    compressor_name = os.path.basename(args.compressor_path.rstrip('/')) if args.compressor_path else "None"
    
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
    
    time_tag = datetime.now().strftime("%m%d_%H%M")
    out_root = os.path.join(BASE_PATH, f"benchmark/long_bench/{'pred_e' if args.e else 'pred'}/{model_name}/{compressor_name}_{time_tag}")
    os.makedirs(out_root, exist_ok=True)
    print(f"Results will be saved in: {out_root}")

    for dataset in datasets:
        with open(os.path.join(out_root, f"{dataset}.jsonl"), 'w') as f: pass

    max_length_limit = 120_000 + 1000
    args.max_model_len = max_length_limit

    if args.ws > 1:
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

    # 记录评测信息到日志文件
    log_path = os.path.join(BASE_PATH, "longbench_eval.log")
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

    try:
        subprocess.run(eval_cmd, check=True)

        # 读取评测结果并写入日志
        result_path = os.path.join(out_root, "result.json")
        if os.path.exists(result_path):
            with open(result_path, "r", encoding="utf-8") as f:
                scores = json.load(f)

            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"Evaluation Results ({'LongBench-E' if args.e else 'LongBench'}):\n")
                f.write(json.dumps(scores, indent=4, ensure_ascii=False))
                f.write("\n" + "="*80 + "\n\n")
            print(f"评测结果已成功写入日志: {log_path}")
        else:
            print(f"未找到评测结果文件: {result_path}")
    except subprocess.CalledProcessError as e:
        print(f"评测脚本执行失败: {e}")
