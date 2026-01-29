import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer

# 默认路径
DEFAULT_DATA_DIR = '/root/autodl-fs/datasets/LongBench/data'
DEFAULT_CONFIG_DIR = 'benchmark/long_bench/config'

def build_chat(tokenizer, prompt, dataset, no_chat_template=False):
    if dataset in ["trec", "triviaqa", "samsum", "lsht", "lcc", "repobench-p"]:
        return prompt
    if not no_chat_template and hasattr(tokenizer, 'apply_chat_template') and tokenizer.chat_template is not None:
        msgs = [
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt},
        ]
        # 尝试匹配 pred.py 中的参数
        try:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except Exception:
            prompt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt

def parse_args():
    parser = argparse.ArgumentParser(description="统计 LongBench 各任务的 Token 长度")
    parser.add_argument("--model_path", type=str, required=True, help="用于加载 tokenizer 的模型路径")
    parser.add_argument("--data_dir", type=str, default=DEFAULT_DATA_DIR, help="LongBench 数据集 jsonl 文件所在目录")
    parser.add_argument("--config_dir", type=str, default=DEFAULT_CONFIG_DIR, help="LongBench 配置文件目录")
    parser.add_argument("--e", action="store_true", help="是否统计 LongBench-E")
    parser.add_argument("--no_chat_template", action="store_true", help="不使用对话模板统计长度")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Loading tokenizer from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    
    if args.e:
        datasets = ["qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "gov_report", "multi_news", "trec", "triviaqa", "samsum", "passage_count", "passage_retrieval_en", "lcc", "repobench-p"]
    else:
        datasets = ["narrativeqa", "qasper", "multifieldqa_en", "multifieldqa_zh", "hotpotqa", "2wikimqa", "musique", "dureader", "gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

    dataset2prompt_path = os.path.join(args.config_dir, "dataset2prompt.json")
    if not os.path.exists(dataset2prompt_path):
        # 尝试备用路径
        dataset2prompt_path = "benchmark/long_bench/config/dataset2prompt.json"
        
    dataset2prompt = json.load(open(dataset2prompt_path, "r"))
    
    print(f"{'Dataset':<25} | {'Count':<6} | {'Mean':<8} | {'Min':<8} | {'Max':<8} | {'Median':<8}")
    print("-" * 75)

    all_lengths = []

    for dataset in datasets:
        file_name = f"{dataset}{'_e' if args.e else ''}.jsonl"
        data_path = os.path.join(args.data_dir, file_name)
        
        if not os.path.exists(data_path):
            # 兼容性处理：如果 data_dir 是 LongBench 根目录，则在 data/ 下查找
            alt_path = os.path.join(args.data_dir, "data", file_name)
            if os.path.exists(alt_path):
                data_path = alt_path
            else:
                # print(f"Warning: {data_path} not found, skipping...")
                continue
            
        lengths = []
        with open(data_path, 'r', encoding="utf-8") as f:
            for line in tqdm(f, desc=f"Processing {dataset}", leave=False):
                json_obj = json.loads(line)
                prompt_format = dataset2prompt[dataset]
                prompt = prompt_format.format(**json_obj)
                
                # 应用对话模板
                prompt = build_chat(tokenizer, prompt, dataset, args.no_chat_template)
                
                tokenized_prompt = tokenizer(prompt, truncation=False).input_ids
                lengths.append(len(tokenized_prompt))
        
        if lengths:
            mean_len = np.mean(lengths)
            min_len = np.min(lengths)
            max_len = np.max(lengths)
            median_len = np.median(lengths)
            
            all_lengths.extend(lengths)
            print(f"{dataset:<25} | {len(lengths):<6} | {mean_len:<8.1f} | {min_len:<8} | {max_len:<8} | {median_len:<8.1f}")

    if all_lengths:
        print("-" * 75)
        print(f"{'Overall':<25} | {len(all_lengths):<6} | {np.mean(all_lengths):<8.1f} | {np.min(all_lengths):<8} | {np.max(all_lengths):<8} | {np.median(all_lengths):<8.1f}")

if __name__ == "__main__":
    main()
