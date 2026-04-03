import torch
import random
import json
import os
import math
import subprocess
import glob
import uuid
import fire

from torch.utils.data import IterableDataset, DataLoader
from datasets import IterableDataset as HfIterableDataset
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, set_seed
from tqdm import tqdm

set_seed(218)
random.seed(218)


class ShardedDataset(IterableDataset):
    def __init__(self, data_source, data_cls, total_max_items, data_max_len, tokenizer,
                 skip_factor=1, min_edu_score=0, use_float_edu_score=False, gen_attn_mask=False,
                 min_sample_len=0):
        self.data_source = data_source
        self.data_cls = data_cls
        self.total_max_items = total_max_items
        self.data_max_len = data_max_len
        self.tokenizer = tokenizer
        self.skip_factor = 1.0 / skip_factor  # 保留数据的比例
        self.min_edu_score = min_edu_score
        self.use_float_edu_score = use_float_edu_score
        self.gen_attn_mask = gen_attn_mask
        self.min_sample_len = min_sample_len  # 用于过滤短于此长度的样本

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # 单进程
            world_size = 1
            rank = 0
        else:  # 多进程
            world_size = worker_info.num_workers
            rank = worker_info.id

        # 为每个 worker 计算 max_items
        if self.total_max_items:
            remainder = self.total_max_items % world_size
            per_worker_max_items = self.total_max_items // world_size
            if rank < remainder:
                per_worker_max_items += 1
        else:
            per_worker_max_items = None

        token_buffer = []
        mask_buffer = [] if self.gen_attn_mask else None
        sample_id_counter = 1
        cnt = 0
        eos_token = self.tokenizer.eos_token
        bos_token = self.tokenizer.bos_token if self.tokenizer.bos_token is not None else ""

        for i, sample in enumerate(self.data_source):
            # 分片逻辑：当前样本由该 worker 处理
            if i % world_size != rank:
                continue

            # 按比例随机跳过部分数据
            if self.skip_factor < 1.0 - 1e-6 and random.random() > self.skip_factor:
                continue

            # 处理样本，提取文本和分数
            _edu_score = 999
            if isinstance(sample, dict):
                text = bos_token + sample["text"] + eos_token
                if "metadata" in sample:
                    if self.use_float_edu_score:
                        _edu_score = sample["metadata"].get("score", 5)
                    else:
                        _edu_score = sample["metadata"].get("int_score", 5)
            else:
                text = bos_token + sample.text + eos_token
                if hasattr(sample, "metadata"):
                    if self.use_float_edu_score:
                        _edu_score = sample.metadata.get("score", 5)
                    else:
                        _edu_score = sample.metadata.get("int_score", 5)

            if _edu_score < self.min_edu_score:
                continue

            tokens = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0]

            # 过滤掉分词后长度过短的样本
            if len(tokens) < self.min_sample_len:
                continue

            token_buffer.append(tokens)
            if self.gen_attn_mask:
                # 为每个样本生成唯一的 ID 作为 mask
                mask_buffer.append(torch.full_like(tokens, sample_id_counter))
                sample_id_counter += 1

            # 为减少 `torch.cat` 的调用次数，当 buffer 中有多个 tensor 时再合并
            if len(token_buffer) > 1:
                token_buffer = [torch.cat(token_buffer)]
                if self.gen_attn_mask:
                    mask_buffer = [torch.cat(mask_buffer)]

            # 从缓冲区中生成并 yield 固定长度的 chunk
            while token_buffer and len(token_buffer[0]) >= self.data_max_len:
                # 从缓冲区头部取出 data_max_len 长度的 tokens
                chunk = token_buffer[0][:self.data_max_len]

                output_data = {
                    "input_ids": chunk.tolist(),
                    "data_cls": self.data_cls
                }

                if self.gen_attn_mask:
                    # 同样取出对应长度的 mask
                    mask_chunk = mask_buffer[0][:self.data_max_len]
                    output_data["attention_mask"] = (mask_chunk - mask_chunk.min() + 1).tolist()
                    # 更新 mask 缓冲区，保留剩余部分
                    mask_buffer[0] = mask_buffer[0][self.data_max_len:]

                yield output_data

                cnt += 1
                if per_worker_max_items and cnt >= per_worker_max_items:
                    return

                # 更新 token 缓冲区，保留剩余部分
                token_buffer[0] = token_buffer[0][self.data_max_len:]


class TransformIterableDataset(IterableDataset):
    def __init__(self, data_source, transform):
        self.data_source = data_source
        self.transform = transform

    def __iter__(self):
        for sample in self.data_source:
            transformed = self.transform(sample)
            if transformed is None:
                continue
            yield transformed


def get_dataloader(dataset, num_workers):
    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=num_workers,
        collate_fn=lambda x: x
    )


def collect_loader_items(loader, target_count, desc):
    data_lst = []
    for item in tqdm(loader, total=target_count, desc=desc):
        data_lst.append(item)
        if len(data_lst) >= target_count:
            break
    if len(data_lst) < target_count:
        print(f"[WARN] {desc}: collected {len(data_lst)} < target {target_count}")
    return data_lst


def load_local_dataset(dataset_path, split="train", streaming=False):
    parquet_glob = os.path.join(dataset_path, "data", "*.parquet")
    parquet_files = sorted(glob.glob(parquet_glob))
    if parquet_files:
        return load_dataset("parquet", data_files=parquet_files, split=split, streaming=streaming)
    return load_dataset(dataset_path, split=split, streaming=streaming)


def render_chat_messages(tokenizer, messages):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": False,
    }
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def build_uuid_kv_dialog(tokenizer, min_target_tokens, max_target_tokens):
    min_target_tokens = max(1024, int(min_target_tokens))
    max_target_tokens = max(min_target_tokens, int(max_target_tokens))

    prompt_styles = [
        "Extract the value corresponding to the specified key in the JSON object below.",
        "Read the JSON object below and return the value for the requested key.",
        "Given the JSON object below, identify the exact value for the specified key.",
    ]
    followup_styles = [
        'Key: "{key}"\nThe value associated with the specified key is:',
        'Please retrieve the value for key "{key}".\nAnswer with the exact UUID:',
        'Find the exact UUID value for the following key.\nKey: "{key}"\nValue:',
    ]

    text = None
    for _ in range(6):
        target_tokens = random.randint(min_target_tokens, max_target_tokens)
        approx_entries = max(48, int(target_tokens / 68))
        approx_entries = max(48, int(approx_entries * random.uniform(0.9, 1.1)))

        kv_pairs = {
            str(uuid.uuid4()): str(uuid.uuid4())
            for _ in range(approx_entries)
        }
        keys = list(kv_pairs.keys())
        random.shuffle(keys)
        num_turns = min(len(keys), random.randint(4, 6))

        first_prompt = (
            f"{random.choice(prompt_styles)}\n\n"
            f"JSON data:\n{json.dumps(kv_pairs, ensure_ascii=True)}\n\n"
            f'{random.choice(followup_styles).format(key=keys[0])}'
        )
        messages = [
            {"role": "user", "content": first_prompt},
            {"role": "assistant", "content": kv_pairs[keys[0]]},
        ]

        for key in keys[1:num_turns]:
            messages.append(
                {"role": "user", "content": random.choice(followup_styles).format(key=key)}
            )
            messages.append({"role": "assistant", "content": kv_pairs[key]})

        text = render_chat_messages(tokenizer, messages)
        actual_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
        if actual_tokens >= min_target_tokens:
            return text

    return text


def build_reasoning_transform(tokenizer):
    def _transform(sample):
        messages = sample.get("messages")
        if not messages:
            return None
        clean_messages = []
        for message in messages:
            role = message.get("role")
            content = message.get("content")
            if role not in {"system", "user", "assistant"} or not content:
                continue
            clean_messages.append({"role": role, "content": content})

        if not clean_messages:
            return None

        try:
            text = render_chat_messages(tokenizer, clean_messages)
        except Exception:
            text = ""
            for message in clean_messages:
                text += f"{message['role']}: {message['content']}\n"
        return {"text": text}

    return _transform

def shuffle_jsonl(input_path, output_path):
    """
    使用Linux `shuf` 命令对输入的JSONL文件进行随机化，并保存到输出路径
    :param input_path: 输入的 JSONL 文件路径
    :param output_path: 输出的 JSONL 文件路径
    """
    try:
        # 使用 subprocess 调用 `shuf` 命令对文件进行随机化
        with open(output_path, 'w') as outfile:
            subprocess.run(['shuf', input_path], stdout=outfile, check=True)
        print(f"随机化后的文件已保存到 {output_path}")
    except Exception as e:
        print(f"处理文件时出错：{e}")

# =================== 各 Version 的数据处理函数 ===================

def process_v2_4(model_cls, tkn, data_max_len, num_workers, num_samples=100_000):
    from datatrove.pipeline.readers import ParquetReader
    data_fineweb_edu = ParquetReader("/root/autodl-fs/datasets/fineweb-edu/sample/10BT")()
    # data_hermes = get_any_dataset("../datasets/OpenHermes-2.5", tkn)["train"]
    output_path = f"/root/autodl-fs/datasets/deltakv_{model_cls}_train_num{num_samples}"
    if data_max_len != 1024:
        output_path += f"_seqlen{data_max_len}"
    os.makedirs(output_path, exist_ok=True)

    # 创建分片数据集并处理
    datasets = [
        get_dataloader(ShardedDataset(data_fineweb_edu, "general", None, data_max_len, tkn), num_workers), # 10B tokens
    ]
    
    cnt = 0
    data_lst = []
    for _data in datasets:
        for item in tqdm(_data):
            data_lst.append(item)
            cnt += 1
            if cnt >= num_samples:
                break
    
    data = Dataset.from_list(data_lst)
    data.shuffle(seed=218).save_to_disk(output_path)
    
    print(f"有{cnt * data_max_len / 1e6}B tokens")
    print(f"有{cnt}条数据")

def process_v3_0(model_cls, tkn, data_max_len, num_workers, num_samples=100_000):
    from datatrove.pipeline.readers import ParquetReader
    # Fineweb-edu
    data_fineweb_edu = ParquetReader("/root/autodl-fs/datasets/fineweb-edu/sample/10BT")()
    # CCI3-HQ (CHI)
    data_chi = load_dataset("/autodl-fs/data/datasets/CCI3-HQ", split="train", streaming=True)
    data_chi = data_chi.shuffle(seed=218, buffer_size=10000)
    
    output_path = f"/root/autodl-fs/datasets/deltakv_{model_cls}_train_v3.0_num{num_samples}"
    if data_max_len != 1024:
        output_path += f"_seqlen{data_max_len}"
    os.makedirs(output_path, exist_ok=True)

    # 1:4 比例: CHI:fineweb = 1:4, 总共 5 份
    num_chi = num_samples // 5
    num_fineweb = num_samples - num_chi
    
    data_lst = []
    
    # 采集 Fineweb 数据, skip_factor=10 表示以 1/10 的概率采样，从而让采样点分布在更广的数据范围
    loader_fineweb = get_dataloader(ShardedDataset(data_fineweb_edu, "general", num_fineweb, data_max_len, tkn, skip_factor=10), num_workers)
    print(f"Sampling {num_fineweb} samples from Fineweb-Edu...")
    for item in tqdm(loader_fineweb, total=num_fineweb):
        data_lst.append(item)
        if len(data_lst) >= num_fineweb:
            break
            
    # 采集 CHI 数据
    loader_chi = get_dataloader(ShardedDataset(data_chi, "chinese", num_chi, data_max_len, tkn), num_workers)
    print(f"Sampling {num_chi} samples from CCI3-HQ...")
    for item in tqdm(loader_chi, total=num_chi):
        data_lst.append(item)
        if len(data_lst) >= num_samples:
            break

    data = Dataset.from_list(data_lst)
    data.shuffle(seed=218).save_to_disk(output_path)

def process_vr1_0(
    model_cls,
    tkn,
    data_max_len,
    num_workers,
    num_samples=100_000,
    dataset_path=None,
    output_root=None,
):
    dataset_path = dataset_path or os.path.expanduser(
        "~/datasets/AM-DeepSeek-R1-Distilled-1.4M-am_0.9M_sample100k"
    )
    output_root = output_root or os.path.expanduser("~/datasets")

    parquet_glob = os.path.join(dataset_path, "data", "*.parquet")
    if os.path.isdir(dataset_path) and os.path.isdir(os.path.join(dataset_path, "data")):
        raw_dataset = load_dataset("parquet", data_files=parquet_glob, split="train")
    else:
        raw_dataset = load_dataset(dataset_path, split="train")
    
    def data_gen():
        for sample in raw_dataset:
            messages = sample["messages"]
            # 仅保留 role 和 content
            clean_messages = [{"role": m["role"], "content": m["content"]} for m in messages]
            try:
                # 使用 tokenizer 的 chat template 将对话转换为文本
                text = tkn.apply_chat_template(clean_messages, tokenize=False, add_generation_prompt=False)
            except Exception:
                # 如果没有 template，则使用简单的拼接方式
                text = ""
                for m in clean_messages:
                    text += f"{m['role']}: {m['content']}\n"
            yield {"text": text}

    output_path = os.path.join(output_root, f"deltakv_{model_cls}_train_vr1.0_num{num_samples}")
    if data_max_len != 1024:
        output_path += f"_seqlen{data_max_len}"
    os.makedirs(output_path, exist_ok=True)

    # 使用 ShardedDataset 进行分片处理
    loader = get_dataloader(ShardedDataset(data_gen(), "r1_distill", num_samples, data_max_len, tkn), num_workers)
    
    data_lst = []
    print(f"Sampling {num_samples} samples from DeepSeek-R1-Distilled (vr1.0)...")
    for item in tqdm(loader, total=num_samples):
        data_lst.append(item)
        if len(data_lst) >= num_samples:
            break
            
    data = Dataset.from_list(data_lst)
    data.shuffle(seed=218).save_to_disk(output_path)
    print(f"数据处理完成，保存至 {output_path}")


def process_vi1_0(
    model_cls,
    tkn,
    data_max_len,
    num_workers,
    num_samples=100_000,
    fineweb_path=None,
    reasoning_path=None,
    output_root=None,
    reasoning_ratio=0.08,
    uuid_kv_ratio=0.02,
    fineweb_skip_factor=1,
):
    fineweb_path = fineweb_path or os.path.expanduser("~/datasets/fineweb-edu-1B")
    reasoning_path = reasoning_path or os.path.expanduser(
        "~/datasets/AM-DeepSeek-R1-Distilled-1.4M-am_0.9M_sample100k"
    )
    output_root = output_root or os.path.expanduser("~/datasets")

    if reasoning_ratio < 0 or uuid_kv_ratio < 0 or reasoning_ratio + uuid_kv_ratio >= 1:
        raise ValueError("reasoning_ratio 和 uuid_kv_ratio 必须非负，且和小于 1")

    num_reasoning = int(round(num_samples * reasoning_ratio))
    num_uuid_kv = int(round(num_samples * uuid_kv_ratio))
    num_fineweb = num_samples - num_reasoning - num_uuid_kv

    if num_fineweb <= 0:
        raise ValueError("FineWeb 样本数必须大于 0")

    ratio_tag = (
        f"_fw{int(round(num_fineweb / num_samples * 100)):02d}"
        f"_r{int(round(num_reasoning / num_samples * 100)):02d}"
        f"_uuid{int(round(num_uuid_kv / num_samples * 100)):02d}"
    )
    output_path = os.path.join(
        output_root,
        f"deltakv_{model_cls}_train_vi1.0{ratio_tag}_num{num_samples}"
    )
    if data_max_len != 1024:
        output_path += f"_seqlen{data_max_len}"
    os.makedirs(output_path, exist_ok=True)

    print(
        "[vi1.0] mixture:",
        {
            "fineweb": num_fineweb,
            "reasoning": num_reasoning,
            "synthetic_uuid_kv": num_uuid_kv,
            "fineweb_path": fineweb_path,
            "reasoning_path": reasoning_path,
            "output_path": output_path,
        },
    )

    data_lst = []

    data_fineweb = load_local_dataset(fineweb_path, split="train", streaming=True)
    if isinstance(data_fineweb, HfIterableDataset):
        data_fineweb = data_fineweb.shuffle(seed=218, buffer_size=10_000)

    loader_fineweb = get_dataloader(
        ShardedDataset(
            data_fineweb,
            "general",
            num_fineweb,
            data_max_len,
            tkn,
            skip_factor=fineweb_skip_factor,
            min_sample_len=max(64, min(128, data_max_len // 128)),
        ),
        num_workers,
    )
    data_lst.extend(collect_loader_items(loader_fineweb, num_fineweb, "Sampling FineWeb"))

    if num_reasoning > 0:
        raw_reasoning = load_local_dataset(reasoning_path, split="train", streaming=True)
        if isinstance(raw_reasoning, HfIterableDataset):
            raw_reasoning = raw_reasoning.shuffle(seed=218, buffer_size=10_000)
        reasoning_dataset = TransformIterableDataset(
            raw_reasoning,
            build_reasoning_transform(tkn),
        )
        loader_reasoning = get_dataloader(
            ShardedDataset(
                reasoning_dataset,
                "reasoning_chat",
                num_reasoning,
                data_max_len,
                tkn,
                min_sample_len=max(128, min(512, data_max_len // 16)),
            ),
            num_workers,
        )
        data_lst.extend(
            collect_loader_items(loader_reasoning, num_reasoning, "Sampling Reasoning Chats")
        )

    if num_uuid_kv > 0:
        synthetic_uuid_data = []
        min_target_tokens = max(2048, int(data_max_len * 0.45))
        max_target_tokens = max(min_target_tokens, int(data_max_len * 0.85))
        for _ in tqdm(range(num_uuid_kv), desc="Generating Synthetic UUID-KV"):
            synthetic_uuid_data.append(
                {
                    "text": build_uuid_kv_dialog(
                        tkn,
                        min_target_tokens=min_target_tokens,
                        max_target_tokens=max_target_tokens,
                    )
                }
            )
        loader_uuid = get_dataloader(
            ShardedDataset(
                synthetic_uuid_data,
                "synthetic_uuid_kv",
                num_uuid_kv,
                data_max_len,
                tkn,
                min_sample_len=max(1024, min(4096, data_max_len // 4)),
            ),
            num_workers,
        )
        data_lst.extend(
            collect_loader_items(loader_uuid, num_uuid_kv, "Packing Synthetic UUID-KV")
        )

    data = Dataset.from_list(data_lst)
    data.shuffle(seed=218).save_to_disk(output_path)
    print(f"[vi1.0] 数据处理完成，保存至 {output_path}")

# =================== 主函数 ===================

def main(
    version,
    tkn_path="../mock/models/Meta-Llama-3-8B-Instruct",
    num_workers=2,
    data_max_len=1024,
    num_samples=100_000,
    dataset_path=None,
    output_root=None,
    fineweb_path=None,
    reasoning_path=None,
    reasoning_ratio=0.08,
    uuid_kv_ratio=0.02,
    fineweb_skip_factor=1,
):
    tkn = AutoTokenizer.from_pretrained(tkn_path, use_fast=True)
    print("bos", tkn.bos_token)
    print("eos", tkn.eos_token)
    model_cls = None
    if "llama-3" in tkn_path.lower():
        model_cls = "llama3"
    elif "llama-2" in tkn_path.lower():
        model_cls = "llama2"
    elif "qwen3" in tkn_path.lower():
        model_cls = "qwen3"
    elif "qwen" in tkn_path.lower():
        model_cls = "qwen"
    elif "phi" in tkn_path.lower():
        model_cls = "phi"
    else:
        raise ValueError

    # 根据 version 调用不同处理函数
    if version == "v2.4":
        process_v2_4(model_cls, tkn, data_max_len, num_workers, num_samples)
    elif version == "v3.0":
        process_v3_0(model_cls, tkn, data_max_len, num_workers, num_samples)
    elif version == "vr1.0":
        process_vr1_0(
            model_cls,
            tkn,
            data_max_len,
            num_workers,
            num_samples,
            dataset_path=dataset_path,
            output_root=output_root,
        )
    elif version == "vi1.0":
        process_vi1_0(
            model_cls,
            tkn,
            data_max_len,
            num_workers,
            num_samples,
            fineweb_path=fineweb_path,
            reasoning_path=reasoning_path,
            output_root=output_root,
            reasoning_ratio=reasoning_ratio,
            uuid_kv_ratio=uuid_kv_ratio,
            fineweb_skip_factor=fineweb_skip_factor,
        )
    else:
        raise ValueError(f"Unsupported version: {version}")

if __name__ == "__main__":
    fire.Fire(main)
