import json
import os
import time
from collections import defaultdict
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from fire import Fire
import jsonlines
from tqdm import tqdm
from transformers import AutoTokenizer

from deltakv.get_chat_api import get_generate_api
from benchmark.niah.gen_niah import generate_text

BASE_PATH = '/root/autodl-fs/deltakv_outputs'


def _load_or_generate_data(
    tokenizer,
    online_test: bool,
    data_path: str,
    context_lengths: list[int],
    num_samples_per_setting: int = 5,
):
    """
    加载或在线生成测试数据。

    Args:
        online_test (bool): 如果为 True，则在线生成数据；否则从 data_path 加载。
        data_path (str): 数据文件的路径 (jsonl格式)。
        context_lengths (list[int]): 在线生成时要测试的上下文长度（单位：千tokens）。
        num_samples_per_setting (int): 每个设置（长度和深度）要生成的样本数量。

    Returns:
        list: 加载或生成的数据列表。
    """
    if not online_test:
        with jsonlines.open(data_path) as reader:
            data = list(reader)
        # 逆序处理，通常长文本在后，优先测试
        return list(reversed(data))
    else:
        data = []
        # 定义在文本中插入“大海捞针”任务的位置比例
        depth_percents = [0, 0.11, 0.22, 0.33, 0.44, 0.56, 0.67, 0.78, 0.89, 1.0]
        for seq_len_k in context_lengths:
            seq_len = seq_len_k * 1000
            for ratio in depth_percents:
                for _ in range(num_samples_per_setting):
                    task_text, secret_key, total_tokens, ratio = generate_text(seq_len, ratio, tokenizer)
                    data.append(
                        {
                            "task": task_text,
                            "answer": secret_key,
                            "total_tokens": total_tokens,
                            "ratio": ratio,
                        }
                    )
        return data


def test(
    model_path: str,
    output_path: str = 'results',
    tokenizer_path: str = None,
    data_path: str = "benchmark/needle_in_haystack_tasks.jsonl",
    max_context_k_tokens: int = 1000_000_000,  # 默认不跳过
    online_test: bool = True,
    context_lengths: str = "16,32,64",
    max_new_tokens: int = 20,
    min_new_tokens: int = 1,
    model_cls: str = 'deltakv',
    compressor_path: str = None,
    use_cache: bool = True,
    cuda_device: int = 0,
    backend: str = 'hf',

    # kv compress infer config
    num_sink_tokens: int = 64,
    num_recent_tokens: int = 512,
    full_attn_layers: str = '0,1,2,3,8,16,22',
    num_top_tokens: int = 512,
    use_compression: bool = True,
    use_cluster: bool = False,
    cluster_ratio: float = 0.1,
    vllm_sparse_method: str = "",
    chunk_prefill_size: int = 32768,
    gpu_memory_utilization: float = 0.8,
    max_model_len: int = 128000,
    pyramid_last_layer_ratio: float = None,
    enforce_eager: bool = True,

    # Palu related arguments
    lt_bits: int = 16,
    lt_group_size: int = 0,
    lt_sym: bool = True,
    lt_clip_ratio: float = 1.0,
    lt_hadamard: bool = False,
):
    """
    在“大海捞针”任务上评测模型的性能。

    该函数通过将一个“秘密”信息（针）放置在一段长文本（干草堆）中，
    然后要求模型找出这个秘密信息，来测试模型在长上下文处理上的能力。

    评测可以从现有数据集加载，也可以在线实时生成。
    """
    tail_token_size = num_recent_tokens

    infer_config = {
        'num_sink_tokens': num_sink_tokens,
        'num_recent_tokens': num_recent_tokens,
        'tail_token_size': tail_token_size,
        'full_attn_layers': full_attn_layers,
        'num_top_tokens': num_top_tokens,
        'use_compression': use_compression,
        'use_cluster': use_cluster,
        'cluster_ratio': cluster_ratio,
        'vllm_sparse_method': vllm_sparse_method,
        'chunk_prefill_size': chunk_prefill_size,
        'gpu_memory_utilization': gpu_memory_utilization,
        'max_model_len': max_model_len,
        'pyramid_last_layer_ratio': pyramid_last_layer_ratio,
        'enforce_eager': enforce_eager,
        'lt_bits': lt_bits,
        'lt_group_size': lt_group_size,
        'lt_sym': lt_sym,
        'lt_clip_ratio': lt_clip_ratio,
        'lt_hadamard': lt_hadamard,
    }
    chat = get_generate_api(
        model_path, 
        infer_config, 
        compressor_path, 
        tokenizer_path, 
        model_cls=model_cls, 
        use_cache=use_cache,
        cuda_device=cuda_device,
        backend=backend
    )

    if compressor_path is not None:
        compressor_name = os.path.basename(compressor_path.rstrip('/'))
    else:
        compressor_name = "None"
    
    time_tag = datetime.now().strftime("%m%d_%H%M")
    output_dir = os.path.join(BASE_PATH, output_path, f"{compressor_name}_{time_tag}")
    os.makedirs(output_dir, exist_ok=True)
    json_save_path = os.path.join(output_dir, "accuracy_rates.json")
    heatmap_save_path = os.path.join(output_dir, "accuracy_heatmap.pdf")
    print(f"Results will be saved in: {output_dir}")

    # 如果min_new_tokens > 1，则强制生成固定长度的token
    if min_new_tokens > 1:
        max_new_tokens = min_new_tokens

    # 解析在线测试的上下文长度
    if isinstance(context_lengths, int):
        context_lengths_list = [context_lengths]
    else:
        context_lengths_list = [int(x) for x in context_lengths.split(",")]

    # 加载或生成数据
    if tokenizer_path is None:
        tokenizer_path = model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    data = _load_or_generate_data(
        tokenizer,
        online_test=online_test,
        data_path=data_path,
        context_lengths=context_lengths_list,
    )

    # 初始化准确率统计
    accuracies = defaultdict(int)
    total_counts = defaultdict(int)
    correct_predictions = 0
    total_iterations = 0

    for item in tqdm(data, desc="Testing..."):
        # 将 total_tokens 转换为千为单位，方便分组
        context_k = item["total_tokens"] // 1000
        if context_k > max_context_k_tokens:
            continue

        total_iterations += 1
        start_time = time.time()

        # 模型生成回复
        response = chat(item["task"], max_new_tokens=max_new_tokens)

        # 准备用于统计的key
        depth_ratio = round(item["ratio"], 3)
        accuracy_key = f"{context_k}-{depth_ratio}"
        total_counts[accuracy_key] += 1

        print(f"Context(k)-Depth: {accuracy_key}, Response: {response}")

        # 检查答案是否正确
        if item["answer"] in response:
            correct_predictions += 1
            accuracies[accuracy_key] += 1
            elapsed_time = round(time.time() - start_time, 3)
            overall_accuracy = round(correct_predictions / total_iterations, 3)
            print(
                f"✅ Correct! | Key: {accuracy_key} | Time: {elapsed_time}s | Overall Acc: {overall_accuracy}"
            )
        else:
            print(f"❌ Incorrect. | Key: {accuracy_key}")

    accuracy_rates = {
        key: round(accuracies[key] / total_counts[key], 3)
        for key in total_counts
    }

    print("\n--- Final Accuracies ---")
    print(json.dumps(accuracy_rates, indent=4))

    # 保存结果到指定的JSON文件
    with open(json_save_path, "w") as f:
        json.dump(accuracy_rates, f, indent=4)
    print(f"\nResults saved to {json_save_path}")

    # --- Generate and Save Heatmap ---
    if not accuracy_rates:
        print("No accuracy data to generate heatmap.")
        if __name__ == "__main__":
            return
        else:
            return

    parsed_data = []
    context_lengths_seen = set()
    depth_ratios_seen = set()

    for key, acc in accuracy_rates.items():
        try:
            context_k, depth_ratio_str = key.split('-')
            context_k = int(context_k)
            depth_ratio = float(depth_ratio_str)
            parsed_data.append({
                'Context Length (K)': context_k,
                'Depth Ratio': depth_ratio,
                'Accuracy': acc
            })
            context_lengths_seen.add(context_k)
            depth_ratios_seen.add(depth_ratio)
        except ValueError:
            print(f"Warning: Could not parse key '{key}'. Skipping for heatmap.")
            continue

    if not parsed_data:
        print("Could not parse any accuracy data for heatmap.")
        if __name__ == "__main__":
            return
        else:
            return

    df = pd.DataFrame(parsed_data)

    try:
        # Pivot the dataframe to create a matrix for the heatmap
        heatmap_data = df.pivot_table(
            index='Depth Ratio',
            columns='Context Length (K)',
            values='Accuracy'
        )
        # Ensure all defined depths and lengths are in the pivot table, fill missing with NaN
        heatmap_data = heatmap_data.reindex(
            index=sorted(list(depth_ratios_seen), reverse=True),
            columns=sorted(list(context_lengths_seen))
        )

        plt.figure(figsize=(max(10, len(context_lengths_seen)), max(8, len(depth_ratios_seen))))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",  # Show accuracy as float
            cmap='viridis',
            vmin=0.0,
            vmax=1.0,
            cbar_kws={'label': 'Accuracy'}
        )
        plt.title(f'Needle in a Haystack Accuracy\nModel: {compressor_name}')
        plt.ylabel('Depth in Context (%)')
        plt.xlabel('Context Length (K tokens)')
        plt.yticks([i + 0.5 for i in range(len(heatmap_data.index))], [f'{x:.0%}' for x in heatmap_data.index], rotation=0)
        plt.xticks(rotation=0)

        # Save the plot
        plt.savefig(heatmap_save_path, bbox_inches='tight')
        plt.close()
        print(f"Heatmap saved to {heatmap_save_path}")

    except Exception as e:
        print(f"Failed to generate heatmap: {e}")


if __name__ == "__main__":
    Fire(test)