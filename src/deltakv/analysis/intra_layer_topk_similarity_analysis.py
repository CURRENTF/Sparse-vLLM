import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import fire
from tqdm import tqdm
import torch.nn.functional as F

# 用于存储抓取到的 QKV 状态
class QKVCollector:
    def __init__(self):
        self.q_states = []
        self.k_states = []
        self.v_states = []

    def clear(self):
        self.q_states = []
        self.k_states = []
        self.v_states = []

collector = QKVCollector()

def get_activation_hook(layer_idx, name):
    def hook(module, input, output):
        # output shape: [batch, seq_len, hidden_size]
        # Qwen2 linear output might be a tuple or tensor depending on implementation, 
        # but usually it's [batch, seq, dim]
        if isinstance(output, tuple):
            output = output[0]
        if name == 'q':
            collector.q_states.append(output.detach().cpu())
        elif name == 'k':
            collector.k_states.append(output.detach().cpu())
        elif name == 'v':
            collector.v_states.append(output.detach().cpu())
    return hook

def patch_model_with_hooks(model):
    for i, layer in enumerate(model.model.layers):
        layer.self_attn.q_proj.register_forward_hook(get_activation_hook(i, 'q'))
        layer.self_attn.k_proj.register_forward_hook(get_activation_hook(i, 'k'))
        layer.self_attn.v_proj.register_forward_hook(get_activation_hook(i, 'v'))
    return model

def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.bfloat16)
    model = patch_model_with_hooks(model)
    tkn = AutoTokenizer.from_pretrained(model_path)
    return model, tkn

def get_prefill_qkv(input_text, model, tkn, max_seq_len=512):
    model.eval()
    inputs = tkn(input_text, return_tensors="pt", truncation=True, max_length=max_seq_len).to(model.device)
    
    collector.clear()
    with torch.no_grad():
        model(input_ids=inputs.input_ids, use_cache=False)
    
    # collector.q_states 包含 num_layers 个 tensor，每个是 [1, seq_len, hidden_size]
    return {
        'q': [s[0].to(torch.float32) for s in collector.q_states],
        'k': [s[0].to(torch.float32) for s in collector.k_states],
        'v': [s[0].to(torch.float32) for s in collector.v_states]
    }

def calculate_cosine_sim_matrix(tensor):
    """
    tensor: [seq_len, dim]
    returns: [seq_len, seq_len] matrix
    """
    normalized_tensor = F.normalize(tensor, p=2, dim=1)
    sim_matrix = torch.mm(normalized_tensor, normalized_tensor.t())
    return sim_matrix

def main(
    model_path,
    data_path="/root/autodl-fs/datasets/LongBench/data/hotpotqa.jsonl",
    sample_num=50,
    max_seq_len=1024,
    ks=[16, 32, 64],
    output_dir="visualization/intra_qkv_topk_sim"
):
    print(f"Loading model: {model_path}")
    model, tkn = load_model_and_tokenizer(model_path)
    
    data_samples = []
    try:
        if os.path.exists(data_path):
            with open(data_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= sample_num: break
                    d = json.loads(line.strip())
                    data_samples.append(f"Context: {d['context']}\nQuestion: {d['input']}\n")
        else:
            print(f"Data path {data_path} not found, using dummy data.")
            data_samples = ["Dummy text for analysis. " * 100] * sample_num
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    num_layers = model.config.num_hidden_layers
    
    # 结果存储: results[key][k][layer_idx] = list of mean top-k sims
    results = {
        'k': {k: torch.zeros(num_layers) for k in ks},
        'v': {k: torch.zeros(num_layers) for k in ks}
    }
    sample_counts = torch.zeros(num_layers)

    for sample_idx, sample in enumerate(tqdm(data_samples, desc="Samples")):
        qkv_data = get_prefill_qkv(sample, model, tkn, max_seq_len)
        seq_len = qkv_data['k'][0].shape[0]
        
        if seq_len <= max(ks):
            print(f"Sample {sample_idx} seq_len {seq_len} is too short for ks {ks}, skipping.")
            continue

        for layer_idx in range(num_layers):
            for key in ['k', 'v']:
                states = qkv_data[key][layer_idx] # [seq_len, dim]
                sim_matrix = calculate_cosine_sim_matrix(states)
                
                # 排除自身相似度 (对角线设为 -1)
                sim_matrix.fill_diagonal_(-1.0)
                
                for k in ks:
                    # 对每个 token 找 top-k 相似度
                    topk_values, _ = torch.topk(sim_matrix, k=k, dim=1) # [seq_len, k]
                    mean_topk = topk_values.mean() # 整个序列的平均 top-k 相似度
                    results[key][k][layer_idx] += mean_topk
            
            sample_counts[layer_idx] += 1

    # 计算平均值
    for key in ['k', 'v']:
        for k in ks:
            results[key][k] /= sample_counts

    # 绘图
    os.makedirs(output_dir, exist_ok=True)
    
    for key in ['k', 'v']:
        plt.figure(figsize=(12, 6))
        for k in ks:
            plt.plot(range(num_layers), results[key][k].numpy(), label=f'Top-{k}', marker='o')
        
        plt.title(f"Average Top-K Cosine Similarity per Layer ({key.upper()})")
        plt.xlabel("Layer Index")
        plt.ylabel("Mean Cosine Similarity")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, f"{key}_topk_sim.png"), dpi=300)
        plt.close()

    # 保存数据
    summary = {
        'ks': ks,
        'results': {
            key: {str(k): results[key][k].tolist() for k in ks}
            for key in ['k', 'v']
        }
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)

    print(f"Top-K analysis completed. Results saved to {output_dir}")

if __name__ == "__main__":
    fire.Fire(main)