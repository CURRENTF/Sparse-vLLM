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

def vis_intra_matrix(matrix, title, filename, output_dir):
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap='viridis', annot=False)
    plt.title(title)
    plt.xlabel("Token Index")
    plt.ylabel("Token Index")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def main(
    model_path,
    data_path="/root/autodl-fs/datasets/LongBench/data/hotpotqa.jsonl",
    sample_num=50,
    max_seq_len=512,
    max_new_tokens=20,
    output_dir="visualization/intra_qkv_sim"
):
    print(f"Loading model: {model_path}")
    model, tkn = load_model_and_tokenizer(model_path)
    
    data_samples = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_num: break
                d = json.loads(line.strip())
                data_samples.append(f"Context: {d['context']}\nQuestion: {d['input']}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    num_layers = model.config.num_hidden_layers
    
    # 我们为每一层统计平均相似度随距离变化的曲线
    # dist_sim[layer_idx][distance] = sum_sim
    max_dist = max_seq_len
    q_dist_sim = torch.zeros(num_layers, max_dist)
    k_dist_sim = torch.zeros(num_layers, max_dist)
    v_dist_sim = torch.zeros(num_layers, max_dist)
    dist_count = torch.zeros(max_dist)

    for sample_idx, sample in enumerate(tqdm(data_samples, desc="Samples")):
        qkv_data = get_prefill_qkv(sample, model, tkn, max_seq_len)
        seq_len = qkv_data['q'][0].shape[0]
        
        for layer_idx in range(num_layers):
            for key in ['q', 'k', 'v']:
                states = qkv_data[key][layer_idx] # [seq_len, dim]
                sim_matrix = calculate_cosine_sim_matrix(states)
                
                # 仅在第一个样本的某些层可视化矩阵
                if sample_idx == 0 and layer_idx in [0, num_layers // 2, num_layers - 1]:
                    vis_intra_matrix(
                        sim_matrix.numpy(), 
                        f"Layer {layer_idx} {key.upper()} Intra-Sequence Similarity", 
                        f"layer_{layer_idx}_{key}_intra_sim.png", 
                        output_dir
                    )
                
                # 统计随距离变化的相似度
                for dist in range(seq_len):
                    # 提取对角线偏移
                    diag = torch.diagonal(sim_matrix, offset=dist)
                    if key == 'q': q_dist_sim[layer_idx, dist] += diag.sum()
                    elif key == 'k': k_dist_sim[layer_idx, dist] += diag.sum()
                    elif key == 'v': v_dist_sim[layer_idx, dist] += diag.sum()
                    
                    if layer_idx == 0: # 只需要计一次数
                        dist_count[dist] += diag.numel()

    # 计算平均值并绘图
    plt.figure(figsize=(12, 6))
    valid_dist = (dist_count > 0)
    x = np.arange(max_dist)[valid_dist.numpy()]
    
    # 选几个代表层画图
    for l_idx in [0, num_layers // 2, num_layers - 1]:
        q_y = (q_dist_sim[l_idx][valid_dist] / dist_count[valid_dist]).numpy()
        plt.plot(x, q_y, label=f'Layer {l_idx} Q')
        
    plt.title("Average Intra-Sequence Similarity vs Token Distance")
    plt.xlabel("Distance between tokens")
    plt.ylabel("Mean Cosine Similarity")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "sim_vs_distance.png"), dpi=300)
    plt.close()

    # 层间平均序列相似度（所有距离的平均）
    layer_avg_sim = (q_dist_sim.sum(dim=1) + k_dist_sim.sum(dim=1) + v_dist_sim.sum(dim=1)) / (3 * dist_count.sum())
    plt.figure(figsize=(10, 5))
    plt.bar(range(num_layers), layer_avg_sim.numpy())
    plt.title("Global Average Intra-Sequence Similarity per Layer")
    plt.xlabel("Layer Index")
    plt.ylabel("Average Cosine Similarity")
    plt.savefig(os.path.join(output_dir, "layer_avg_intra_sim.png"), dpi=300)
    plt.close()

    print(f"Intra-sequence analysis completed. Results saved to {output_dir}")

if __name__ == "__main__":
    fire.Fire(main)
