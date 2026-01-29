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
        # output shape: [batch, seq_len, num_heads * head_dim] or [batch, seq_len, hidden_size]
        # 我们只关心最新的 token (decode 阶段 seq_len=1)
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

def greedy_decode(input_text, model, tkn, max_new_tokens):
    model.eval()

    inputs = tkn(input_text, return_tensors="pt").to(model.device)
    input_ids_prompt = inputs.input_ids

    current_generated_ids = input_ids_prompt
    all_step_qkv = []
    past_key_values = None

    if max_new_tokens == 0:
        raise ValueError

    with torch.no_grad():
        # --- Prefill Phase ---
        # 我们这里暂时只关注 decode 阶段的层间相似度，或者可以将 prefill 的最后一个 token 纳入
        collector.clear()
        prefill_outputs = model(
            input_ids=input_ids_prompt,
            use_cache=True
        )
        logits_prefill = prefill_outputs.logits
        past_key_values = prefill_outputs.past_key_values

        # 收集 prefill 最后一个 token 的 QKV
        # collector 中现在存的是整个 sequence 的 QKV，取最后一个
        q = [s[:, -1:, :] for s in collector.q_states]
        k = [s[:, -1:, :] for s in collector.k_states]
        v = [s[:, -1:, :] for s in collector.v_states]
        all_step_qkv.append({'q': q, 'k': k, 'v': v})

        next_token_logits = logits_prefill[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        current_generated_ids = torch.cat([current_generated_ids, next_token_id], dim=-1)

        if tkn.eos_token_id is not None and next_token_id.item() == tkn.eos_token_id:
            return tkn.decode(current_generated_ids[0], skip_special_tokens=True), all_step_qkv

        # --- Decode Phase ---
        for _ in range(max_new_tokens - 1):
            collector.clear()
            decode_input_ids = next_token_id

            outputs_decode_step = model(
                input_ids=decode_input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )

            logits_decode_step = outputs_decode_step.logits
            past_key_values = outputs_decode_step.past_key_values

            # collector 中现在存的是这一个 step 的 QKV
            q = [s.clone() for s in collector.q_states]
            k = [s.clone() for s in collector.k_states]
            v = [s.clone() for s in collector.v_states]
            all_step_qkv.append({'q': q, 'k': k, 'v': v})

            next_token_logits = logits_decode_step[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            current_generated_ids = torch.cat([current_generated_ids, next_token_id], dim=-1)

            if tkn.eos_token_id is not None and next_token_id.item() == tkn.eos_token_id:
                break

    output_text = tkn.decode(current_generated_ids[0], skip_special_tokens=True)
    return output_text, all_step_qkv

def load_data(data_path, sample_num=10):
    data_list = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_num:
                    break
                d = json.loads(line.strip())
                data_list.append(f"Context: {d['context']}\nQuestion: {d['input']}\n")
    except Exception as e:
        print(f"Error loading data: {e}")
    return data_list

def calculate_cosine_similarity(states_i, states_j):
    """
    states_i: [1, 1, dim]
    """
    si = states_i.view(-1).to(torch.float32)
    sj = states_j.view(-1).to(torch.float32)
    return F.cosine_similarity(si, sj, dim=0).item()

def vis_matrix(matrix, title, filename, output_dir):
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap='viridis', annot=False)
    plt.title(title)
    plt.xlabel("Layer Index")
    plt.ylabel("Layer Index")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300)
    plt.close()

def main(
    model_path,
    data_path="/root/autodl-fs/datasets/LongBench/data/hotpotqa.jsonl",
    sample_num=50,
    max_new_tokens=20,
    output_dir="visualization/qkv_sim"
):
    print(f"Loading model: {model_path}")
    model, tkn = load_model_and_tokenizer(model_path)
    
    data_samples = load_data(data_path, sample_num)
    num_layers = model.config.num_hidden_layers
    
    q_sim_total = torch.zeros(num_layers, num_layers)
    k_sim_total = torch.zeros(num_layers, num_layers)
    v_sim_total = torch.zeros(num_layers, num_layers)
    total_steps = 0
    
    for sample in tqdm(data_samples, desc="Samples"):
        _, all_step_qkv = greedy_decode(sample, model, tkn, max_new_tokens)
        
        for step_data in all_step_qkv:
            total_steps += 1
            qs = step_data['q']
            ks = step_data['k']
            vs = step_data['v']
            
            # 这里的 qs 是一个 list，长度为 num_layers
            for i in range(num_layers):
                for j in range(i, num_layers):
                    q_sim = calculate_cosine_similarity(qs[i], qs[j])
                    k_sim = calculate_cosine_similarity(ks[i], ks[j])
                    v_sim = calculate_cosine_similarity(vs[i], vs[j])
                    
                    q_sim_total[i, j] += q_sim
                    k_sim_total[i, j] += k_sim
                    v_sim_total[i, j] += v_sim
                    
    if total_steps > 0:
        q_sim_avg = q_sim_total / total_steps
        k_sim_avg = k_sim_total / total_steps
        v_sim_avg = v_sim_total / total_steps
        
        # 填充下三角
        for i in range(num_layers):
            for j in range(i):
                q_sim_avg[i, j] = q_sim_avg[j, i]
                k_sim_avg[i, j] = k_sim_avg[j, i]
                v_sim_avg[i, j] = v_sim_avg[j, i]
        
        vis_matrix(q_sim_avg.numpy(), "Query States Cosine Similarity", "q_sim.png", output_dir)
        vis_matrix(k_sim_avg.numpy(), "Key States Cosine Similarity", "k_sim.png", output_dir)
        vis_matrix(v_sim_avg.numpy(), "Value States Cosine Similarity", "v_sim.png", output_dir)
        
        # 计算相邻层相似度趋势
        q_adj = [q_sim_avg[i, i+1].item() for i in range(num_layers-1)]
        k_adj = [k_sim_avg[i, i+1].item() for i in range(num_layers-1)]
        v_adj = [v_sim_avg[i, i+1].item() for i in range(num_layers-1)]
        
        plt.figure(figsize=(10, 6))
        plt.plot(q_adj, label='Query')
        plt.plot(k_adj, label='Key')
        plt.plot(v_adj, label='Value')
        plt.title("Adjacent Layer Cosine Similarity")
        plt.xlabel("Layer Index (i to i+1)")
        plt.ylabel("Cosine Similarity")
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, "adjacent_sim.png"), dpi=300)
        plt.close()
        
    print(f"Results saved to {output_dir}")

if __name__ == "__main__":
    fire.Fire(main)
