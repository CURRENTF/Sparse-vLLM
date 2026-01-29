import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import matplotlib.pyplot as plt
import seaborn as sns  # 新增：用于主题设置
import os
import numpy as np
import fire
from tqdm import tqdm


def patch_model(model):
    # 这里的 raw_forward 已经是绑定到 model 实例的方法了
    raw_forward = model.forward

    def forward(*args, **kwargs):
        input_ids = kwargs.get("input_ids")
        if input_ids is None and len(args) > 0:
            input_ids = args[0]

        if input_ids is not None and input_ids.shape[1] == 1:
            kwargs["output_attentions"] = True
            _backup = model.config._attn_implementation
            model.config._attn_implementation = 'eager'

        res = raw_forward(*args, **kwargs)

        if input_ids is not None and input_ids.shape[1] == 1:
            model.config._attn_implementation = _backup

        return res

    model.forward = forward
    return model


def load_model_and_tokenizer(model_path):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="cuda:0", torch_dtype=torch.bfloat16)
    model = patch_model(model)
    tkn = AutoTokenizer.from_pretrained(model_path)
    return model, tkn


def greedy_decode(input_text, model, tkn, max_new_tokens):
    model.eval()

    inputs = tkn(input_text, return_tensors="pt").to(model.device)
    input_ids_prompt = inputs.input_ids

    current_generated_ids = input_ids_prompt
    collected_decode_attentions = []
    past_key_values = None

    if max_new_tokens == 0:
        raise ValueError

    with torch.no_grad():
        # --- Prefill Phase ---
        prefill_outputs = model(
            input_ids=input_ids_prompt,
            use_cache=True
        )
        logits_prefill = prefill_outputs.logits
        past_key_values = prefill_outputs.past_key_values

        next_token_logits = logits_prefill[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        current_generated_ids = torch.cat([current_generated_ids, next_token_id], dim=-1)

        if tkn.eos_token_id is not None and next_token_id.item() == tkn.eos_token_id:
            print("由于生成EOS，直接退出，收集失败")
            output_text = tkn.decode(current_generated_ids[0], skip_special_tokens=True)
            return output_text, collected_decode_attentions

        # --- Decode Phase ---
        for _ in range(max_new_tokens - 1):
            decode_input_ids = next_token_id

            outputs_decode_step = model(
                input_ids=decode_input_ids,
                past_key_values=past_key_values,
                use_cache=True
            )

            logits_decode_step = outputs_decode_step.logits
            past_key_values = outputs_decode_step.past_key_values

            if outputs_decode_step.attentions is not None and outputs_decode_step.attentions[0] is not None:
                step_attentions = tuple(att.detach().cpu() for att in outputs_decode_step.attentions)
                collected_decode_attentions.append(step_attentions)

            next_token_logits = logits_decode_step[:, -1, :]
            next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            current_generated_ids = torch.cat([current_generated_ids, next_token_id], dim=-1)

            if tkn.eos_token_id is not None and next_token_id.item() == tkn.eos_token_id:
                break

    output_text = tkn.decode(current_generated_ids[0], skip_special_tokens=True)
    return output_text, collected_decode_attentions


def load_data(data_path, sample_num=10):
    data_list = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_num:
                    break
                d = json.loads(line.strip())
                data_list.append(f"Context: {d['context']}\nQuestion: {d['input']}\n")
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        return []
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in {data_path} on line {i + 1}: {e}")
        return data_list
    return data_list


def cumsum_attn_score_for_each_layer(attn_map_at_one_decode_step):
    processed_layers_attn = []
    if attn_map_at_one_decode_step is None:
        return []

    for layer_attn in attn_map_at_one_decode_step:
        if layer_attn.ndim != 4:
            raise ValueError(f"Expected 4D attention tensor, got {layer_attn.ndim}D with shape {layer_attn.shape}")

        if layer_attn.shape[2] == 1:
            squeezed_attn = layer_attn.squeeze(2)  # Shape: (batch_size, num_heads, key_len)
        else:
            raise ValueError(f"attn shape: {layer_attn.shape}")

        # 1. Average over the head dimension
        avg_head_attn = torch.mean(squeezed_attn, dim=1)  # Shape: (batch_size, key_len)
        # 2. Calculate cumulative sum along the key_sequence_length dimension
        cumsum_attn = torch.cumsum(avg_head_attn, dim=-1)  # Shape: (batch_size, key_len)
        processed_layers_attn.append(cumsum_attn)

    return processed_layers_attn


def vis_cumsum_attn_for_each_layer_each_step(step_i, layer_i, attn_cumsum_tensor, output_dir="visualizations"):
    if attn_cumsum_tensor is None or attn_cumsum_tensor.numel() == 0:
        print(f"  Visualization skipped for step {step_i + 1}, layer {layer_i + 1}: Empty tensor.")
        return

    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
        except OSError as e:
            print(f"Error creating directory {output_dir}: {e}. Visualizations will not be saved.")
            return

    # 设置主题
    sns.set_theme(style="whitegrid")

    data_to_plot = attn_cumsum_tensor[0].cpu().float().numpy()

    if data_to_plot.ndim != 1:
        print(f"  Visualization skipped for step {step_i + 1}, layer {layer_i + 1}: Expected 1D data.")
        return

    plt.figure(figsize=(10, 4))
    plt.plot(data_to_plot)
    plt.title(f"Cumulative Attention: Decode Step {step_i + 1}, Layer {layer_i + 1}")
    plt.xlabel("Key Sequence Position (Token Index in KV Cache)")
    plt.ylabel("Cumulative Attention Score (Averaged over Heads)")
    plt.grid(True)

    filename = f"cumsum_step_{step_i + 1}_layer_{layer_i + 1}.jpg"
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, dpi=1000)
    except Exception as e:
        print(f"    Error saving plot to {filepath}: {e}")
    plt.close()


def cal_attn_map_similarity(attn_cumsum_tensor_layer_i, attn_cumsum_tensor_layer_j):
    avg_head_attn_i = torch.mean(attn_cumsum_tensor_layer_i.squeeze(), dim=0)
    avg_head_attn_j = torch.mean(attn_cumsum_tensor_layer_j.squeeze(), dim=0)
    assert avg_head_attn_i.ndim == 1
    vi, idx_i = torch.topk(avg_head_attn_i, k=min(256, avg_head_attn_i.shape[-1]), dim=-1)
    return torch.sum(avg_head_attn_j[idx_i]).item()


def calculate_token_count_for_threshold(attn_map_one_step, threshold=0.9):
    """
    计算每一层每个 head 达到指定注意力分数阈值所需的 token 数量。
    返回: (num_layers, num_heads) tensor, 以及当前 key_len
    """
    counts_per_layer = []
    key_len = 0
    for layer_attn in attn_map_one_step:
        # layer_attn shape: (batch_size, num_heads, 1, key_len)
        squeezed_attn = layer_attn.squeeze(2)  # (batch_size, num_heads, key_len)
        key_len = squeezed_attn.shape[-1]
        
        # 对每个 head 的注意力分数进行降序排序
        sorted_attn, _ = torch.sort(squeezed_attn, dim=-1, descending=True)
        
        # 计算累加分数
        cumsum_attn = torch.cumsum(sorted_attn, dim=-1)
        
        # 找到第一个达到阈值的索引
        indices = torch.argmax((cumsum_attn >= threshold).to(torch.int), dim=-1)
        token_counts = indices + 1
        counts_per_layer.append(token_counts.detach().cpu())
        
    # 假设 batch_size = 1，返回 (num_layers, num_heads)
    return torch.stack(counts_per_layer).squeeze(1), key_len


def vis_token_count_stats(all_step_data, output_dir="visualization"):
    """
    可视化 token 统计信息。
    all_step_data: list of dicts {'counts': tensor(num_layers, num_heads), 'seq_len': int}
    """
    if not all_step_data:
        return
        
    num_layers = all_step_data[0]['counts'].shape[0]
    num_heads = all_step_data[0]['counts'].shape[1]
    
    # 1. 按层汇总数据
    layer_counts = [[] for _ in range(num_layers)]
    layer_ratios = [[] for _ in range(num_layers)]
    
    # 2. 按序列长度汇总数据 (用于分析趋势)
    seq_lens = []
    avg_counts_per_step = []
    
    for step_data in all_step_data:
        counts = step_data['counts'].float().numpy() # (num_layers, num_heads)
        slen = step_data['seq_len']
        seq_lens.append(slen)
        avg_counts_per_step.append(np.mean(counts))
        
        for l in range(num_layers):
            layer_counts[l].extend(counts[l])
            layer_ratios[l].extend(counts[l] / slen)

    # 打印详细表格
    print("\n" + "="*90)
    print(f"Attention Concentration Stats (Threshold=0.9, Total Steps: {len(all_step_data)})")
    print("-" * 90)
    header = f"{'Layer':<6} | {'Mean':<8} | {'Median':<8} | {'H-Min':<7} | {'H-Max':<7} | {'H-Std':<7} | {'Ratio':<8}"
    print(header)
    print("-" * 90)
    
    for i in range(num_layers):
        c_data = np.array(layer_counts[i])
        r_data = np.array(layer_ratios[i])
        print(f"{i:<6} | {np.mean(c_data):<8.1f} | {np.median(c_data):<8.1f} | "
              f"{np.min(c_data):<7.0f} | {np.max(c_data):<7.0f} | {np.std(c_data):<7.1f} | "
              f"{np.mean(r_data):<8.2%}")
    print("="*90)

    # 可视化 1: Layer vs Ratio (归一化后的注意力宽度)
    plt.figure(figsize=(12, 6))
    means_r = [np.mean(r) for r in layer_ratios]
    medians_r = [np.median(r) for r in layer_ratios]
    plt.plot(range(num_layers), means_r, label='Mean Ratio', marker='o')
    plt.plot(range(num_layers), medians_r, label='Median Ratio', marker='s')
    plt.title("Attention Width Ratio per Layer (Token Count / Seq Len)")
    plt.xlabel("Layer Index")
    plt.ylabel("Ratio of Seq Len needed for 90% Attention")
    plt.legend()
    plt.grid(True)
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "layer_attn_ratio.jpg"), dpi=300)
    plt.close()

    # 可视化 2: Seq Len vs Token Count (展示随长度增长的趋势)
    plt.figure(figsize=(12, 6))
    plt.scatter(seq_lens, avg_counts_per_step, alpha=0.5, label='Per Step Avg')
    # 绘制趋势线
    if len(seq_lens) > 1:
        z = np.polyfit(seq_lens, avg_counts_per_step, 1)
        p = np.poly1d(z)
        plt.plot(seq_lens, p(seq_lens), "r--", label='Trend Line')
        
    plt.title("Tokens Needed vs. Sequence Length")
    plt.xlabel("Current Sequence Length (KV Cache Size)")
    plt.ylabel("Avg Tokens for 90% Attention")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "seq_len_vs_token_count.jpg"), dpi=300)
    plt.close()


def vis_layer_similarity_matrix(
        similarity_matrix,
        output_dir="visualization",
        dpi=1000,
        show_values=True,
        annot_fmt=".1f",
        annot_fontsize=6,
        text_color_threshold=0.5
):
    """Visualizes the layer-wise attention similarity matrix as a heatmap."""
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(12, 10))

    if isinstance(similarity_matrix, torch.Tensor):
        matrix_np = similarity_matrix.detach().cpu().numpy()
    elif isinstance(similarity_matrix, np.ndarray):
        matrix_np = similarity_matrix
    else:
        raise TypeError("similarity_matrix must be a PyTorch Tensor or a NumPy array.")

    im = plt.imshow(matrix_np, cmap='viridis', aspect='auto')

    plt.colorbar(label="Attention Similarity Score")
    plt.title(f"Layer Attention Similarity")
    plt.xlabel("Layer Index (j)")
    plt.ylabel("Layer Index (i)")

    num_layers = matrix_np.shape[0]
    if num_layers > 0:
        ticks = list(range(num_layers))
        tick_labels = [str(t) for t in ticks]  # 修改：从 0 开始编号
        tick_fontsize = 10 if num_layers <= 20 else 8
        if num_layers > 30:
            step = max(1, num_layers // 15)
            ticks = list(range(0, num_layers, step))
            tick_labels = [str(t) for t in ticks]

        plt.xticks(ticks, tick_labels, fontsize=tick_fontsize)
        plt.yticks(ticks, tick_labels, fontsize=tick_fontsize)

    if show_values and num_layers > 0:
        actual_vmin, actual_vmax = im.get_clim()
        for i in range(num_layers):
            for j in range(num_layers):
                value = matrix_np[i, j]
                norm_val = 0.5 if actual_vmax == actual_vmin else (value - actual_vmin) / (actual_vmax - actual_vmin)
                text_color = "white" if norm_val < text_color_threshold else "black"
                plt.text(j, i, format(value, annot_fmt), ha="center", va="center", color=text_color, fontsize=annot_fontsize)

    filename = f"Layer Attention Similarity.jpg"
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    try:
        plt.savefig(filepath, dpi=dpi, bbox_inches='tight')
        print(f"    Saved layer similarity heatmap to {filepath}")
    except Exception as e:
        print(f"    Error saving layer similarity plot to {filepath}: {e}")
    finally:
        plt.close()


def main(
        model_path="../models/Llama-3-8B-Instruct-262k",
        data_path="/autodl-fs/data/datasets/LongBench/data/hotpotqa.jsonl",
        sample_num=50,
        max_new_tokens_generation=20
):
    """
    分析模型层间注意力相似度的主函数。
    """
    # 统一设置主题
    sns.set_theme(style="whitegrid")

    print(f"Loading model and tokenizer from: {model_path}")
    model, tkn = load_model_and_tokenizer(model_path)

    print(f"Loading data from: {data_path} (first {sample_num} samples)")
    data_samples = load_data(data_path, sample_num)
    if not data_samples:
        print("No data loaded. Exiting.")
        return

    num_layers = model.config.num_hidden_layers
    layers_sim = torch.zeros(num_layers, num_layers, device="cpu")
    all_token_counts = []
    sim_cal_step = 0

    # 使用 tqdm 监视样本处理进度
    for i, sample in enumerate(tqdm(data_samples, desc="Processing Samples")):
        input_text = sample
        output_text, collected_attentions = greedy_decode(input_text, model, tkn, max_new_tokens=max_new_tokens_generation)

        if not collected_attentions:
            continue

        for step_idx, attn_map_one_step in enumerate(collected_attentions):
            sim_cal_step += 1
            
            # 计算 0.9 阈值所需的 token 数量及当前序列长度
            token_counts, current_seq_len = calculate_token_count_for_threshold(attn_map_one_step, threshold=0.9)
            all_token_counts.append({
                'counts': token_counts, 
                'seq_len': current_seq_len
            })
            
            for layer_i in range(num_layers):
                for layer_j in range(layer_i, num_layers):
                    sim_ij = cal_attn_map_similarity(attn_map_one_step[layer_i], attn_map_one_step[layer_j])
                    layers_sim[layer_i, layer_j] += sim_ij

    if sim_cal_step > 0:
        # 可视化 Token 统计
        vis_token_count_stats(all_token_counts)
        
        layers_sim /= sim_cal_step
        for i in range(num_layers):
            for j in range(i):
                layers_sim[i, j] = layers_sim[j, i]

        vis_layer_similarity_matrix(layers_sim)

        # 额外的滑动窗口相似度趋势图
        sims_lst = []
        window_size = 8
        for i in range(num_layers - window_size + 1):
            avg_sim = torch.mean(layers_sim[i, i:i + window_size]).item()
            sims_lst.append(avg_sim)

        plt.figure(figsize=(10, 6))
        plt.plot(range(len(sims_lst)), sims_lst, marker='o', linestyle='-', linewidth=1.5)  # 添加圆点标记
        plt.title(f"Mean Similarity in {window_size}-layer Window")
        plt.xlabel("Start Layer Index")
        plt.ylabel("Mean Similarity")
        plt.grid(True)  # 显式开启网格（whitegrid已默认开启）

        os.makedirs("visualization", exist_ok=True)
        plt.savefig("visualization/8layers_sim.jpg", dpi=1000)
        print("Completed similarity analysis.")
    else:
        print("No steps were processed for similarity calculation.")


if __name__ == '__main__':
    fire.Fire(main)