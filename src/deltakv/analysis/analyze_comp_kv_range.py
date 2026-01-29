import torch
import matplotlib.pyplot as plt
import numpy as np
import os
import fire
from tqdm import tqdm
from transformers import AutoTokenizer
from datasets import load_from_disk
from deltakv.configs.model_config_cls import KVQwen2Config
from deltakv.modeling.qwen2.qwen2_e2e_cluster import Qwen2KVClusterCompress
from deltakv.get_chat_api import load_compressor
from deltakv.analysis.colors import (
    COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_TERTIARY_LIGHT,
    COLOR_PRIMARY_LIGHT, COLOR_SECONDARY_LIGHT, COLOR_NEUTRAL, COLOR_SPECIAL
)
import seaborn as sns

os.environ['ANALYSIS'] = '1'

def fake_quantize_4bit(x, bits=4, group_size=-1):
    """
    模拟 Per-token 非对称量化
    x shape: (bs, seq_len, dim)
    """
    shape = x.shape
    if group_size > 0:
        # Group-wise 逻辑 (可选)
        x = x.view(*shape[:-1], -1, group_size)
    
    # 计算每个 token 的 min/max
    # dim 为最后一个维度
    mn = x.min(dim=-1, keepdim=True)[0]
    mx = x.max(dim=-1, keepdim=True)[0]
    
    qmax = 2**bits - 1
    scale = (mx - mn) / (qmax + 1e-6)
    
    # 量化
    x_q = torch.round((x - mn) / (scale + 1e-6)).clamp(0, qmax)
    
    # 反量化
    x_dq = x_q * scale + mn
    
    if group_size > 0:
        x_dq = x_dq.view(shape)
    return x_dq

def analyze_range(model, tokenizer, samples, device="cuda", do_quant=True):
    model.to(device)
    model.eval()
    
    from deltakv.modeling.qwen2 import qwen2_e2e_cluster
    
    results = {
        "bf16_loss": [],
        "int4_loss": [],
        "bf16_mse": [],
        "int4_mse": []
    }
    
    all_layer_values = [[] for _ in range(len(model.model.layers))]
    all_raw_layer_values = [[] for _ in range(len(model.model.layers))]
    all_ideal_res_values = [[] for _ in range(len(model.model.layers))]
    sample_3d_data = None

    # 保存原始方法
    original_recon_methods = []
    for layer in model.model.layers:
        original_recon_methods.append(layer.self_attn.comp_then_reconstruct)

    print(f"\n[Analysis] Running simulation on {len(samples)} samples...")
    for sample in tqdm(samples):
        if isinstance(sample, str):
            inputs = tokenizer(sample, return_tensors="pt").to(device)
        else:
            input_ids = torch.tensor(sample["input_ids"]).unsqueeze(0).to(device)
            inputs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
        
        # 1. BF16 模式 (默认)
        qwen2_e2e_cluster.CURRENT_RUN_MODE = 'comp'
        with torch.no_grad():
            out_bf16 = model(**inputs, labels=inputs["input_ids"])
            
            # 计算 MSE 并还原 NTP Loss
            sum_mse_bf16 = 0
            count_layers = 0
            for i, layer in enumerate(model.model.layers):
                attn = layer.self_attn
                if attn.buffer_recon_kv is not None and attn.buffer_raw_kv is not None:
                    layer_mse = torch.nn.functional.mse_loss(attn.buffer_recon_kv, attn.buffer_raw_kv).item()
                    sum_mse_bf16 += layer_mse
                    count_layers += 1
                    # 收集数值用于范围统计
                    if attn.buffer_comp_kv is not None:
                        all_layer_values[i].append(attn.buffer_comp_kv.detach().cpu().float().numpy().flatten())
                    if attn.buffer_raw_kv is not None:
                        all_raw_layer_values[i].append(attn.buffer_raw_kv.detach().cpu().float().numpy().flatten())
                    if getattr(attn, 'buffer_ideal_res', None) is not None:
                        all_ideal_res_values[i].append(attn.buffer_ideal_res.detach().cpu().float().numpy().flatten())
            
            # model 返回的是 ntp_loss + sum_mse
            pure_ntp_bf16 = out_bf16.loss.item() - sum_mse_bf16
            results["bf16_loss"].append(pure_ntp_bf16)
            results["bf16_mse"].append(sum_mse_bf16 / count_layers if count_layers > 0 else 0)

        # 2. INT4 模拟量化模式
        if do_quant:
            # 动态替换方法注入量化
            def make_quant_recon(attn_module, old_method):
                def quant_recon(key_states, value_states):
                    bs, seq_len, k_dim = key_states.shape
                    kv_flat = torch.cat([key_states, value_states], dim=-1)
                    sink_size = 16
                    old_method(key_states, value_states)
                    
                    # 拦截并量化
                    comp_kv_bf16 = attn_module.buffer_comp_kv
                    comp_kv_int4 = fake_quantize_4bit(comp_kv_bf16, bits=4)
                    
                    # 重新使用量化后的值进行 Up 投影 and 重建
                    up_comp_bf16 = attn_module.compress_up(comp_kv_bf16)
                    recon_bf16 = attn_module.buffer_recon_kv
                    recon_int4 = (recon_bf16[:, sink_size:, :] - up_comp_bf16) + attn_module.compress_up(comp_kv_int4)
                    
                    # 更新 buffer
                    attn_module.buffer_recon_kv = torch.cat([recon_bf16[:, :sink_size, :], recon_int4], dim=1)
                    return torch.split(attn_module.buffer_recon_kv, k_dim, dim=-1)
                return quant_recon

            for layer in model.model.layers:
                layer.self_attn.comp_then_reconstruct = make_quant_recon(layer.self_attn, layer.self_attn.comp_then_reconstruct)
            
            with torch.no_grad():
                out_int4 = model(**inputs, labels=inputs["input_ids"])
                
                sum_mse_int4 = 0
                for i, layer in enumerate(model.model.layers):
                    attn = layer.self_attn
                    sum_mse_int4 += torch.nn.functional.mse_loss(attn.buffer_recon_kv, attn.buffer_raw_kv).item()
                
                pure_ntp_int4 = out_int4.loss.item() - sum_mse_int4
                results["int4_loss"].append(pure_ntp_int4)
                results["int4_mse"].append(sum_mse_int4 / count_layers if count_layers > 0 else 0)

            # 恢复原始方法，防止干扰下一次循环
            for i, layer in enumerate(model.model.layers):
                layer.self_attn.comp_then_reconstruct = original_recon_methods[i]

    # 统计数值范围
    layer_stats = []
    final_all_values = []
    final_all_raw_values = []
    final_all_ideal_res_values = []
    
    print("\n[Analysis] Computing Range Statistics...")
    for i, (values_list, raw_values_list, ideal_res_list) in enumerate(zip(all_layer_values, all_raw_layer_values, all_ideal_res_values)):
        if not values_list: continue
        if i == len(model.model.layers) // 2 and sample_3d_data is None:
             dim = model.config.kv_compressed_size
             sample_3d_data = values_list[0].reshape(-1, dim)
        
        flat_kv = np.concatenate(values_list)
        stats = {
            "layer": i,
            "min": np.min(flat_kv),
            "max": np.max(flat_kv),
            "mean": np.mean(flat_kv),
            "std": np.std(flat_kv),
            "abs_max": np.max(np.abs(flat_kv)),
            "p99": np.percentile(flat_kv, 99),
            "p01": np.percentile(flat_kv, 1)
        }
        layer_stats.append(stats)
        final_all_values.extend(np.random.choice(flat_kv, min(100000, len(flat_kv)), replace=False).tolist())
        
        if raw_values_list:
            flat_raw_kv = np.concatenate(raw_values_list)
            final_all_raw_values.extend(np.random.choice(flat_raw_kv, min(100000, len(flat_raw_kv)), replace=False).tolist())
            
        if ideal_res_list:
            flat_res_kv = np.concatenate(ideal_res_list)
            final_all_ideal_res_values.extend(np.random.choice(flat_res_kv, min(100000, len(flat_res_kv)), replace=False).tolist())

    # 输出 Loss 对比结果
    print("\n" + "="*50)
    print(f"{ 'Metric':<15} | { 'BF16 (Mean)':<15} | { 'INT4 (Mean)':<15} | { 'Delta':<10}")
    print("-" * 50)
    avg_bf16_loss = np.mean(results["bf16_loss"])
    avg_int4_loss = np.mean(results["int4_loss"])
    avg_bf16_mse = np.mean(results["bf16_mse"])
    avg_int4_mse = np.mean(results["int4_mse"])
    
    print(f"{ 'NTP Loss':<15} | {avg_bf16_loss:<15.4f} | {avg_int4_loss:<15.4f} | {avg_int4_loss - avg_bf16_loss:<+10.4f}")
    print(f"{ 'KV MSE':<15} | {avg_bf16_mse:<15.6f} | {avg_int4_mse:<15.6f} | {avg_int4_mse - avg_bf16_mse:<+10.6f}")
    print("="*50)
    
    return layer_stats, np.array(final_all_values), np.array(final_all_raw_values), np.array(final_all_ideal_res_values), sample_3d_data

def plot_3d_analysis(sample_data, output_path):
    if sample_data is None: return
    # Standardize style
    plt.style.use('seaborn-v0_8-paper')
    
    fig = plt.figure(figsize=(4.8, 2.8))  # 稍微加宽一点点
    ax = fig.add_subplot(111, projection='3d')
    
    Z = np.abs(sample_data)
    seq_len, dim = Z.shape
    if seq_len > 256: Z = Z[:256, :]; seq_len = 256
    X_idx, Y_idx = np.arange(dim), np.arange(seq_len)
    X, Y = np.meshgrid(X_idx, Y_idx)
    
    # 使用 antialiased=False 减少高频噪点，提高渲染清晰度
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9, antialiased=False)
    
    # ax.set_title("3D Abs-Value Distribution")
    ax.set_xlabel('Channel', fontsize=8, labelpad=-3)
    ax.set_ylabel('Token', fontsize=8, labelpad=-3)
    ax.set_zlabel('Abs Value', fontsize=8, labelpad=-3, rotation=90)
    ax.tick_params(axis='both', which='major', labelsize=7, pad=-1)
    
    # 进一步调整相机距离，让物体更小一些，确保文字不超限
    ax.dist = 12
    
    # 显式添加颜色轴，拉近距离 (pad 从 0.08 降到 0.02)
    cbar = fig.colorbar(surf, ax=ax, shrink=0.4, aspect=12, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label('Amp', fontsize=8)
    
    # 调整视角，让分布看起来更立体
    ax.view_init(elev=30, azim=225)
    
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.07, dpi=300)
    plt.close()
    print(f"[Visualization] Compact 3D plot saved to {output_path}")

def plot_distribution(layer_stats, all_values, all_raw_values, all_ideal_res_values, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Standardize style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")
    
    # 1. Global Distribution (Comp KV)
    plt.figure(figsize=(3, 2.5))
    # 使用 weights 将 y 轴转换为比例，并减少 bins
    weights = np.ones_like(all_values) * 100 / len(all_values)
    plt.hist(all_values, bins=50, color=COLOR_PRIMARY_LIGHT, alpha=0.8, weights=weights)
    plt.xlabel("Compressed KV Value Distribution")
    plt.ylabel("Percentage (%)")
    plt.yscale('log')
    from matplotlib.ticker import LogFormatterMathtext
    plt.gca().yaxis.set_major_formatter(LogFormatterMathtext())
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comp_kv_global_dist.pdf"), bbox_inches='tight', pad_inches=0.07)
    plt.close()

    # 1.5 Global Distribution (Raw KV)
    if all_raw_values is not None and len(all_raw_values) > 0:
        plt.figure(figsize=(3, 2))
        weights = np.ones_like(all_raw_values) * 100 / len(all_raw_values)
        plt.hist(all_raw_values, bins=50, color=COLOR_SECONDARY_LIGHT, alpha=0.8, weights=weights)
        plt.xlabel("Value")
        plt.ylabel("Percentage (%)")
        plt.yscale('log')
        from matplotlib.ticker import LogFormatterMathtext
        plt.gca().yaxis.set_major_formatter(LogFormatterMathtext())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "raw_kv_global_dist.pdf"), bbox_inches='tight', pad_inches=0.07)
        plt.close()

    # 1.6 Global Distribution (Ideal Residual)
    if all_ideal_res_values is not None and len(all_ideal_res_values) > 0:
        plt.figure(figsize=(3, 2))
        weights = np.ones_like(all_ideal_res_values) * 100 / len(all_ideal_res_values)
        # Use a distinct color (e.g., green/teal from husl palette) or specific constant
        # COLOR_TERTIARY_LIGHT isn't defined in snippet, using a hardcoded one or from palette
        plt.hist(all_ideal_res_values, bins=50, color=COLOR_TERTIARY_LIGHT, alpha=0.8, weights=weights)
        plt.xlabel("Value")
        plt.ylabel("Percentage (%)")
        plt.yscale('log')
        from matplotlib.ticker import LogFormatterMathtext
        plt.gca().yaxis.set_major_formatter(LogFormatterMathtext())
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "ideal_res_global_dist.pdf"), bbox_inches='tight', pad_inches=0.07)
        plt.close()

    # 1.7 Global Distribution Fit (Raw KV) - New
    # if all_raw_values is not None and len(all_raw_values) > 0:
    #     plt.figure(figsize=(3, 2))
    #     sns.kdeplot(all_raw_values, color=COLOR_SECONDARY_LIGHT, fill=True, alpha=0.3)
    #     plt.yscale('log')
    #     plt.xlim(-400, 200)
    #     plt.xlabel("Value")
    #     plt.ylabel("Log Frequency")
    #     plt.gca().set_yticks([])
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, "raw_kv_global_dist_fit.pdf"), bbox_inches='tight', pad_inches=0.07)
    #     plt.close()

    # 1.8 Global Distribution Fit (Ideal Residual) - New
    # if all_ideal_res_values is not None and len(all_ideal_res_values) > 0:
    #     plt.figure(figsize=(3, 2))
    #     sns.kdeplot(all_ideal_res_values, color=COLOR_TERTIARY, fill=True, alpha=0.3)
    #     plt.yscale('log')
    #     plt.xlim(-20, 20)
    #     plt.xlabel("Value")
    #     plt.ylabel("Log Frequency")
    #     plt.gca().set_yticks([])
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(output_dir, "ideal_res_global_dist_fit.pdf"), bbox_inches='tight', pad_inches=0.07)
    #     plt.close()

    # 2. Range across layers
    plt.figure(figsize=(3, 2))
    layers = [s['layer'] for s in layer_stats]
    mins = [s['min'] for s in layer_stats]
    maxs = [s['max'] for s in layer_stats]
    p01s = [s['p01'] for s in layer_stats]
    p99s = [s['p99'] for s in layer_stats]
    means = [s['mean'] for s in layer_stats]
    
    plt.fill_between(layers, mins, maxs, color=COLOR_NEUTRAL, alpha=0.2, label='Min-Max Range')
    plt.fill_between(layers, p01s, p99s, color=COLOR_PRIMARY_LIGHT, alpha=0.3, label='1st-99th Percentile')
    plt.plot(layers, means, color=COLOR_SECONDARY, label='Mean')
    plt.xlabel("Layer Index")
    plt.ylabel("Value Range")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comp_kv_layer_range.pdf"), bbox_inches='tight', pad_inches=0.07)
    plt.close()

    # 3. Absolute Maximum Value per Layer
    plt.figure(figsize=(3, 2))
    plt.bar(layers, [s['abs_max'] for s in layer_stats], color=COLOR_SECONDARY_LIGHT)
    plt.xlabel("Layer Index")
    plt.ylabel("Abs Max")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comp_kv_layer_abs_max.pdf"), bbox_inches='tight', pad_inches=0.07)
    plt.close()

    # 4. Standard Deviation per Layer
    plt.figure(figsize=(3, 2))
    plt.plot(layers, [s['std'] for s in layer_stats], marker='o', color=COLOR_TERTIARY)
    plt.xlabel("Layer Index")
    plt.ylabel("Std Dev")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "comp_kv_layer_std.pdf"), bbox_inches='tight', pad_inches=0.07)
    plt.close()
    
    print(f"[Visualization] Individual plots saved to {output_dir}")

def main(
    model_path="/root/autodl-fs/models/Qwen2.5-7B-Instruct-1M",
    compressor_path=None,
    dataset_path="/root/autodl-fs/datasets/deltakv_qwen_train_num80000_seqlen8192/",
    num_samples=10,
    output_dir="/root/autodl-fs/visuals/comp_kv_range",
    text=None,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    print(f"[Init] Loading config and model from {model_path}...")
    config = KVQwen2Config.from_pretrained(compressor_path if compressor_path else model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen2KVClusterCompress.from_pretrained(model_path, config=config, torch_dtype=torch.bfloat16, device_map=0)
    
    if compressor_path:
        print(f"[Init] Loading compressor weights from {compressor_path}...")
        state_dict = load_compressor(compressor_path=compressor_path)
        msg = model.load_state_dict(state_dict, strict=False)
        print(f"[Init] Load Status: {msg}")

    samples = []
    if text:
        samples = [text]
    else:
        print(f"[Data] Loading dataset from {dataset_path}...")
        tokenized_dataset = load_from_disk(dataset_path)
        train_dataset = tokenized_dataset
        samples = [train_dataset[i] for i in range(min(num_samples, len(train_dataset)))]
    
    layer_stats, all_values, all_raw_values, all_ideal_res_values, sample_3d = analyze_range(model, tokenizer, samples, device=device)
    
    plot_distribution(layer_stats, all_values, all_raw_values, all_ideal_res_values, output_dir)
    plot_3d_analysis(sample_3d, os.path.join(output_dir, "comp_kv_3d_abs_dist.pdf"))

if __name__ == "__main__":
    fire.Fire(main)
