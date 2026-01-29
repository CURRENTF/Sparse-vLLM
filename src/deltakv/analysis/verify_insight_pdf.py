import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import fire
from tqdm import tqdm
from deltakv.analysis.colors import (
    COLOR_PRIMARY, COLOR_PRIMARY_LIGHT, COLOR_SECONDARY, COLOR_SECONDARY_LIGHT, 
    COLOR_NEUTRAL, COLOR_GRID, COLOR_BLACK, 
    TEXT_HIGHLIGHT_1, TEXT_HIGHLIGHT_2
)

# Set style for academic plotting
plt.style.use('seaborn-v0_8-paper')
sns.set_palette("husl")

class InsightCollector:
    def __init__(self):
        self.k_states = {}  # layer_idx -> list of tensors

    def clear(self):
        self.k_states = {}

collector = InsightCollector()

def get_activation_hook(layer_idx):
    def hook(module, input, output):
        # Qwen2 attention output is (attn_output, attn_weights, past_key_values) if output_attentions=True
        # But we hook on k_proj. output is [bs, seq_len, num_heads * head_dim]
        # We need to reshape to [bs, seq_len, num_heads, head_dim] then transpose to [bs, num_heads, seq_len, head_dim]
        # However, simple k_proj output is just [bs, seq_len, hidden_dim] (before rope/reshape)
        # To be safe and compatible with various models, let's just grab the projection output.
        # Ideally we want post-RoPE keys, but pre-RoPE is also fine for similarity analysis (often even better for clustering).
        # Let's assume we capture the linear projection output.
        if layer_idx not in collector.k_states:
            collector.k_states[layer_idx] = []
        collector.k_states[layer_idx].append(output.detach().cpu())
    return hook

def patch_model(model):
    # Hook into K projection
    for i, layer in enumerate(model.model.layers):
        # Qwen2 naming
        layer.self_attn.k_proj.register_forward_hook(get_activation_hook(i))
    return model

def load_data(data_path, sample_num=1):
    data_list = []
    import json
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= sample_num:
                    break
                d = json.loads(line.strip())
                # LongBench format
                context = d.get('context', '')
                inp = d.get('input', '')
                data_list.append(f"Context: {context}\nQuestion: {inp}\n")
    except Exception as e:
        print(f"Error loading data from {data_path}: {e}")
        exit(1)
    
    if not data_list:
        print(f"No data loaded from {data_path}")
        exit(1)
        
    return data_list

def run_experiment(
    model_path="Qwen/Qwen2.5-0.5B-Instruct",
    output_dir="/root/autodl-fs/visuals/insight_experiments",
    seq_len=2048,
    sample_num=20,
    device=None
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            device_map=device, 
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            trust_remote_code=True
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model = patch_model(model)
    model.eval()

    print("Preparing data...")
    data_path = "/root/autodl-fs/datasets/LongBench/data/narrativeqa.jsonl"
    # Load multiple samples
    real_data = load_data(data_path, sample_num=sample_num)
    
    # Stats containers (aggregated across all samples)
    all_max_sims = []
    all_distances = []
    
    # For Exp 5 & 6 (Reservoir sampling to save memory)
    # We will collect a fixed number of vectors from each sample
    # to perform SVD and Norm analysis globally.
    reservoir_raw = []
    reservoir_res = []
    reservoir_res_random = []
    MAX_VECTORS_PER_SAMPLE = 2000
    
    print(f"Running analysis on {len(real_data)} samples...")
    
    for sample_idx, input_text in enumerate(tqdm(real_data, desc="Processing Samples")):
        # Clear collector for the new sample
        collector.clear()
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=seq_len, truncation=True).to(device)
        
        with torch.no_grad():
            model(**inputs)
            
        # Analyze layers for this sample
        # Analyze middle layers (representation is usually best there)
        # Note: collector keys are layer indices. We can just use the keys from the first run or fixed indices.
        # Assuming model structure doesn't change, num_layers is constant.
        num_layers = model.config.num_hidden_layers
        target_layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
        
        for layer_idx in target_layers:
            if layer_idx not in collector.k_states:
                continue
                
            # Concatenate collected states (if multiple chunks in one forward, usually just 1 for prefill)
            # k_state shape: [1, seq_len, hidden_dim] (assuming bs=1)
            k_tensor = torch.cat(collector.k_states[layer_idx], dim=1).squeeze(0).float()
            
            # Normalize for Cosine Similarity
            k_norm = F.normalize(k_tensor, p=2, dim=-1)
            
            seq_len_actual = k_norm.shape[0]
            sink_size = 16
            
            if seq_len_actual <= sink_size:
                continue
            
            # Sim Matrix: [seq_len, seq_len]
            sim_matrix = torch.matmul(k_norm, k_norm.t())
            
            # Mask out future and self
            # We want max_{j < t} sim(t, j)
            mask = torch.tril(torch.ones_like(sim_matrix), diagonal=-1)
            
            # Apply mask
            masked_sim = sim_matrix.masked_fill(mask == 0, -1.0)
            
            # For each token t > sink_size
            valid_indices = torch.arange(sink_size, seq_len_actual)
            
            # Get max similarity and the index of that similarity
            # top_vals: [num_valid], top_indices: [num_valid]
            top_vals, top_indices = masked_sim[valid_indices].max(dim=1)
            
            # Calculate distances: current_pos - best_match_pos
            distances = valid_indices.to(top_indices.device) - top_indices
            
            all_max_sims.extend(top_vals.tolist())
            all_distances.extend(distances.tolist())
            
            # --- Collect Vectors for Exp 5 & 6 ---
            # Randomly sample indices to keep memory check
            num_valid = len(valid_indices)
            if num_valid > MAX_VECTORS_PER_SAMPLE:
                perm = torch.randperm(num_valid)[:MAX_VECTORS_PER_SAMPLE]
                sampled_valid_indices = valid_indices[perm]
                sampled_ref_indices = top_indices[perm]
            else:
                sampled_valid_indices = valid_indices
                sampled_ref_indices = top_indices
            
            # Get Raw vectors (Original K)
            raw_vecs = k_tensor[sampled_valid_indices]
            ref_vecs = k_tensor[sampled_ref_indices]
            res_vecs = raw_vecs - ref_vecs
            
            # --- Random Baseline ---
            # For each valid index, pick a random index from [sink_size, current_index)
            # sampled_valid_indices contains absolute positions
            # We need to generate random reference indices < current_pos
            rand_ref_indices = []
            for curr_pos in sampled_valid_indices:
                # Random int between sink_size and curr_pos
                # If curr_pos <= sink_size, this shouldn't happen due to logic above
                if curr_pos > sink_size:
                    r_idx = torch.randint(low=sink_size, high=curr_pos, size=(1,)).item()
                else:
                    r_idx = 0 # Fallback
                rand_ref_indices.append(r_idx)
            rand_ref_indices = torch.tensor(rand_ref_indices, device=k_tensor.device)
            
            ref_random_vecs = k_tensor[rand_ref_indices]
            res_random_vecs = raw_vecs - ref_random_vecs
            
            reservoir_raw.append(raw_vecs.cpu())
            reservoir_res.append(res_vecs.cpu())
            reservoir_res_random.append(res_random_vecs.cpu())

    os.makedirs(output_dir, exist_ok=True)

    # --- Plotting Experiment 1: Max Similarity Distribution ---
    print("Plotting Experiment 1...")
    plt.figure(figsize=(3, 2))
    sns.histplot(all_max_sims, stat="percent", kde=True, bins=50, color=COLOR_PRIMARY_LIGHT)
    plt.xlabel("Max Cosine Similarity with History Token")
    plt.ylabel("Tokens Percentage (%)")
    plt.xlim(0.5, 1.0)
    plt.grid(True, linestyle='--', alpha=0.7, color=COLOR_GRID)
    
    # Add some text stats
    avg_sim = np.mean(all_max_sims)
    plt.axvline(avg_sim, color=COLOR_SECONDARY, linestyle='--', label=f'Mean: {avg_sim:.2f}')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp1_similarity_dist.pdf"), bbox_inches='tight', pad_inches=0.07)
    print(f"Saved exp1_similarity_dist.pdf to {output_dir}")
    plt.close()

    # --- Plotting Experiment 3: Reference Distance Distribution ---
    print("Plotting Experiment 3...")
    plt.figure(figsize=(3, 2))
    
    import pandas as pd
    all_distances = np.array(all_distances)
    # Calculate ratios
    local_mask = all_distances <= 16
    local_ratio = np.mean(local_mask) * 100
    global_ratio = 100 - local_ratio

    # We use a log scale on X axis
    log_bins = np.logspace(np.log10(1), np.log10(np.max(all_distances)), 50)
    
    # weights = np.ones_like(all_distances) * 100.0 / len(all_distances)
    # Using plt.hist for more robust control over split colors
    plt.hist(all_distances[all_distances <= 16], bins=log_bins, color=COLOR_PRIMARY_LIGHT,
             weights=np.ones_like(all_distances[all_distances <= 16]) * 100.0 / len(all_distances), alpha=0.8)
    plt.hist(all_distances[all_distances > 16], bins=log_bins, color=COLOR_SECONDARY_LIGHT,
             weights=np.ones_like(all_distances[all_distances > 16]) * 100.0 / len(all_distances), alpha=0.8)
    
    plt.xscale('log')
    plt.xlabel("Log Distance to Most Similar Token")
    plt.ylabel("Tokens Percentage (%)")
    plt.grid(True, linestyle='--', alpha=0.7, which='major', color=COLOR_GRID)

    # Annotate "Global" vs "Local" with percentages
    ax = plt.gca()
    plt.axvline(x=16, color=COLOR_BLACK, linestyle=':', alpha=0.5)
    
    # Use relative coordinates (transAxes) for robust positioning
    # Left side (Local) - Align Left to avoid clipping
    ax.text(0.05, 0.95, f'Local(CacheGen)\nDistance $\leq$ 16\nTokens: {local_ratio:.1f}%',
            transform=ax.transAxes, ha='left', va='top', fontsize=8,
            color=TEXT_HIGHLIGHT_1,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=TEXT_HIGHLIGHT_1, alpha=0.5))

    # Right side (Global/Ours) - Align Right to avoid clipping
    ax.text(0.95, 0.95, f'Global(Ours)\nDistance > 16\nTokens: {global_ratio:.1f}%',
            transform=ax.transAxes, ha='right', va='top', fontsize=8,
            color=TEXT_HIGHLIGHT_2,
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=TEXT_HIGHLIGHT_2, alpha=0.5))

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp3_reference_distance_dist.pdf"), bbox_inches='tight', pad_inches=0.07)
    print(f"Saved exp3_reference_distance_dist.pdf to {output_dir}")
    plt.close()

    # --- Prepare Data for Exp 5 & 6 ---
    if reservoir_raw:
        # Concatenate all collected vectors
        # Shape: [N_total, hidden_dim]
        all_raw_vecs = torch.cat(reservoir_raw, dim=0)
        all_res_vecs = torch.cat(reservoir_res, dim=0)
        all_res_random_vecs = torch.cat(reservoir_res_random, dim=0)
        
        # --- Experiment 5: SVD Spectrum Analysis ---
        print("Running SVD for Experiment 5...")
        # Center the data? Usually PCA centers data, SVD on raw data is also fine for energy
        # Let's do SVD on raw vectors (energy spectrum)
        
        # Limit size for SVD speed if too huge (e.g., max 20k vectors is enough for distribution)
        MAX_SVD_SAMPLES = 20000
        if all_raw_vecs.shape[0] > MAX_SVD_SAMPLES:
            perm = torch.randperm(all_raw_vecs.shape[0])[:MAX_SVD_SAMPLES]
            svd_raw_input = all_raw_vecs[perm]
            svd_res_input = all_res_vecs[perm]
            svd_rand_input = all_res_random_vecs[perm]
        else:
            svd_raw_input = all_raw_vecs
            svd_res_input = all_res_vecs
            svd_rand_input = all_res_random_vecs
            
        # Compute Singular Values
        _, S_raw, _ = torch.svd(svd_raw_input)
        _, S_res, _ = torch.svd(svd_res_input)
        _, S_rand, _ = torch.svd(svd_rand_input)
        
        # Calculate Cumulative Explained Variance Ratio
        # energy = S^2
        energy_raw = S_raw ** 2
        energy_res = S_res ** 2
        energy_rand = S_rand ** 2
        
        cum_var_raw = torch.cumsum(energy_raw, dim=0) / torch.sum(energy_raw)
        cum_var_res = torch.cumsum(energy_res, dim=0) / torch.sum(energy_res)
        cum_var_rand = torch.cumsum(energy_rand, dim=0) / torch.sum(energy_rand)
        
        # Plot Exp 5
        print("Plotting Experiment 5...")
        plt.figure(figsize=(3, 2))
        x_axis = np.arange(1, len(cum_var_raw) + 1)
        
        plt.plot(x_axis, cum_var_res.numpy(), label='Residual KV (Ours)', color=COLOR_PRIMARY, linewidth=1.5)
        plt.plot(x_axis, cum_var_raw.numpy(), label='Original KV', color=COLOR_NEUTRAL, linestyle='--', linewidth=1.5)
        plt.plot(x_axis, cum_var_rand.numpy(), label='Random Residual', color=COLOR_BLACK, linestyle=':', linewidth=1.5)
        
        plt.xlabel("Number of Components (Rank)")
        plt.ylabel("Cumulative Variance")
        plt.legend(loc='lower right')
        plt.grid(True, linestyle='--', alpha=0.5, color=COLOR_GRID)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "exp5_svd_variance.pdf"), bbox_inches='tight', pad_inches=0.07)
        print(f"Saved exp5_svd_variance.pdf to {output_dir}")
        plt.close()
        
        # --- Experiment 6: Norm Distribution ---
        print("Plotting Experiment 6...")
        # L2 Norm
        norms_raw = torch.norm(all_raw_vecs, p=2, dim=-1).numpy()
        norms_res = torch.norm(all_res_vecs, p=2, dim=-1).numpy()
        norms_rand = torch.norm(all_res_random_vecs, p=2, dim=-1).numpy()
        
        plt.figure(figsize=(3, 2))
        # 统一风格：百分比柱状图 + KDE 曲线
        sns.histplot(norms_raw, label='Original KV', color=COLOR_NEUTRAL, stat="percent", kde=True, bins=30, alpha=0.1)
        sns.histplot(norms_res, label='Residual KV (Ours)', color=COLOR_PRIMARY, stat="percent", kde=True, bins=30, alpha=0.3)
        # sns.histplot(norms_rand, label='Random Residual', color=COLOR_BLACK, stat="percent", kde=True, alpha=0.3, fill=False, linestyle=':', linewidth=0.5)
        
        plt.xlabel("L2 Norm Magnitude")
        plt.ylabel("Tokens Percentage (%)")
        # plt.ylim(0, 3)
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5, color=COLOR_GRID)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "exp6_norm_dist.pdf"), bbox_inches='tight', pad_inches=0.07)
        print(f"Saved exp6_norm_dist.pdf to {output_dir}")
        plt.close()


if __name__ == "__main__":
    fire.Fire(run_experiment)