import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import numpy as np
import fire
from tqdm import tqdm
from typing import Optional
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
        self.target_layers = None

    def clear(self):
        self.k_states = {}

    def set_target_layers(self, target_layers):
        self.target_layers = None if target_layers is None else set(target_layers)

collector = InsightCollector()

def get_activation_hook(layer_idx):
    def hook(module, input, output):
        if collector.target_layers is not None and layer_idx not in collector.target_layers:
            return
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

def patch_model(model, target_layers):
    collector.set_target_layers(target_layers)
    # Hook into K projection
    for i, layer in enumerate(model.model.layers):
        if i not in collector.target_layers:
            continue
        layer.self_attn.k_proj.register_forward_hook(get_activation_hook(i))
    return model


def parse_target_layers(target_layers, num_layers):
    if target_layers is None:
        layers = [num_layers // 4, num_layers // 2, 3 * num_layers // 4]
    elif isinstance(target_layers, str):
        layers = [int(x.strip()) for x in target_layers.split(",") if x.strip()]
    elif isinstance(target_layers, int):
        layers = [int(target_layers)]
    else:
        layers = [int(x) for x in target_layers]
    layers = sorted({layer for layer in layers if 0 <= layer < num_layers})
    if not layers:
        raise ValueError("No valid target layers were resolved.")
    return layers


def parse_alpha_values(alpha_values):
    if alpha_values is None:
        alpha_values = [0.0, 0.0001, 0.001, 0.01, 0.05, 0.1]
    elif isinstance(alpha_values, str):
        alpha_values = [float(x.strip()) for x in alpha_values.split(",") if x.strip()]
    elif isinstance(alpha_values, (int, float)):
        alpha_values = [float(alpha_values)]
    else:
        alpha_values = [float(x) for x in alpha_values]

    deduped = []
    seen = set()
    for alpha in alpha_values:
        key = float(alpha)
        if key not in seen:
            seen.add(key)
            deduped.append(key)
    return deduped


def resolve_data_path(data_path):
    candidates = [
        data_path,
        "/home/haojitai/datasets/LongBench/data/narrativeqa.jsonl",
        "/root/autodl-fs/datasets/LongBench/data/narrativeqa.jsonl",
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    raise FileNotFoundError(
        "Could not find a valid LongBench jsonl file. "
        f"Tried: {candidates}"
    )


def build_center_indices(seq_len, sink_size, cluster_ratio, stride_alpha):
    if cluster_ratio <= 0:
        raise ValueError(f"cluster_ratio must be positive, got {cluster_ratio}")

    sink_size = min(int(sink_size), int(seq_len))
    centers = list(range(sink_size))
    if sink_size >= seq_len:
        return torch.tensor(centers, dtype=torch.long)

    base_step = max(1, int(1 / cluster_ratio))
    pos = sink_size
    while pos < seq_len:
        centers.append(pos)
        if stride_alpha <= 0.0:
            step = base_step
        else:
            t = max(0, pos - sink_size)
            step = base_step + int(float(stride_alpha) * float(t))
            if step < 1:
                step = 1
        pos += step

    return torch.tensor(centers, dtype=torch.long)


def compute_best_similarity(
    k_norm: torch.Tensor,
    sink_size: int,
    *,
    candidate_indices: Optional[torch.Tensor] = None,
    allow_self: bool = False,
    query_block_size: int = 1024,
):
    seq_len = k_norm.shape[0]
    device = k_norm.device
    if seq_len <= sink_size:
        empty = torch.empty((0,), dtype=k_norm.dtype, device=device)
        empty_idx = torch.empty((0,), dtype=torch.long, device=device)
        return empty, empty_idx

    best_vals = []
    best_indices = []

    if candidate_indices is None:
        for q_start in range(sink_size, seq_len, query_block_size):
            q_end = min(seq_len, q_start + query_block_size)
            q = k_norm[q_start:q_end]
            keys = k_norm[:q_end]
            scores = torch.matmul(q, keys.t())

            row_pos = torch.arange(q_start, q_end, device=device).view(-1, 1)
            col_pos = torch.arange(0, q_end, device=device).view(1, -1)
            valid_mask = col_pos <= row_pos if allow_self else col_pos < row_pos
            scores = scores.masked_fill(~valid_mask, -1.0)

            vals, idx = scores.max(dim=1)
            best_vals.append(vals)
            best_indices.append(idx)
    else:
        candidate_indices = candidate_indices.to(device=device, dtype=torch.long)
        candidate_states = k_norm[candidate_indices]
        for q_start in range(sink_size, seq_len, query_block_size):
            q_end = min(seq_len, q_start + query_block_size)
            q = k_norm[q_start:q_end]
            scores = torch.matmul(q, candidate_states.t())

            row_pos = torch.arange(q_start, q_end, device=device).view(-1, 1)
            valid_mask = (
                candidate_indices.view(1, -1) <= row_pos
                if allow_self
                else candidate_indices.view(1, -1) < row_pos
            )
            scores = scores.masked_fill(~valid_mask, -1.0)

            if not torch.all(valid_mask.any(dim=1)):
                raise RuntimeError("Some query positions do not have any valid historical centers.")

            vals, rel_idx = scores.max(dim=1)
            idx = candidate_indices[rel_idx]
            best_vals.append(vals)
            best_indices.append(idx)

    return torch.cat(best_vals, dim=0), torch.cat(best_indices, dim=0)


def _safe_mean(values):
    return float(np.mean(values)) if len(values) > 0 else float("nan")


def _safe_median(values):
    return float(np.median(values)) if len(values) > 0 else float("nan")


def summarize_cluster_similarity(
    baseline_sims,
    baseline_distances,
    cluster_history_sims,
    cluster_history_distances,
    cluster_runtime_sims,
    cluster_runtime_distances,
    center_fractions,
    alpha_values,
    long_range_threshold,
):
    baseline_sims = np.asarray(baseline_sims, dtype=np.float32)
    baseline_distances = np.asarray(baseline_distances, dtype=np.float32)
    long_range_mask = baseline_distances > float(long_range_threshold)

    summary = {
        "baseline": {
            "num_tokens": int(baseline_sims.shape[0]),
            "overall_mean_similarity": _safe_mean(baseline_sims),
            "overall_median_similarity": _safe_median(baseline_sims),
            "overall_mean_distance": _safe_mean(baseline_distances),
            "long_range_threshold": int(long_range_threshold),
            "long_range_token_ratio": float(long_range_mask.mean()) if baseline_sims.size else 0.0,
            "long_range_mean_similarity": _safe_mean(baseline_sims[long_range_mask]),
            "long_range_median_similarity": _safe_median(baseline_sims[long_range_mask]),
            "long_range_mean_distance": _safe_mean(baseline_distances[long_range_mask]),
        },
        "alphas": [],
    }

    baseline_overall = summary["baseline"]["overall_mean_similarity"]
    baseline_long = summary["baseline"]["long_range_mean_similarity"]

    for alpha in alpha_values:
        hist_sims = np.asarray(cluster_history_sims[alpha], dtype=np.float32)
        hist_distances = np.asarray(cluster_history_distances[alpha], dtype=np.float32)
        runtime_sims = np.asarray(cluster_runtime_sims[alpha], dtype=np.float32)
        runtime_distances = np.asarray(cluster_runtime_distances[alpha], dtype=np.float32)

        if hist_sims.shape != baseline_sims.shape:
            raise ValueError(
                f"Shape mismatch for alpha={alpha}: "
                f"baseline={baseline_sims.shape}, history={hist_sims.shape}"
            )

        hist_overall = _safe_mean(hist_sims)
        hist_long = _safe_mean(hist_sims[long_range_mask])
        runtime_overall = _safe_mean(runtime_sims)
        runtime_long = _safe_mean(runtime_sims[long_range_mask])

        summary["alphas"].append(
            {
                "stride_alpha": float(alpha),
                "avg_center_fraction": _safe_mean(center_fractions[alpha]),
                "num_tokens": int(hist_sims.shape[0]),
                "history_only": {
                    "overall_mean_similarity": hist_overall,
                    "overall_abs_drop": float(baseline_overall - hist_overall),
                    "overall_rel_drop_pct": float((baseline_overall - hist_overall) / baseline_overall * 100.0) if baseline_overall else 0.0,
                    "overall_mean_distance": _safe_mean(hist_distances),
                    "long_range_mean_similarity": hist_long,
                    "long_range_abs_drop": float(baseline_long - hist_long),
                    "long_range_rel_drop_pct": float((baseline_long - hist_long) / baseline_long * 100.0) if baseline_long else 0.0,
                    "long_range_mean_distance": _safe_mean(hist_distances[long_range_mask]),
                },
                "runtime_visible": {
                    "overall_mean_similarity": runtime_overall,
                    "overall_abs_drop": float(baseline_overall - runtime_overall),
                    "overall_rel_drop_pct": float((baseline_overall - runtime_overall) / baseline_overall * 100.0) if baseline_overall else 0.0,
                    "overall_mean_distance": _safe_mean(runtime_distances),
                    "long_range_mean_similarity": runtime_long,
                    "long_range_abs_drop": float(baseline_long - runtime_long),
                    "long_range_rel_drop_pct": float((baseline_long - runtime_long) / baseline_long * 100.0) if baseline_long else 0.0,
                    "long_range_mean_distance": _safe_mean(runtime_distances[long_range_mask]),
                },
            }
        )

    return summary


def plot_cluster_similarity(summary, output_dir):
    alphas = [item["stride_alpha"] for item in summary["alphas"]]
    x = np.arange(len(alphas))
    labels = [f"{alpha:g}" for alpha in alphas]

    baseline_overall = summary["baseline"]["overall_mean_similarity"]
    baseline_long = summary["baseline"]["long_range_mean_similarity"]

    history_overall = [item["history_only"]["overall_mean_similarity"] for item in summary["alphas"]]
    history_long = [item["history_only"]["long_range_mean_similarity"] for item in summary["alphas"]]
    runtime_overall = [item["runtime_visible"]["overall_mean_similarity"] for item in summary["alphas"]]
    runtime_long = [item["runtime_visible"]["long_range_mean_similarity"] for item in summary["alphas"]]

    plt.figure(figsize=(4.8, 2.8))
    plt.axhline(baseline_overall, color=COLOR_NEUTRAL, linestyle="--", linewidth=1.3, label="Full History (Overall)")
    plt.axhline(baseline_long, color=COLOR_BLACK, linestyle=":", linewidth=1.3, label="Full History (Long-Range)")
    plt.plot(x, history_overall, marker="o", linewidth=1.5, color=COLOR_PRIMARY, label="Cluster History Only")
    plt.plot(x, history_long, marker="o", linewidth=1.5, color=COLOR_SECONDARY, label="Cluster History Only (Long-Range)")
    plt.plot(x, runtime_overall, marker="s", linewidth=1.3, color=COLOR_PRIMARY_LIGHT, label="Cluster Runtime-Visible")
    plt.plot(x, runtime_long, marker="s", linewidth=1.3, color=COLOR_SECONDARY_LIGHT, label="Cluster Runtime-Visible (Long-Range)")
    plt.xticks(x, labels)
    plt.xlabel("stride_alpha")
    plt.ylabel("Mean Max Similarity")
    plt.grid(True, linestyle="--", alpha=0.6, color=COLOR_GRID)
    plt.legend(fontsize=6.5, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "exp2_cluster_similarity_vs_alpha.pdf"), bbox_inches="tight", pad_inches=0.07)
    print(f"Saved exp2_cluster_similarity_vs_alpha.pdf to {output_dir}")
    plt.close()


def build_model_load_kwargs(device):
    if isinstance(device, str):
        device = device.strip()
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if device == "cpu":
        return {"device_map": {"": "cpu"}, "torch_dtype": torch.float32}, "cpu"
    if device.startswith("cuda"):
        return {"device_map": {"": device}, "torch_dtype": torch.float16}, device
    if device == "auto":
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        return {"device_map": "auto", "torch_dtype": dtype}, None
    raise ValueError(f"Unsupported device specification: {device}")

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
    data_path=None,
    seq_len=2048,
    sample_num=20,
    device=None,
    target_layers=None,
    sink_size=16,
    cluster_ratio=0.1,
    alpha_values="0.0,0.0001,0.001,0.01,0.05,0.1",
    long_range_threshold=16,
    query_block_size=1024,
):
    seq_len = int(seq_len)
    sample_num = int(sample_num)
    sink_size = int(sink_size)
    cluster_ratio = float(cluster_ratio)
    long_range_threshold = int(long_range_threshold)
    query_block_size = int(query_block_size)
    model_kwargs, resolved_device = build_model_load_kwargs(device)
    alpha_values = parse_alpha_values(alpha_values)

    print(f"Loading model from {model_path}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, 
            trust_remote_code=True,
            **model_kwargs,
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model_device = resolved_device or str(next(model.parameters()).device)
    target_layers = parse_target_layers(target_layers, model.config.num_hidden_layers)
    model = patch_model(model, target_layers)
    model.eval()
    print(f"Using target layers: {target_layers}")

    print("Preparing data...")
    data_path = resolve_data_path(data_path)
    print(f"Loading samples from {data_path}")
    # Load multiple samples
    real_data = load_data(data_path, sample_num=sample_num)
    
    # Stats containers (aggregated across all samples)
    all_max_sims = []
    all_distances = []
    cluster_history_sims = {alpha: [] for alpha in alpha_values}
    cluster_history_distances = {alpha: [] for alpha in alpha_values}
    cluster_runtime_sims = {alpha: [] for alpha in alpha_values}
    cluster_runtime_distances = {alpha: [] for alpha in alpha_values}
    center_fractions = {alpha: [] for alpha in alpha_values}
    
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
        
        inputs = tokenizer(input_text, return_tensors="pt", max_length=seq_len, truncation=True).to(model_device)
        
        with torch.no_grad():
            model(**inputs)
            
        # Analyze layers for this sample
        for layer_idx in target_layers:
            if layer_idx not in collector.k_states:
                continue
                
            # Concatenate collected states (if multiple chunks in one forward, usually just 1 for prefill)
            # k_state shape: [1, seq_len, hidden_dim] (assuming bs=1)
            k_tensor = torch.cat(collector.k_states[layer_idx], dim=1).squeeze(0).float().to(model_device)
            
            # Normalize for Cosine Similarity
            k_norm = F.normalize(k_tensor, p=2, dim=-1)
            
            seq_len_actual = k_norm.shape[0]
            
            if seq_len_actual <= sink_size:
                continue

            # For each token t > sink_size, compute max_{j < t} sim(t, j)
            valid_indices = torch.arange(sink_size, seq_len_actual)
            top_vals, top_indices = compute_best_similarity(
                k_norm,
                sink_size,
                allow_self=False,
                query_block_size=query_block_size,
            )
            
            # Calculate distances: current_pos - best_match_pos
            distances = valid_indices.to(top_indices.device) - top_indices
            
            all_max_sims.extend(top_vals.cpu().tolist())
            all_distances.extend(distances.cpu().tolist())

            for alpha in alpha_values:
                center_idx = build_center_indices(
                    seq_len=seq_len_actual,
                    sink_size=sink_size,
                    cluster_ratio=cluster_ratio,
                    stride_alpha=alpha,
                )
                center_fractions[alpha].append(float(center_idx.numel()) / float(seq_len_actual))

                cluster_hist_vals, cluster_hist_idx = compute_best_similarity(
                    k_norm,
                    sink_size,
                    candidate_indices=center_idx,
                    allow_self=False,
                    query_block_size=query_block_size,
                )
                cluster_runtime_vals, cluster_runtime_idx = compute_best_similarity(
                    k_norm,
                    sink_size,
                    candidate_indices=center_idx,
                    allow_self=True,
                    query_block_size=query_block_size,
                )

                cluster_history_sims[alpha].extend(cluster_hist_vals.cpu().tolist())
                cluster_history_distances[alpha].extend(
                    (valid_indices.to(cluster_hist_idx.device) - cluster_hist_idx).cpu().tolist()
                )
                cluster_runtime_sims[alpha].extend(cluster_runtime_vals.cpu().tolist())
                cluster_runtime_distances[alpha].extend(
                    (valid_indices.to(cluster_runtime_idx.device) - cluster_runtime_idx).cpu().tolist()
                )
            
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

    summary = summarize_cluster_similarity(
        baseline_sims=all_max_sims,
        baseline_distances=all_distances,
        cluster_history_sims=cluster_history_sims,
        cluster_history_distances=cluster_history_distances,
        cluster_runtime_sims=cluster_runtime_sims,
        cluster_runtime_distances=cluster_runtime_distances,
        center_fractions=center_fractions,
        alpha_values=alpha_values,
        long_range_threshold=long_range_threshold,
    )

    summary["config"] = {
        "model_path": model_path,
        "data_path": data_path,
        "seq_len": int(seq_len),
        "sample_num": int(sample_num),
        "target_layers": [int(x) for x in target_layers],
        "sink_size": int(sink_size),
        "cluster_ratio": float(cluster_ratio),
        "alpha_values": [float(x) for x in alpha_values],
        "query_block_size": int(query_block_size),
        "device": model_device,
    }

    summary_path = os.path.join(output_dir, "exp2_cluster_similarity_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Saved exp2_cluster_similarity_summary.json to {output_dir}")

    print("Cluster similarity summary:")
    for item in summary["alphas"]:
        alpha = item["stride_alpha"]
        hist_long = item["history_only"]["long_range_mean_similarity"]
        hist_drop = item["history_only"]["long_range_abs_drop"]
        hist_drop_pct = item["history_only"]["long_range_rel_drop_pct"]
        run_long = item["runtime_visible"]["long_range_mean_similarity"]
        run_drop = item["runtime_visible"]["long_range_abs_drop"]
        run_drop_pct = item["runtime_visible"]["long_range_rel_drop_pct"]
        center_frac = item["avg_center_fraction"]
        print(
            f"[alpha={alpha:g}] centers={center_frac:.4f} "
            f"history_long={hist_long:.4f} (drop={hist_drop:.4f}, {hist_drop_pct:.2f}%) "
            f"runtime_long={run_long:.4f} (drop={run_drop:.4f}, {run_drop_pct:.2f}%)"
        )

    plot_cluster_similarity(summary, output_dir)

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
