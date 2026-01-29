# -*- coding: utf-8 -*-
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os
from deltakv.analysis.colors import COLOR_PRIMARY, COLOR_SECONDARY, COLOR_TERTIARY, COLOR_GRID

def visualize_detach_ablation():
    # --- 配置 ---
    PROJECT = "DeltaKV"
    # MSE_DETACH: 7dxp04k4 (Created first)
    # NTP_DETACH: c8f7m4zl (Created second)
    # BASELINE: 8ax9x47z (Train Both)
    RUNS_CONFIG = {
        "7dxp04k4": {"label": "Train NTP Only", "color": COLOR_PRIMARY},
        "c8f7m4zl": {"label": "Train MSE Only", "color": COLOR_SECONDARY},
        "8ax9x47z": {"label": "Train Both", "color": COLOR_TERTIARY}
    }
    METRICS = ["train/loss", "train/ntp_loss", "train/mse_loss"]
    TITLES = ["Total Loss $\downarrow$", "NTP Loss $\downarrow$", "MSE Loss $\downarrow$"]
    OUTPUT_DIR = "/root/autodl-fs/visuals/"
    WINDOW_SIZE = 100
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    api = wandb.Api()
    
    # 获取数据
    plot_data = {}
    for run_id, info in RUNS_CONFIG.items():
        print(f"Fetching data for run {run_id} ({info['label']})...")
        run = api.run(f"{PROJECT}/{run_id}")
        # 获取完整历史记录
        history = run.history(keys=METRICS + ["_step"], samples=20000)
        if history.empty:
            print(f"Warning: Run {run_id} has no history.")
            continue
        
        # 排序
        history = history.sort_values("_step")
        
        # 为每个指标单独处理（去除NaN以保证平滑连续）
        run_metrics_data = {}
        for metric in METRICS:
            if metric in history.columns:
                # 提取非空数据，并处理可能存在的重复 step
                metric_df = history[["_step", metric]].dropna()
                if not metric_df.empty:
                    # 如果同一个 step 有多个记录，取平均值
                    metric_df = metric_df.groupby("_step")[metric].mean().reset_index()
                    # 使用 EMA (指数移动平均) 进行平滑
                    metric_df[f"{metric}_smooth"] = metric_df[metric].ewm(span=WINDOW_SIZE, adjust=False).mean()
                    run_metrics_data[metric] = metric_df
        
        plot_data[run_id] = run_metrics_data

    # --- 绘图 ---
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, axes = plt.subplots(1, 3, figsize=(10, 1.9))
    
    for i, metric in enumerate(METRICS):
        ax = axes[i]
        for run_id, info in RUNS_CONFIG.items():
            if run_id not in plot_data or metric not in plot_data[run_id]:
                continue
            
            df = plot_data[run_id][metric]
            smooth_col = f"{metric}_smooth"
            ax.plot(df["_step"], df[smooth_col], label=info["label"], color=info["color"], linewidth=1.5)

        ax.set_xlabel("Steps")
        ax.set_ylabel(TITLES[i])
        if metric == 'train/ntp_loss':
            ax.set_ylim(top=3.5)
        ax.grid(True, linestyle='--', alpha=0.7, color=COLOR_GRID)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.10), ncol=3, frameon=True)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "detach_ablation_comparison.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Successfully saved ablation visualization to: {save_path}")

if __name__ == "__main__":
    visualize_detach_ablation()
