# -*- coding: utf-8 -*-
import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os
import json
import argparse
import subprocess
import sys
from tqdm import tqdm
from deltakv.analysis.colors import COLOR_PRIMARY, COLOR_SECONDARY, COLOR_GRID, COLOR_BLACK, COLOR_TERTIARY

# --- 配置 ---
PROJECT = "DeltaKV"
GROUP = "llama_hyperparams_ablation_v1"
OUTPUT_JSON = "/root/autodl-fs/deltakv_outputs/ablation_results.json"
VISUALS_DIR = "/root/autodl-fs/visuals/"
BENCH_LOG = "/root/autodl-fs/deltakv_outputs/ablation_decode_tp.log"
METRIC = "train/ntp_loss"
ORIGINAL_BASELINE = 2.335

# 默认测试模型和参数
DEFAULT_MODEL = "/root/autodl-fs/models/Llama-3.1-8B-Instruct-Qwen-ver"
DEFAULT_LENGTH = 120000
DEFAULT_BS = 8

DEFAULTS = {
    'kv_size': 512,
    'seq_chunk': 4,
    'inter_size': 4096,
    'cluster_ratio': 0.1
}

def export_data():
    """步骤 1：从 WandB 获取训练 Loss 和最终模型路径并存入 JSON"""
    api = wandb.Api()
    try:
        # 增加时长过滤条件：_runtime > 3600 秒
        runs = api.runs(f"{PROJECT}", filters={
            "group": GROUP,
            "summary_metrics._runtime": {"$gt": 3600}
        })
    except Exception as e:
        print(f"获取 WandB 数据失败: {e}")
        return

    run_data = []
    print(f"正在从 WandB 下载 Group: {GROUP} 且时长 > 1h 的数据...")
    
    for run in tqdm(runs):
        # 再次确认时长
        if run.summary.get("_runtime", 0) < 3600:
            continue

        config = run.config
        # 获取历史指标数据
        history = run.history(keys=[METRIC], pandas=True)
        if history.empty or METRIC not in history.columns:
            continue
            
        # 计算最后 200 个 step 的平均 Loss
        final_loss = history[METRIC].tail(200).mean()
        
        # 获取模型路径：优先使用 final_output_dir，那是训练结束时保存的完整模型
        deltakv_path = config.get("final_output_dir")
        if not deltakv_path:
            deltakv_path = config.get("deltakv_path")

        run_data.append({
            'kv_size': config.get("kv_compressed_size"),
            'seq_chunk': config.get("seq_chunk_size"),
            'inter_size': config.get("compressor_intermediate_size"),
            'cluster_ratio': config.get("cluster_ratio"),
            'deltakv_path': deltakv_path,
            'final_loss': float(final_loss),
            'throughput': None
        })

    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    with open(OUTPUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(run_data, f, indent=4, ensure_ascii=False)
    print(f"数据已导出至: {OUTPUT_JSON}")

def run_benchmarks():
    """步骤 2：自动化运行吞吐量测试，精确传递对应消融参数"""
    if not os.path.exists(OUTPUT_JSON):
        print(f"错误: 找不到文件 {OUTPUT_JSON}")
        return

    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
        run_data = json.load(f)
    
    df = pd.DataFrame(run_data)
    
    # 准备日志文件
    os.makedirs(os.path.dirname(BENCH_LOG), exist_ok=True)
    with open(BENCH_LOG, "w") as f:
        f.write("=== Ablation Throughput Benchmark Log ===\n")

    # 同步代码
    print("正在执行 git pull...")
    subprocess.run(["git", "pull"], check=False)

    # 确定消融任务列表
    ablation_params = ['kv_size', 'seq_chunk', 'inter_size', 'cluster_ratio']
    tasks = []
    # 使用 set 记录已经加入的任务 deltakv_path，避免重复测试 (比如 baseline 配置在每个维度都会出现)
    seen_paths = set()
    for col in ablation_params:
        mask = pd.Series(True, index=df.index)
        for other_col, def_val in DEFAULTS.items():
            if other_col != col:
                mask &= (df[other_col] == def_val)
        
        # 仅选择 throughput 为空的数据
        mask &= df['throughput'].isnull()
        
        targets = df[mask].sort_values(col)
        for _, row in targets.iterrows():
            if row['deltakv_path'] and row['deltakv_path'] not in seen_paths:
                tasks.append((col, row))
                seen_paths.add(row['deltakv_path'])

    if not tasks:
        print("所有配置均已有吞吐量数据，无需进一步测试。")
        return

    print(f"共识别到 {len(tasks)} 个待测试消融配置。详细日志: {BENCH_LOG}")

    # 使用 tqdm 显示测试进度，不重定向 tqdm 本身到日志
    for col, row in tqdm(tasks, desc="Benchmark Progress", unit="config", file=sys.stdout):
        # 构造 hyper_params 参数，确保所有消融参数与训练时一致
        hp = {
            "gpu_memory_utilization": 0.9,
            "kv_compressed_size": int(row['kv_size']),
            "deltakv_k_neighbors": int(row['seq_chunk']),  # 推理引擎中的 k_neighbors 对应训练时的 seq_chunk_size
            "compressor_intermediate_size": int(row['inter_size']),
            "cluster_ratio": float(row['cluster_ratio']),
            "num_top_tokens": 2048,
            "deltakv_path": row['deltakv_path'],
            "full_attn_layers": "0,1,2,8,18",
            "deltakv_full_pool_reserve_ratio": 0.1,
            "max_num_batched_tokens": DEFAULT_BS*4096,
            "chunk_prefill_size": 4096,
            "max_num_seqs_in_batch": DEFAULT_BS,
        }
        
        cmd = [
            "python", "scripts/bench_sparse_vllm.py",
            "--model_path", DEFAULT_MODEL,
            "--lengths", str(DEFAULT_LENGTH),
            "--batch_sizes", str(DEFAULT_BS),
            "--methods", "deltakv-triton-v4",
            "--output_len", "768",
            "--hyper_params", json.dumps(hp)
        ]
        
        with open(BENCH_LOG, "a") as log_f:
            log_f.write(f"\n\n" + "="*60 + f"\n>>> Testing {col}={row[col]}\n")
            log_f.write(f"Command: {' '.join(cmd)}\n" + "-"*60 + "\n")
            log_f.flush()
            
            # 使用 Popen 将子进程输出导向日志文件，主进程保持 tqdm 更新
            # 设置 CUDA_LAUNCH_BLOCKING=1 以便同步测试
            env = os.environ.copy()
            # env["CUDA_LAUNCH_BLOCKING"] = "1"
            process = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, env=env)
            process.wait()

    print(f"\n所有测试已完成。请查看日志: {BENCH_LOG}")
    print(f"请手动将解析出的 'Decode: XXX.XX tok/s' 填回 {OUTPUT_JSON}，然后运行 --step 3 绘图。")

def plot_data():
    """步骤 3：读取 JSON 数据，绘制 1x4 合并大图"""
    if not os.path.exists(OUTPUT_JSON):
        print(f"错误: 找不到文件 {OUTPUT_JSON}")
        return

    with open(OUTPUT_JSON, 'r', encoding='utf-8') as f:
        run_data = json.load(f)
    
    df = pd.DataFrame(run_data)
    os.makedirs(VISUALS_DIR, exist_ok=True)
    
    ablation_params = [
        ('kv_size', 'Compressed KV Size'),
        ('seq_chunk', 'Num of Ref Tokens'),
        ('inter_size', 'Compressor Inter Size'),
        ('cluster_ratio', 'Reference Stride')
    ]

    plt.style.use('seaborn-v0_8-whitegrid')
    # 使用 sharey=True 共享左轴
    fig, axes = plt.subplots(1, 4, figsize=(12, 2.5), sharey=True)

    # 找到基准吞吐量 (所有参数均为默认值)
    baseline_mask = pd.Series(True, index=df.index)
    for k, v in DEFAULTS.items():
        baseline_mask &= (df[k] == v)
    baseline_rows = df[baseline_mask & df['throughput'].notnull()]
    baseline_tp = baseline_rows['throughput'].mean() if not baseline_rows.empty else None

    # 存放全局图例的句柄和标签
    all_handles = []
    all_labels = []
    
    # 预先创建一个用于共享的右轴对象
    shared_ax2 = None

    for i, (col, title) in enumerate(ablation_params):
        ax1 = axes[i]
        mask = pd.Series(True, index=df.index)
        for other_col, def_val_loop in DEFAULTS.items():
            if other_col != col:
                mask &= (df[other_col] == def_val_loop)
        
        plot_df = df[mask].copy()
        if col == 'cluster_ratio':
            plot_df[col] = 1.0 / plot_df[col]
            
        plot_df = plot_df.sort_values(col)
        if plot_df.empty:
            continue

        # --- 绘制左轴 (NTP Loss) ---
        lns1 = ax1.plot(plot_df[col], plot_df['final_loss'], marker='o', markersize=4, 
                        linewidth=1.2, color=COLOR_PRIMARY, label='NTP Loss', alpha=0.8)
        lns_base = ax1.axhline(y=ORIGINAL_BASELINE, color=COLOR_BLACK, linestyle='--', linewidth=1.0, label='Baseline')
        
        # 标记默认值点
        def_val = DEFAULTS[col]
        if col == 'cluster_ratio':
            def_val = 1.0 / def_val
        def_row = plot_df[plot_df[col] == def_val]
        if not def_row.empty:
            lns_def = ax1.plot(def_val, def_row['final_loss'].iloc[0], 'ro', markersize=6, label='Default', zorder=5)
            if 'Default' not in all_labels:
                all_handles.extend(lns_def)
                all_labels.append('Default')

        # 设置左轴范围
        ax1.set_ylim(2.3, 2.6)
        
        if 'NTP Loss' not in all_labels:
            all_handles.extend(lns1)
            all_handles.append(lns_base)
            all_labels.extend(['NTP Loss', 'NTP Baseline'])

        ax1.set_xlabel(title)
        if i == 0:
            ax1.set_ylabel(r"NTP Loss $\downarrow$", color=COLOR_PRIMARY)
        
        # 只有最左边的图显示左轴刻度标签
        ax1.tick_params(axis='y', labelleft=(i == 0), labelcolor=COLOR_PRIMARY)
        ax1.grid(True, linestyle='--', alpha=0.5, color=COLOR_GRID)

        # --- 绘制右轴 (Decode Throughput) ---
        ax2 = ax1.twinx()
        if shared_ax2 is None:
            shared_ax2 = ax2
        else:
            ax2.sharey(shared_ax2)

        if baseline_tp:
            # 计算相对吞吐量百分比
            plot_df['rel_tp'] = (plot_df['throughput'] / baseline_tp) * 100
            plot_df['is_oom'] = plot_df['throughput'].isnull()
            # OOM 统一放在 100% 的位置标记
            plot_df.loc[plot_df['is_oom'], 'rel_tp'] = 100.0

            # 分段绘制连线：有效点之间用红色，涉及 OOM 的用灰色
            for j in range(len(plot_df) - 1):
                p1 = plot_df.iloc[j]
                p2 = plot_df.iloc[j+1]
                is_oom_seg = p1['is_oom'] or p2['is_oom']
                seg_color = 'gray' if is_oom_seg else COLOR_SECONDARY
                seg_alpha = 0.3 if is_oom_seg else 0.6
                ax2.plot([p1[col], p2[col]], [p1['rel_tp'], p2['rel_tp']], 
                         color=seg_color, linestyle='-', linewidth=1.0, alpha=seg_alpha, zorder=1)

            valid_df = plot_df[~plot_df['is_oom']]
            oom_df = plot_df[plot_df['is_oom']]

            # 绘制有效数据点 (红色方块)
            lns_tp_pts = ax2.plot(valid_df[col], valid_df['rel_tp'], marker='s', markersize=4,
                                 linestyle='', color=COLOR_SECONDARY, label='Throughput', zorder=2)
            
            # 标记 OOM (100% 处的灰色叉号)
            if not oom_df.empty:
                lns_oom = ax2.plot(oom_df[col], oom_df['rel_tp'], marker='x', color='gray', 
                                  linestyle='', markersize=6, label='OOM', zorder=3)
                if 'OOM' not in all_labels:
                    all_handles.extend(lns_oom)
                    all_labels.append('OOM')

            if 'Throughput' not in all_labels:
                all_handles.extend(lns_tp_pts)
                all_labels.append('Throughput')

            # 只有最右边的图显示右轴标签和刻度
            if i == 3:
                ax2.set_ylabel(r"Rel. Throughput (%) $\uparrow$", color=COLOR_SECONDARY)
                ax2.tick_params(axis='y', labelright=True, labelcolor=COLOR_SECONDARY)
            else:
                ax2.tick_params(axis='y', labelright=False)
            
            ax2.grid(False)

    # 在图像上方添加统一图例
    if all_handles:
        # 按照特定顺序排序图例
        order = ['NTP Loss', 'NTP Baseline', 'Default', 'Throughput', 'OOM']
        handle_dict = dict(zip(all_labels, all_handles))
        final_handles = [handle_dict[l] for l in order if l in handle_dict]
        final_labels = [l for l in order if l in handle_dict]

        fig.legend(final_handles, final_labels, loc='upper center', 
                   bbox_to_anchor=(0.5, 0.98), ncol=5, frameon=True, fontsize='small')

    plt.tight_layout()
    # 留出空间给顶部的 legend
    plt.subplots_adjust(top=0.82)
    
    save_path = os.path.join(VISUALS_DIR, "ablation_combined.pdf")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"合并图表已保存: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ablation Analysis Pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3], required=True, 
                        help="1: Export Loss/Path from WandB, 2: Run Throughput Benchmarks, 3: Plot Combined Graph")
    args = parser.parse_args()

    if args.step == 1:
        export_data()
    elif args.step == 2:
        run_benchmarks()
    elif args.step == 3:
        plot_data()
