import matplotlib.pyplot as plt
import numpy as np

# Data setup
num_steps = 767
bs_list = [1, 16]

# --- Conservative Estimation Factors ---
RECON_OPT_FACTOR = 0.5  # 2x speedup
VIEW_OPT_FACTOR = 0.4   # 2.5x speedup

# --- BS = 1 Data (from new Profiling) ---
total_bs1 = 42.6
recon_bs1 = 10.9
view_bs1 = 2.9
forward_bs1 = total_bs1 - recon_bs1 - view_bs1

# BS = 1 Optimized Estimation
recon_bs1_opt = recon_bs1 * RECON_OPT_FACTOR
view_bs1_opt = view_bs1 * VIEW_OPT_FACTOR
forward_bs1_opt = forward_bs1
total_bs1_opt = forward_bs1_opt + recon_bs1_opt + view_bs1_opt

# --- BS = 16 Data (from old Profiling) ---
forward_bs16 = 29.0
recon_bs16 = 37.3
view_bs16 = 24.7
total_bs16 = forward_bs16 + recon_bs16 + view_bs16

# BS = 16 Optimized Estimation
recon_bs16_opt = recon_bs16 * RECON_OPT_FACTOR
view_bs16_opt = view_bs16 * VIEW_OPT_FACTOR
forward_bs16_opt = forward_bs16
total_bs16_opt = forward_bs16_opt + recon_bs16_opt + view_bs16_opt

# Organizing data
categories = ['Batch Size = 1', 'Batch Size = 16']
forward_data = [forward_bs1, forward_bs1_opt, forward_bs16, forward_bs16_opt]
recon_data = [recon_bs1, recon_bs1_opt, recon_bs16, recon_bs16_opt]
view_data = [view_bs1, view_bs1_opt, view_bs16, view_bs16_opt]

# x positions
x = np.array([0, 1, 3, 4])
tick_labels = ['Current', 'Optimized (Est.)', 'Current', 'Optimized (Est.)']

fig, ax = plt.subplots(figsize=(8, 5))
width = 0.7

# Stacked Bar Chart
ax.bar(x, forward_data, width, label='Core Forward (Attn/MLP)', color='#4C72B0', edgecolor='white')
ax.bar(x, recon_data, width, bottom=forward_data, label='KV Reconstruction (DeltaKV)', color='#DD8452', edgecolor='white')
ax.bar(x, view_data, width, bottom=np.array(forward_data) + np.array(recon_data), label='View & Slot Management', color='#55A868', edgecolor='white')

# Totals and Throughput labels on top
totals = [total_bs1, total_bs1_opt, total_bs16, total_bs16_opt]
real_bs = [1, 1, 16, 16]

for i, val in enumerate(x):
    total = totals[i]
    tps = (real_bs[i] * 1000) / total
    ax.text(val, total + 1.5, f'{total:.1f} ms\n({tps:.1f} t/s)', ha='center', va='bottom', fontsize=10, fontweight='bold')

# X-axis formatting
ax.set_xticks(x)
ax.set_xticklabels(tick_labels, fontsize=11)

# Group labels below the axis
ax.text(0.5, -18, 'Batch Size = 1', ha='center', va='top', fontsize=14, fontweight='bold')
ax.text(3.5, -18, 'Batch Size = 16', ha='center', va='top', fontsize=14, fontweight='bold')

# BS=1 speedup: Box moved UP to avoid overlapping with bars
ax.annotate('', xy=(0.5, total_bs1_opt), xytext=(0.5, total_bs1),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(0.8, total_bs1 + 8, f'{total_bs1/total_bs1_opt:.1f}x Speedup',
        fontsize=12, color='red', fontweight='bold', ha='center', va='bottom',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5))

# BS=16 speedup: Restore original position (between the bars)
ax.annotate('', xy=(3.5, total_bs16_opt), xytext=(3.5, total_bs16),
            arrowprops=dict(arrowstyle='<->', color='red', lw=2))
ax.text(3.5 + 0.1, (total_bs16 + total_bs16_opt)/2, f'{total_bs16/total_bs16_opt:.1f}x Speedup', 
        fontsize=12, color='red', fontweight='bold', ha='left', va='center',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", lw=1.5))

# Decoration
ax.set_ylabel('Step Latency (ms)', fontsize=12)
ax.legend(loc='upper left', frameon=True, shadow=False, fontsize=10)
ax.grid(axis='y', linestyle='--', alpha=0.4)
ax.set_ylim(0, 135)

plt.tight_layout()
plt.subplots_adjust(bottom=0.2) 
plt.savefig('assets/deltakv_latency_comparison_bs1_vs_bs16.pdf')
print("Updated comparison chart saved to deltakv_latency_comparison_bs1_vs_bs16.pdf")
