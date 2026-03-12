import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
import os

# Set global font to sans-serif for a clean, modern look
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']

# Labels and Data
labels_legend = ['Sparse-vLLM (SnapKV)', 'Sparse-vLLM (DeltaKV)', 'Sparse-vLLM (OmniKV)', 'Sparse-vLLM (Full Attn)', 'vLLM (Baseline)']
short_labels = ['Snap', 'Delta', 'Omni', 'Full', 'vLLM']

# Data from Table 4
data_128k = [338.8, 187.0, 216.7, 135.0, 143.2]
data_256k = [168.8, 120.6, 115.9, 69.5, 70.2]
data_512k = [0, 67.7, 0, 32.1, 33.1]

# Brighter colors for all Sparse-vLLM methods, grey for vLLM baseline
colors = ['#3b82f6', '#10b981', '#a855f7', '#f59e0b', '#d1d5db']
text_colors = ['#2563eb', '#059669', '#9333ea', '#d97706', '#9ca3af']

fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor='white')
fig.subplots_adjust(top=0.75, bottom=0.1, wspace=0.15)

datasets = [data_128k, data_256k, data_512k]
titles = ['128K Context', '256K Context', '512K Context']

def draw_rounded_bar(ax, center_x, height, width, color):
    if height <= 0: return
    
    rx = width * 0.35
    x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
    y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
    
    # Estimate ry to make corners visually circular based on data aspect ratio
    ry = (rx / x_range) * y_range
    if height < ry: ry = height
    
    left = center_x - width/2
    right = center_x + width/2
    
    Path = mpath.Path
    kappa = 0.55228
    
    verts = [
        (left, 0),
        (right, 0),
        (right, height - ry),
        (right, height - ry + ry * kappa),
        (right - rx + rx * kappa, height),
        (right - rx, height),
        (left + rx, height),
        (left + rx - rx * kappa, height),
        (left, height - ry + ry * kappa),
        (left, height - ry),
        (left, 0),
    ]
    
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.LINETO,
        Path.CURVE4,
        Path.CURVE4,
        Path.CURVE4,
        Path.CLOSEPOLY,
    ]
    
    path = mpath.Path(verts, codes)
    patch = mpatches.PathPatch(path, facecolor=color, edgecolor='none', zorder=2)
    ax.add_patch(patch)

for ax, data, title in zip(axes, datasets, titles):
    ax.set_facecolor('white')
    
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    x = np.arange(len(data))
    width = 0.75
    
    max_val = max(data) if max(data) > 0 else 100
    ax.set_ylim(-max_val * 0.15, max_val * 1.35)
    ax.set_xlim(-0.6, len(data)-0.4)
    
    # Baseline
    ax.plot([-0.5, len(data)-0.5], [0, 0], color='#e5e7eb', linewidth=2, zorder=1)
    
    for j, (val, color, short_lab, text_c) in enumerate(zip(data, colors, short_labels, text_colors)):
        if val > 0:
            draw_rounded_bar(ax, x[j], val, width, color)
            
            # Main value text above the bar
            ax.text(x[j], val + (max_val*0.02), f'{val:.1f}', 
                    ha='center', va='bottom', fontsize=15, fontweight='bold', color='#1f2937')
            
            # Speedup multiplier annotation
            vllm_val = data[-1]
            if vllm_val > 0 and j != 4:
                speedup = val / vllm_val
                # Highlight "vLLM" suffix for the maximum throughput OR for DeltaKV (the paper's core method)
                text_str = f'{speedup:.1f}x'
                    
                ax.text(x[j], val + (max_val*0.11), text_str, 
                        ha='center', va='bottom', fontsize=11, fontweight='bold', color=text_c)
            
            # Abbreviation inside the bar near the top
            txt_color = 'white' if j < 4 else '#4b5563'
            text_y = val - (max_val * 0.07)
            
            if text_y > max_val * 0.05:
                ax.text(x[j], text_y, short_lab, 
                        ha='center', va='center', fontsize=12, fontweight='bold', color=txt_color)
            else:
                ax.text(x[j], val/2, short_lab, 
                        ha='center', va='center', fontsize=11, fontweight='bold', color=txt_color)

    # Category title at the bottom
    ax.text(np.mean(x), -max_val*0.08, title, ha='center', va='top', fontsize=15, fontweight='bold', color='#4b5563')

# Global Legend at the top
handles = [mpatches.Rectangle((0,0),1,1, color=c, lw=0) for c in colors]
legend = fig.legend(handles, labels_legend, loc='upper center', bbox_to_anchor=(0.5, 0.96), 
                    ncol=5, frameon=False, fontsize=13, handlelength=1.2, handleheight=1.2,
                    columnspacing=1.5, handletextpad=0.5)

for text in legend.get_texts():
    text.set_color('#374151')
    text.set_fontweight('bold')

fig.suptitle('Sparse-vLLM Decoding Throughput (tokens/s)', fontsize=22, fontweight='bold', color='#1f2937', y=1.08)

os.makedirs('assets', exist_ok=True)
save_path = 'assets/sparse_vllm_throughput.png'
plt.savefig(save_path, dpi=300, bbox_inches='tight')
print(f"Chart successfully saved to {save_path}!")
