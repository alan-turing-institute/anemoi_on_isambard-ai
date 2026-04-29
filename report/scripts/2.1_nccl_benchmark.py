"""
Figure 5 — NCCL All-Reduce benchmark bandwidth vs node count.
Data source: nccl-tests all_reduce_perf, Isambard-AI.
Theoretical ceilings:
  NVLink 4.0: 450 GB/s unidirectional per GH200 GPU
  Slingshot-11: 100 GB/s per node (4 NICs × 25 GB/s)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

nodes      = [1,     10,   50,   200]
gpus       = [4,     40,   200,  800]
bandwidth  = [342.5, 92.7, 91.2, 70.8]
labels     = ['1 node\n(4 GPUs)', '10 nodes\n(40 GPUs)',
               '50 nodes\n(200 GPUs)', '200 nodes\n(800 GPUs)']
colors     = ['#2196F3', '#FF9800', '#FF9800', '#E53935']  # blue = NVLink, orange = Slingshot stable, red = drop

NVLINK_PEAK     = 450   # GB/s — NVLink 4.0 unidirectional per GH200 GPU
SLINGSHOT_PEAK  = 100   # GB/s — Slingshot-11: 4 NICs × 25 GB/s per node

fig, ax = plt.subplots(figsize=(9, 6))

bars = ax.bar(labels, bandwidth, color=colors, width=0.55, edgecolor='white', linewidth=1.2)

for bar, bw in zip(bars, bandwidth):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 6,
            f'{bw} GB/s', ha='center', va='bottom', fontsize=12, fontweight='bold')

# Theoretical ceilings
ax.axhline(NVLINK_PEAK, color='#2196F3', linestyle='--', linewidth=1.4, alpha=0.7)
ax.text(-0.45, NVLINK_PEAK + 8, f'NVLink 4.0 theoretical peak ({NVLINK_PEAK} GB/s)',
        ha='left', va='bottom', fontsize=10, color='#1565C0')

ax.axhline(SLINGSHOT_PEAK, color='#FF9800', linestyle='--', linewidth=1.4, alpha=0.7)
ax.text(3.45, SLINGSHOT_PEAK + 6, f'Slingshot-11 theoretical peak ({SLINGSHOT_PEAK} GB/s)',
        ha='right', va='bottom', fontsize=10, color='#E65100')

ax.set_ylabel('Peak Bus Bandwidth (GB/s)')
ax.set_title('NCCL All-Reduce Benchmark — Peak Bus Bandwidth', fontweight='bold')
ax.set_ylim(0, 520)

nvlink_patch    = mpatches.Patch(color='#2196F3', label='NVLink (intra-node)')
slingshot_patch = mpatches.Patch(color='#FF9800', label='Slingshot (stable)')
drop_patch      = mpatches.Patch(color='#E53935', label='Slingshot (degraded)')
ax.legend(handles=[nvlink_patch, slingshot_patch, drop_patch], loc='upper right')

fig.tight_layout()
out = '../plots/2.1_nccl_benchmark.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
