"""
Figure 11 — NCCL AllReduce saturation: RING_LL + TREE_LL kernel time vs backward window.
Data source: nsys stats GPU kernel summary (ncclDevKernel_AllReduce_Sum_f32_*),
             nsys NVTX backward wall time. O96, compiled BF16, rank 0.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

cases            = ['1 node', '2 nodes', '10 nodes', '25 nodes', '50 nodes', '100 nodes']
ring_ll          = [42.6,  129.2, 287.8, 317.3,   5.5,  15.1]
tree_ll          = [ 0.0,   16.4,  41.7,  59.8, 615.2, 503.8]
backward_window  = [734.9, 744.2, 737.2, 764.9, 748.2, 738.4]
saturation       = [  6,     20,    45,    49,    83,    70]   # %

x = np.arange(len(cases))
width = 0.38

fig, ax1 = plt.subplots(figsize=(11, 6))

bar_ring = ax1.bar(x - width/2, ring_ll, width, label='RING_LL AllReduce', color='#1976D2', alpha=0.9)
bar_tree = ax1.bar(x - width/2, tree_ll, width, bottom=ring_ll, label='TREE_LL AllReduce',
                   color='#E53935', alpha=0.9)

# Backward window as a separate bar
bar_bwd  = ax1.bar(x + width/2, backward_window, width, label='Backward NVTX window',
                   color='#78909C', alpha=0.5)

ax1.set_ylabel('Time per step (ms)')
ax1.set_xticks(x)
ax1.set_xticklabels(cases)
ax1.set_ylim(0, 950)
ax1.set_title('NCCL AllReduce Kernel Time vs Backward Window\n(f32 AllReduce kernels, rank 0)',
              fontweight='bold')

# Saturation % on secondary axis
ax2 = ax1.twinx()
ax2.plot(x, saturation, color='#F57F17', marker='D', linewidth=2.2,
         markersize=8, label='AllReduce saturation (%)', zorder=5)
ax2.set_ylabel('AllReduce saturation (% of backward window)', color='#F57F17')
ax2.tick_params(axis='y', labelcolor='#F57F17')
ax2.set_ylim(0, 120)

for xi, sat in zip(x, saturation):
    ax2.annotate(f'{sat}%', (xi, sat),
                 textcoords='offset points', xytext=(0, 8),
                 ha='center', fontsize=11, fontweight='bold', color='#E65100')

# Annotate the algorithm switch
ax1.axvline(3.5, color='#C62828', linestyle='--', linewidth=1.5, alpha=0.6)
ax1.text(3.55, 870, 'RING_LL → TREE_LL\nalgorithm switch',
         fontsize=10, color='#C62828', va='top')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', framealpha=0.9)

fig.tight_layout()
out = '../plots/4.3_nccl_allreduce.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
