"""
Figure 9 — Multi-node scaling efficiency vs GPU count.
Data source: nsys NVTX step median, rank 0. O96, compiled BF16, batch 8/GPU.
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
    'legend.fontsize': 11,
})

nodes      = [1,   2,    10,   25,   50,   100]
gpus       = [4,   8,    40,   100,  200,  400]
efficiency = [96.1, 94.2, 94.6, 90.8, 84.6, 85.6]   # % (1-GPU = 100%)
labels     = ['4\n(1 node)', '8\n(2 nodes)', '40\n(10 nodes)',
               '100\n(25 nodes)', '200\n(50 nodes)', '400\n(100 nodes)']

fig, ax = plt.subplots(figsize=(10, 6))

# Perfect scaling reference
ax.axhline(100, color='#9E9E9E', linestyle='--', linewidth=1.2, alpha=0.7, label='Perfect scaling (100%)')

# NCCL algorithm switch annotation band
ax.axvspan(3.5, 4.5, alpha=0.08, color='#E53935')   # between 25-node and 50-node index

ax.plot(range(len(labels)), efficiency, color='#1565C0', marker='o', linewidth=2.5,
        markersize=9, label='Observed scaling efficiency')

# Colour-code the points by region
for i, (eff, lbl) in enumerate(zip(efficiency, labels)):
    color = '#43A047' if eff >= 93 else ('#FB8C00' if eff >= 88 else '#E53935')
    ax.plot(i, eff, 'o', color=color, markersize=11, zorder=5)
    ax.annotate(f'{eff}%', (i, eff),
                textcoords='offset points', xytext=(0, 10),
                ha='center', fontsize=11, fontweight='bold')

# Annotate the NCCL switch
ax.annotate('NCCL switches\nRING_LL → TREE_LL',
            xy=(4, 84.6), xytext=(4.2, 79),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.5),
            fontsize=10, color='#C62828', ha='left')

ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels)
ax.set_xlabel('GPU count (node count)')
ax.set_ylabel('Scaling efficiency (%)')
ax.set_title('Multi-Node Scaling Efficiency — O96, Compiled BF16\n(relative to 1-GPU step time: 977 ms)',
             fontweight='bold')
ax.set_ylim(70, 107)

legend_handles = [
    plt.Line2D([0], [0], color='#1565C0', linewidth=2.5, marker='o', markersize=8,
               label='Scaling efficiency'),
    plt.Line2D([0], [0], color='#9E9E9E', linestyle='--', linewidth=1.5,
               label='Perfect scaling (100%)'),
    mpatches.Patch(color='#E53935', alpha=0.15, label='RING→TREE transition region'),
]
ax.legend(handles=legend_handles, loc='lower left')

fig.tight_layout()
out = '../plots/4.1_scaling_efficiency.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
