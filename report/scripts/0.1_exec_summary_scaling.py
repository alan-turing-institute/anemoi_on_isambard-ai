"""
Figure 0.1 — Executive summary: multi-node scaling efficiency headline.
Compact horizontal bar chart for the executive summary section.
Data source: nsys NVTX step median, rank 0. O96, compiled BF16.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 14,
    'axes.labelsize': 13,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

scales      = ['2 nodes\n(8 GPUs)', '10 nodes\n(40 GPUs)', '25 nodes\n(100 GPUs)',
               '50 nodes\n(200 GPUs)', '100 nodes\n(400 GPUs)']
efficiency  = [94.2, 94.6, 90.8, 84.6, 85.6]

# Color by tier: green ≥ 93%, orange 88–93%, red < 88%
colors = ['#43A047' if e >= 93 else ('#FB8C00' if e >= 88 else '#E53935')
          for e in efficiency]

fig, ax = plt.subplots(figsize=(8, 4))

bars = ax.barh(scales, efficiency, color=colors, height=0.5,
               edgecolor='white', linewidth=0.8)

# Perfect scaling reference
ax.axvline(100, color='#9E9E9E', linestyle='--', linewidth=1.3, alpha=0.7,
           label='Perfect scaling (100%)')

# RING→TREE boundary annotation
ax.axhline(1.5, color='#C62828', linestyle=':', linewidth=1.2, alpha=0.6)
ax.text(70.5, 1.55, 'RING_LL → TREE_LL', fontsize=9, color='#C62828', va='bottom')

for bar, eff in zip(bars, efficiency):
    ax.text(eff + 0.4, bar.get_y() + bar.get_height() / 2,
            f'{eff}%', va='center', ha='left', fontsize=12, fontweight='bold')

ax.set_xlabel('Scaling efficiency (%)')
ax.set_title('Multi-Node Scaling Efficiency — O96, Compiled BF16', fontweight='bold')
ax.set_xlim(65, 105)
ax.invert_yaxis()

legend_handles = [
    mpatches.Patch(color='#43A047', label='≥ 93%  (AllReduce fully overlapped)'),
    mpatches.Patch(color='#FB8C00', label='88–93%  (transitional)'),
    mpatches.Patch(color='#E53935', label='< 88%  (TREE_LL on critical path)'),
    plt.Line2D([0], [0], color='#9E9E9E', linestyle='--', linewidth=1.3,
               label='Perfect scaling'),
]
ax.legend(handles=legend_handles, loc='lower right', fontsize=10, framealpha=0.9)

fig.tight_layout()
out = '../plots/0.1_exec_summary_scaling.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
