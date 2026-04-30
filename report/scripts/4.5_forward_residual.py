"""
Figure 13 — Forward residual decomposition: baseline forward compute,
ncclDevKernel_Broadcast_RING_LL, and unexplained overhead by node count.

Data sources:
  - Forward residual (step − backward − optimizer median): Full Per-Step
    Timing Statistics table (nsys rank-0 medians).
  - Broadcast_RING_LL per-step cost: nsys gpukernsum total kernel time ÷
    steps completed (1 node/2 nodes/10 nodes: 200 steps; 25 nodes: 80 steps;
    50 nodes: 40 steps; 100 nodes: 24 steps).
  - Broadcast uses RING_LL at all node counts (no TREE_LL variant observed).
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

labels  = ['1 GPU\n(no DDP)', '1 node\n(4 GPUs)', '2 nodes\n(8 GPUs)',
           '10 nodes\n(40 GPUs)', '25 nodes\n(100 GPUs)',
           '50 nodes\n(200 GPUs)', '100 nodes\n(400 GPUs)']

# Forward residual medians (ms) from Full Per-Step Timing Statistics
forward = np.array([261.8, 272.9, 284.3, 284.8, 302.0, 387.9, 369.3])

# Broadcast_RING_LL per step (ms) from nsys gpukernsum
broadcast = np.array([0.0, 4.1, 11.1, 23.6, 37.1, 62.1, 101.6])

# Baseline: single-GPU forward compute floor
baseline = 261.8
base = np.full(len(labels), baseline)

# Unexplained = residual that is neither baseline compute nor broadcast
unexplained = np.maximum(forward - base - broadcast, 0.0)

x = np.arange(len(labels))
width = 0.55

fig, ax = plt.subplots(figsize=(12, 6))

b1 = ax.bar(x, base,        width, label='Baseline forward compute (1-GPU)',
            color='#1565C0', alpha=0.85)
b2 = ax.bar(x, broadcast,   width, bottom=base,
            label='`ncclDevKernel_Broadcast_RING_LL`',
            color='#E53935', alpha=0.85)
b3 = ax.bar(x, unexplained, width, bottom=base + broadcast,
            label='Unexplained overhead',
            color='#B0BEC5', alpha=0.85)

ax.axhline(baseline, color='#1565C0', linestyle='--', linewidth=1.4,
           alpha=0.6, label=f'1-GPU baseline ({baseline} ms)')

# Annotate total bar height and broadcast fraction where notable
for i, (f, b, u) in enumerate(zip(forward, broadcast, unexplained)):
    ax.text(x[i], f + 4, f'{f:.0f} ms', ha='center', va='bottom',
            fontsize=10, fontweight='bold')
    if b > 1:
        pct = b / f * 100
        ax.text(x[i], baseline + b / 2, f'{b:.1f}\n({pct:.0f}%)',
                ha='center', va='center', fontsize=8.5, color='white',
                fontweight='bold')
    if u > 5:
        ax.text(x[i], baseline + b + u / 2, f'?{u:.0f} ms',
                ha='center', va='center', fontsize=9, color='#37474F',
                fontweight='bold')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Forward residual (ms)')
ax.set_title('Forward Residual Decomposition by Node Count\n'
             '(O96, eager BF16; Broadcast uses RING_LL at all scales)',
             fontweight='bold')
ax.set_ylim(0, 440)
ax.legend(loc='upper left', framealpha=0.9)

fig.tight_layout()
out = '../plots/4.5_forward_residual.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
