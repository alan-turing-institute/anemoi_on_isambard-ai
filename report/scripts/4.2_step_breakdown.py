"""
Figure 10 — Multi-node step time phase breakdown (median ms).
Data source: nsys NVTX phase medians, rank 0. O96, compiled BF16.
Forward is derived: step_med - backward_med - optimizer_med.
"""

import matplotlib.pyplot as plt
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
})

labels    = ['1-GPU', '4-GPU\n(1 node)', '8-GPU\n(2 nodes)', '40-GPU\n(10 nodes)',
             '100-GPU\n(25 nodes)', '200-GPU\n(50 nodes)', '400-GPU\n(100 nodes)']
backward  = [708.9, 734.9, 744.2, 737.2, 764.9, 748.2, 738.4]
forward   = [261.8, 272.9, 284.3, 284.8, 302.0, 387.9, 369.3]
optimizer = [  6.3,   8.9,   8.6,  10.7,   9.6,  18.6,  33.6]

x = np.arange(len(labels))
width = 0.55

fig, ax = plt.subplots(figsize=(12, 7))

bar_bwd = ax.bar(x, backward,  width, label='Backward',          color='#1565C0', alpha=0.9)
bar_fwd = ax.bar(x, forward,   width, bottom=backward,           label='Forward (derived)', color='#42A5F5', alpha=0.9)
bar_opt = ax.bar(x, optimizer, width, bottom=[b+f for b,f in zip(backward, forward)],
                 label='Optimizer', color='#90CAF9', alpha=0.9)

# Total step time annotation
totals = [b+f+o for b,f,o in zip(backward, forward, optimizer)]
for i, (tot, bwd, fwd, opt) in enumerate(zip(totals, backward, forward, optimizer)):
    ax.text(i, tot + 8, f'{tot:.0f} ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Annotate the forward jump at 50 nodes
ax.annotate('Forward +86 ms\nvs 25 nodes\n(DDP Broadcast +38 ms\n+ 65 ms unexplained)',
            xy=(5, backward[5] + forward[5] / 2),
            xytext=(5.55, 750),
            arrowprops=dict(arrowstyle='->', color='#C62828', lw=1.3),
            fontsize=9, color='#C62828', ha='left')

ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Median phase time (ms)')
ax.set_title('Multi-Node Step Time Phase Breakdown — O96, Compiled BF16\n(nsys NVTX medians, rank 0)',
             fontweight='bold')
ax.set_ylim(0, 1350)
ax.legend(loc='upper left')

fig.tight_layout()
out = '../plots/4.2_step_breakdown.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
