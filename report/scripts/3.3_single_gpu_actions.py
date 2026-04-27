"""
Figure 8 — Single GPU optimisation actions: throughput comparison.
Data source: Anemoi simple profiler, 40–200 steps per configuration.

Note: torch.compile throughput is depressed by recompilation cost over the short
run; the comparable baseline for compilation overhead is avg batch time (+7.5%),
not end-to-end throughput. Both are shown.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 13,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
})

configs = [
    'Baseline\n(eager BF16, batch 8)',
    'Batch size 16',
    'DataLoader\nworkers 16/32',
    'torch.compile\n(incl. recompile)',
    'FP8\n(compiled)',
    'Fused AdamW\n(compiled)',
]
throughput = [7.93, 7.79, 7.93, 6.27, 6.32, 6.18]   # samples/s
baseline   = 7.93

colors = []
for i, t in enumerate(throughput):
    if i == 0:
        colors.append('#1565C0')   # baseline — blue
    elif t >= baseline:
        colors.append('#2E7D32')   # improvement — green (none here)
    else:
        colors.append('#B71C1C')   # regression — red

fig, ax = plt.subplots(figsize=(11, 6))

x = np.arange(len(configs))
bars = ax.barh(x, throughput, color=colors, edgecolor='white', linewidth=1.0, height=0.55)

ax.axvline(baseline, color='#1565C0', linestyle='--', linewidth=1.5, alpha=0.7,
           label=f'Baseline ({baseline} samples/s)')

for bar, t, cfg in zip(bars, throughput, configs):
    delta = (t - baseline) / baseline * 100
    label = f'{t:.2f}' if delta == 0 else f'{t:.2f}  ({delta:+.1f}%)'
    ax.text(t + 0.05, bar.get_y() + bar.get_height() / 2,
            label, va='center', ha='left', fontsize=11)

ax.set_yticks(x)
ax.set_yticklabels(configs)
ax.invert_yaxis()
ax.set_xlabel('Training throughput (samples/s)')
ax.set_title('Single GPU Optimisation Actions — Throughput vs Baseline\n(O96, GH200, batch 8)',
             fontweight='bold')
ax.set_xlim(0, 10.5)
ax.legend(loc='lower right')

ax.text(0.98, 0.04,
        'All actions fail to improve throughput.\nBottleneck is HBM3e memory bandwidth.',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=10,
        color='#7f0000',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffebee', edgecolor='#ef9a9a'))

fig.tight_layout()
out = '../plots/3.3_single_gpu_actions.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
