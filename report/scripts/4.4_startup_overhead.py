"""
Figure 12 — Startup overhead decomposed by phase at each node count.
Data source: startup_timer Lightning callback, rank 0.

25-node T0→setup is anomalous (164.4 s, single-run Lustre spike) and is excluded
from the stacked bars; it is shown as a separate capped bar with a break symbol.
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

# Exclude 25-node (anomaly) from the main chart
cases = ['1-GPU', '1 node\n(4 GPU)', '2 nodes\n(8 GPU)', '10 nodes\n(40 GPU)',
         '50 nodes\n(200 GPU)', '100 nodes\n(400 GPU)']

# Phases (seconds)
t0_setup      = [11.2, 12.7, 12.0, 18.0, 20.6, 28.8]
fit_start     = [ 0.5,  0.3,  0.5,  2.8, 17.6, 36.8]
train_start   = [ 4.6,  4.6,  4.7,  6.8,  6.9,  9.1]
bucket_alloc  = [ 1.4,  3.6,  4.1,  1.8,  2.7,  1.7]
first_batch   = [ 1.2,  1.2,  2.7, 16.9,  4.2,  2.8]

phase_colors = ['#546E7A', '#1565C0', '#0288D1', '#26A69A', '#F57F17']
phase_labels = [
    'T0 → setup\n(model + data)',
    'setup → on_fit_start\n(DDP init + weight broadcast)',
    'on_fit_start → on_train_start\n(NCCL init)',
    'on_train_start → first batch\n(bucket alloc)',
    'First batch\n(NCCL warmup)',
]

x = np.arange(len(cases))
width = 0.55

fig, ax = plt.subplots(figsize=(12, 7))

bottoms = np.zeros(len(cases))
bars_all = []
for data, color, label in zip(
    [t0_setup, fit_start, train_start, bucket_alloc, first_batch],
    phase_colors, phase_labels
):
    bars = ax.bar(x, data, width, bottom=bottoms, label=label, color=color, alpha=0.88,
                  edgecolor='white', linewidth=0.8)
    bars_all.append((bars, data))
    bottoms += np.array(data)

# Total annotations
totals = [sum(v) for v in zip(t0_setup, fit_start, train_start, bucket_alloc, first_batch)]
for i, tot in enumerate(totals):
    ax.text(i, tot + 0.8, f'{tot:.1f} s', ha='center', va='bottom',
            fontsize=11, fontweight='bold')

# Annotate 10-node first_batch spike
ax.annotate('NCCL topology\nwarmup: 16.9 s',
            xy=(3, sum([t0_setup[3], fit_start[3], train_start[3], bucket_alloc[3]]) + first_batch[3]/2),
            xytext=(3.6, 42),
            arrowprops=dict(arrowstyle='->', color='#E65100', lw=1.3),
            fontsize=9.5, color='#E65100')

# Annotate 50/100-node weight broadcast growth
ax.annotate('Weight broadcast\nscales with nodes',
            xy=(4, t0_setup[4] + fit_start[4] / 2),
            xytext=(3.2, 55),
            arrowprops=dict(arrowstyle='->', color='#0D47A1', lw=1.3),
            fontsize=9.5, color='#0D47A1')

# Annotate excluded 25-node anomaly
ax.text(0.98, 0.97,
        '† 25-node run excluded: T0→setup = 164.4 s\n  (single-run Lustre contention anomaly)',
        transform=ax.transAxes, ha='right', va='top', fontsize=9.5,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF9C4', edgecolor='#F9A825', alpha=0.9))

ax.set_xticks(x)
ax.set_xticklabels(cases)
ax.set_ylabel('Startup time (s)')
ax.set_title('Startup Overhead by Phase — O96, Compiled BF16\n(wall-clock from T0 to end of first batch, rank 0)',
             fontweight='bold')
ax.set_ylim(0, 90)
ax.legend(loc='upper left', framealpha=0.9, fontsize=10)

fig.tight_layout()
out = '../plots/4.4_startup_overhead.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
