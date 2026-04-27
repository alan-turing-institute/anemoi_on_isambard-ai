"""
Figure 7 — ncu roofline scatter: Memory SOL % vs Compute SOL % per kernel type.
Data source: ncu --set roofline, job 4263705 (eager BF16, 500 kernels post-warmup).

Each kernel is plotted as a point with error bars spanning the observed SOL range.
The y = x diagonal separates the memory-bound (right) from compute-bound (top) regions.
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

# (memory_sol_mid, compute_sol_mid, memory_err, compute_err, label, regime)
kernels = [
    (92.0, 33.0,  4.0,  3.0, 'CUTLASS GEMM\n(linear projections)',     'memory'),
    (91.5, 21.0,  1.5,  8.0, 'Element-wise\n(add, mul, copy)',          'memory'),
    (90.0, 53.0,  0.0,  0.0, 'Layer norm\nbackward',                    'memory'),
    (70.0, 87.5,  5.0,  7.5, 'nvjet_hsh\n(graph message-passing)',      'ridge'),
    (70.0, 87.5,  5.0,  7.5, 'FlashAttention\n(fwd/bwd)',               'ridge'),
    (14.0, 56.0,  0.0,  0.0, 'indexFuncLargeIndex\n(sparse routing)',   'latency'),
]

colors = {
    'memory':  '#E53935',
    'ridge':   '#43A047',
    'latency': '#FB8C00',
}
markers = {
    'memory':  'o',
    'ridge':   's',
    'latency': '^',
}

fig, ax = plt.subplots(figsize=(9, 7))

# Shaded regions
ax.fill_betweenx([0, 100], [0, 0],  [100, 100], where=[True]*2,
                 alpha=0.04, color='#E53935', zorder=0)
ax.fill_between([0, 100],  [0, 0],  [100, 100], where=[True]*2,
                alpha=0.04, color='#43A047', zorder=0)

# y = x ridge line
diag = np.linspace(0, 100, 200)
ax.plot(diag, diag, 'k--', linewidth=1.2, alpha=0.4, label='Memory SOL = Compute SOL')

# Region labels
ax.text(82, 18, 'Memory-bound\n(Memory SOL >> Compute SOL)',
        fontsize=10, color='#C62828', ha='center', style='italic', alpha=0.8)
ax.text(18, 82, 'Compute-bound\n(Compute SOL >> Memory SOL)',
        fontsize=10, color='#2E7D32', ha='center', style='italic', alpha=0.8)

# Offset identical nvjet/FlashAttention points slightly for readability
offsets = {
    'nvjet_hsh\n(graph message-passing)':  (-6, 2),
    'FlashAttention\n(fwd/bwd)':           ( 6, -2),
}

plotted_regimes = set()
for mem, comp, mem_err, comp_err, label, regime in kernels:
    dx, dy = offsets.get(label, (0, 0))
    x, y = mem + dx, comp + dy
    xerr = [[mem_err], [mem_err]] if mem_err else None
    yerr = [[comp_err], [comp_err]] if comp_err else None
    ax.errorbar(x, y, xerr=xerr, yerr=yerr,
                fmt=markers[regime], color=colors[regime],
                markersize=12, capsize=4, linewidth=1.5,
                label=regime if regime not in plotted_regimes else None)
    plotted_regimes.add(regime)
    ax.annotate(label, (x, y),
                textcoords='offset points', xytext=(8, 4),
                fontsize=9, color=colors[regime])

ax.set_xlabel('Memory Throughput SOL (%)')
ax.set_ylabel('Compute (SM) Throughput SOL (%)')
ax.set_title('ncu Speed-of-Light: Per-Kernel Performance Regime\n(GH200 O96, eager BF16)', fontweight='bold')
ax.set_xlim(0, 105)
ax.set_ylim(0, 105)

legend_elements = [
    mpatches.Patch(color=colors['memory'],  label='Memory-bound'),
    mpatches.Patch(color=colors['ridge'],   label='Near ridge point'),
    mpatches.Patch(color=colors['latency'], label='Latency/cache-bound'),
    plt.Line2D([0], [0], color='k', linestyle='--', linewidth=1.2, alpha=0.5,
               label='Memory SOL = Compute SOL'),
]
ax.legend(handles=legend_elements, loc='lower right', framealpha=0.9)

fig.tight_layout()
out = '../plots/3.2_roofline.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
