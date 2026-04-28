"""
Figure 6 — GPU kernel time breakdown by type (nsys, compiled BF16, 200 steps).
Data source: nsys stats GPU kernel summary, rank 0.
"""

import matplotlib.pyplot as plt
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

kernels = [
    'Other kernels\n(element-wise, norm, etc.)',
    'D2H transfers',
    'Sparse routing\n(indexSelectLargeIndex)',
    'FlashAttention\n(fwd + bwd)',
    'Graph message-passing\n(nvjet_hsh)',
]
shares = [31, 1, 13, 19, 36]
colors = ['#90A4AE', '#B0BEC5', '#FF8F00', '#42A5F5', '#7B1FA2']

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.barh(kernels, shares, color=colors, edgecolor='white', linewidth=1.0)

for bar, pct in zip(bars, shares):
    x = bar.get_width()
    ax.text(x + 0.5, bar.get_y() + bar.get_height() / 2,
            f'{pct}%', va='center', ha='left', fontsize=13, fontweight='bold')

ax.set_xlabel('Share of GPU kernel time (%)')
ax.set_title('GPU Kernel Time Breakdown — Compiled BF16, 200 Steps', fontweight='bold')
ax.set_xlim(0, 45)
ax.invert_yaxis()

fig.tight_layout()
out = '../plots/3.1_kernel_breakdown.png'
plt.savefig(out, dpi=300, bbox_inches='tight')
print(f'Saved {out}')
plt.close()
