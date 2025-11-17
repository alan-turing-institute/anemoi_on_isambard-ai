import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


plt.style.use('seaborn-v0_8-whitegrid')

plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 18
})

try:
    df = pd.read_csv('1_initial_scaling.csv')
except FileNotFoundError:
    print("Error: '1_initial_scaling.csv' not found.")
    exit()

fig, ax1 = plt.subplots(figsize=(10, 7))

color1 = 'tab:blue'
color2 = 'tab:red'

ax1.loglog(df['Nodes'], df['Slurm Total Time (s)'], color=color1, marker='o', linestyle='-', label='Slurm Total Time (s)')
ax1.set_xlabel('Number of Nodes (Log Scale)')
ax1.set_ylabel('Slurm Total Time (s) (Log Scale)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)

ax1.set_xticks(df['Nodes'])
ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter()) # Use standard numbers for ticks

ax2 = ax1.twinx()
ax2.loglog(df['Nodes'], df['Total Node Hours (h)'], color=color2, marker='s', linestyle='--', label='Total Node Hours (h)')
ax2.set_ylabel('Total Node Hours (h) (Log Scale)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

plt.title('Anemoi Training Strong Scaling Performance', fontweight='bold')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='upper center')

fig.tight_layout()

png_filepath = '../../report/plots/1_strong_scaling_plot.png'
plt.savefig(f"{png_filepath}", dpi=300, bbox_inches='tight')
print(f"Plot saved as '{png_filepath}'")

plt.close()
