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
    df = pd.read_csv('2_initial_scaling_n320.csv')
except FileNotFoundError:
    print("Error: '2_initial_scaling_n320.csv' not found.")
    exit()

# Strong Scaling Analysis Plot

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

ax1.set_xticks(df['Nodes'])
ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

for index, row in df.iterrows():
    ax1.annotate(f"{int(row['Slurm Total Time (s)'])}s",
                 (row['Nodes'], row['Slurm Total Time (s)']),
                 textcoords="offset points",
                 xytext=(-30,-15),
                 ha='left',
                 color=color1)

    ax2.annotate(f"{int(row['Total Node Hours (h)'])}h",
                 (row['Nodes'], row['Total Node Hours (h)']),
                 textcoords="offset points",
                 xytext=(-30,10),
                 ha='left',
                 color=color2)

plt.title('Anemoi Training Strong Scaling Performance', fontweight='bold')

lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center left')

fig.tight_layout()

png_filepath = '../../report/plots/1.3_n320_strong_scaling_plot.png'
plt.savefig(f"{png_filepath}", dpi=300, bbox_inches='tight')
print(f"Plot saved as '{png_filepath}'")

plt.close()

# Analysis of Training vs. Setup Time

ig, ax1 = plt.subplots(figsize=(12, 8))
color1 = 'tab:blue' # Changed colors for better standard visual contrast
color2 = 'tab:red'

ax1.loglog(df['Nodes'], df['Job Training Time'], color=color1, marker='o', linestyle='-', label='Job Training Time (s)')
ax1.set_xlabel('Number of Nodes (Log Scale)')
ax1.set_ylabel('Job Training Time (s) (Log Scale)', color=color1)
ax1.tick_params(axis='y', labelcolor=color1)
ax1.set_xticks(df['Nodes'])
ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

ax2 = ax1.twinx()
ax2.loglog(df['Nodes'], df['Training setup time'], color=color2, marker='s', linestyle='--', label='Training Setup Time (s)')
ax2.set_ylabel('Training Setup Time (s) (Log Scale)', color=color2)
ax2.tick_params(axis='y', labelcolor=color2)

ax1.set_xticks(df['Nodes'])
ax1.get_xaxis().set_major_formatter(mticker.ScalarFormatter())

for index, row in df.iterrows():
    ax1.annotate(f"{int(row['Job Training Time'])}s",
                 (row['Nodes'], row['Job Training Time']),
                 textcoords="offset points",
                 xytext=(-30,-15),
                 ha='left',
                 color=color1)

    ax2.annotate(f"{int(row['Training setup time'])}s",
                 (row['Nodes'], row['Training setup time']),
                 textcoords="offset points",
                 xytext=(-30,10),
                 ha='left',
                 color=color2)

plt.title('Analysis of Training vs. Setup Time', fontweight='bold')
lines, labels = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines + lines2, labels + labels2, loc='center left')
fig.tight_layout()

png_filepath = '../../report/plots/1.4_n320_training_time_analysis.png'
plt.savefig(png_filepath, dpi=300, bbox_inches='tight')
print(f"Plot saved as '{png_filepath}'")

plt.close()