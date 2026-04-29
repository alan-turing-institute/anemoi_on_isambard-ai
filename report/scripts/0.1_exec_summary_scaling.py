"""
Figure 0.1 — Executive summary: multi-node scaling efficiency headline.
Compact horizontal bar chart for the executive summary section.
Data source: nsys NVTX step median, rank 0. O96, compiled BF16.
"""

import plotly.graph_objects as go

scales     = ['2 nodes (8 GPUs)', '10 nodes (40 GPUs)', '25 nodes (100 GPUs)',
              '50 nodes (200 GPUs)', '100 nodes (400 GPUs)']
efficiency = [94.2, 94.6, 90.8, 84.6, 85.6]

colors = ['#43A047' if e >= 93 else ('#FB8C00' if e >= 88 else '#E53935')
          for e in efficiency]

fig = go.Figure()

fig.add_trace(go.Bar(
    y=scales,
    x=efficiency,
    orientation='h',
    marker_color=colors,
    marker_line_color='white',
    marker_line_width=1,
    text=[f'{e}%' for e in efficiency],
    textposition='outside',
    hovertemplate='<b>%{y}</b><br>Efficiency: %{x}%<extra></extra>',
))

# Perfect scaling reference line
fig.add_vline(x=100, line_dash='dash', line_color='#9E9E9E', line_width=1.5,
              annotation_text='Perfect scaling', annotation_position='top right')

# RING→TREE boundary
fig.add_hline(y=1.5, line_dash='dot', line_color='#C62828', line_width=1.2,
              annotation_text='RING_LL → TREE_LL', annotation_position='bottom right',
              annotation_font_color='#C62828')

fig.update_layout(
    title=dict(text='Multi-Node Scaling Efficiency — O96, Compiled BF16',
               font_size=16, x=0.5, xanchor='center'),
    xaxis=dict(title='Scaling efficiency (%)', range=[65, 108]),
    yaxis=dict(autorange='reversed'),
    height=350,
    margin=dict(l=10, r=40, t=60, b=40),
    plot_bgcolor='white',
    paper_bgcolor='white',
    font_size=13,
    showlegend=False,
)
fig.update_xaxes(showgrid=True, gridcolor='#EEEEEE')
fig.update_yaxes(showgrid=False)

fig.write_html('../plots/0.1_exec_summary_scaling.html', include_plotlyjs='cdn')
fig.write_image('../plots/0.1_exec_summary_scaling.png', width=800, height=350, scale=3)
print('Saved 0.1_exec_summary_scaling.html + .png')
