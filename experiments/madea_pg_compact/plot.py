"""Standalone figure: PG throughput, end-to-end time and initializer welfare."""
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

root = Path(sys.argv[1])
pairs = pd.read_csv(root / 'pairs.csv')
hybrid = pd.read_csv(root / 'hybrid.csv')
fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
for nf, group in pairs.groupby('functions'):
  for axis, metric in zip(axes[:2], ['pg_throughput_speedup', 'wall_speedup']):
    grouped = group.groupby('nodes')[metric]
    stats = grouped.agg(['median', 'min', 'max']).reset_index()
    axis.errorbar(stats.nodes, stats['median'],
                  yerr=[stats['median'] - stats['min'], stats['max'] - stats['median']],
                  fmt='o-', capsize=3, label=f'{nf} funzioni')
for axis, title in zip(axes[:2], ['Velocità per proposta PG', 'Velocità totale']):
  axis.axhline(1, color='grey', lw=1)
  axis.set(xlabel='Nodi', ylabel='Speedup: completo / compatto', title=title)
  axis.grid(alpha=.2)
  axis.legend()
axes[0].set_yscale('log')
for nf, group in hybrid.groupby('functions'):
  axes[2].scatter(group.nodes, group.hybrid_welfare_gain_pct, alpha=.65, label=f'{nf} funzioni')
axes[2].axhline(0, color='grey', lw=1)
axes[2].set(xlabel='Nodi', ylabel='Welfare Gurobi locale vs DP (%)', title='Effetto dell’inizializzazione')
axes[2].grid(alpha=.2)
axes[2].legend()
fig.savefig(root / 'summary.png', dpi=180)
fig.savefig(root / 'summary.pdf')
print(root / 'summary.png')
