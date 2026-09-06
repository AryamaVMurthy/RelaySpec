"""Plot matched fitting trajectories from the audited four-arm rate check."""
import json
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path('reports/autoresearch-20260907')
d = json.loads((root / 'rate-check-summary.json').read_text())
if d['status'] != 'pass':
    raise ValueError('Require audited completed fits')
fig, axes = plt.subplots(1, 3, figsize=(11, 3.3), sharey=True)
for i, (ax, indices, title) in enumerate(zip(axes, [[0,1], [2], [3]],
        ['Dense, 512 records', 'Factorized-4096, 2,048 records', 'MLP-4096, 2,048 records'])):
    ref = d['results'][indices[0]]['original_reference']
    series = [(ref, 'Original LR 0.0006', 'black', '--', 's')]
    for j, index in enumerate(indices):
        r = d['results'][index]
        series.append((r, f"LR {r['trial']['learning_rate']:g}", ['#245b87','#b56a2a'][j], '-', ['o','^'][j]))
    for r, label, color, linestyle, marker in series:
        rows = [v for v in r['validation_trajectory'] if v['step'] > 0]
        ax.plot([v['step'] for v in rows], [v['groups']['validation']['objective'] for v in rows],
                label=label, color=color, linestyle=linestyle, marker=marker, markersize=4)
    ax.set_xscale('log', base=2)
    ax.set_xticks([128,512,2048,8192], ['128','512','2,048','8,192'])
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Training updates')
    ax.grid(alpha=.2)
    ax.legend(fontsize=8)
axes[0].set_ylabel('Validation relative interface error')
fig.tight_layout()
for suffix in ['png','pdf']:
    fig.savefig(root / f'rate-check-trajectories.{suffix}', dpi=180)
