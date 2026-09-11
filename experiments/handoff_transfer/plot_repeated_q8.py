"""Plot completed paired timing repetitions without implying fitting-seed evidence."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(a):
    report = json.loads(a.input.read_text())
    assert report['timing_repetitions_complete'] == 3
    methods = [('handoff-r56', 'AUF BA', '#356494'), ('handoff-five', 'AUF five maps', '#c28334')]
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.8), layout='constrained')
    normal = [r['normal_tps'] for r in report['results']['handoff-r56']['per_repeat']]
    axes[0].plot([1, 2, 3], normal, 'o-', color='#777777', label='Normal RelaySpec')
    for index, (key, label, color) in enumerate(methods):
        rows = report['results'][key]['per_repeat']
        assert len(rows) == 3 and all(r['exact_matches'] == r['finish_matches'] == r['count'] == 128 for r in rows)
        axes[0].plot([1, 2, 3], [r['method_tps'] for r in rows], 'o-', color=color, label=label)
        gains = [100 * (r['throughput_ratio'] - 1) for r in rows]
        axes[1].scatter(gains, [index - .08, index, index + .08], color=color, s=32)
        mean = sum(gains) / 3
        axes[1].annotate(f'{mean:+.2f}%', (mean, index), xytext=(8, 0), textcoords='offset points', va='center', color=color)
    axes[0].set_xticks([1, 2, 3]); axes[0].set_xlabel('Timing repetition'); axes[0].set_ylabel('Output tokens / request wall-second')
    axes[0].legend(frameon=False, fontsize=9, loc='best'); axes[0].set_ylim(160, 183)
    axes[1].axvline(0, color='#999999', linestyle='--', linewidth=1)
    axes[1].set_yticks([0, 1], ['AUF BA', 'AUF five maps']); axes[1].set_ylim(1.5, -.5)
    axes[1].set_xlim(-7, 5); axes[1].set_xlabel('Throughput change versus normal (%)')
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.grid(axis='x', alpha=.15)
    fig.suptitle('Qwen8: three completed timing repetitions', fontsize=12)
    fig.supxlabel('128 development requests, cap 2,048, one fitting seed. AR and ZIP comparisons pending.', fontsize=9)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ('.png', '.pdf'):
        fig.savefig(a.out.with_suffix(suffix), dpi=180)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    main(p.parse_args())
