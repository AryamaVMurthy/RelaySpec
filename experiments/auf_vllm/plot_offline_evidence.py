"""Render measured offline results; no synthetic error bars or TPS claims."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    root=Path(__file__).parent
    reports=root/'reports'
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    colors={'zip':'#286397','ce':'#b7652b','auf':'#308169'}
    names={'zip':'ZIP feature loss','ce':'Uniform CE','auf':'AUF'}
    rates=['1e-4','3e-4','1e-3']
    data={}
    for loss in names:
        values=[]
        for rate in rates:
            report=json.loads((reports/f'q8-lr-screen/q8-n512-{loss}-lr{rate}-s42/offline-validation.json').read_text())
            endpoint=next(x for x in report['results'] if x['epoch']==3)
            values.append(endpoint['metrics']['accepted_prefix'])
        axes[0].plot([float(x) for x in rates],values,'o-',label=names[loss],color=colors[loss])
        data[loss]={'lr_screen_prefix':values}
        rate='1e-3' if loss=='zip' else '1e-4'
        report=json.loads((reports/f'q8-main-fits/q8-n4096-{loss}-lr{rate}-s42/offline-validation.json').read_text())
        axes[1].plot([x['epoch'] for x in report['results']],[x['metrics']['accepted_prefix'] for x in report['results']],
                     'o-',label=names[loss]+f' ({rate})',color=colors[loss])
        data[loss]['initial_4096']=report['results']
    f0=data['zip']['initial_4096'][-1]['metrics']['accepted_prefix']
    values=[f0]
    for loss in ['ce','auf']:
        report=json.loads((reports/f'q8-lora-screen-31320/q8-n512-{loss}-r56-lr1e-4-s42/offline-validation.json').read_text())
        values.append(next(x for x in report['results'] if x['epoch']==3)['metrics']['accepted_prefix'])
    axes[2].bar(['ZIP F0','+ CE LoRA','+ AUF LoRA'],values,color=[colors[x] for x in ['zip','ce','auf']])
    for index,value in enumerate(values):axes[2].text(index,value+.09,f'{value:.3f}',ha='center',fontsize=9)
    axes[0].set_xscale('log');axes[0].set_xticks([float(x) for x in rates],rates)
    axes[0].set_xlabel('Learning rate');axes[0].set_title('512 records, 3 epochs')
    axes[1].set_xticks([1,3]);axes[1].set_xlabel('Training epoch');axes[1].set_title('Initial 4,096-record fits')
    axes[2].set_title('ZIP-initialized rank-56 continuation');axes[2].set_xlabel('512 continuation records, 3 epochs')
    for ax in axes:
        ax.set_ylim(0,6.5);ax.set_ylabel('Mean accepted draft prefix (offline)')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(fontsize=8,loc='upper left');axes[1].legend(fontsize=8,loc='upper left')
    out=root/'figures';out.mkdir(exist_ok=True)
    fig.savefig(out/'offline_objective_comparison.png',dpi=180)
    fig.savefig(out/'offline_objective_comparison.pdf')
    (out/'offline_objective_comparison.caption.txt').write_text(
        'Qwen4-to-Qwen8, seed42, 1,024 separate offline-validation records and four fixed anchors per record. '
        'These are teacher-context prefix proxies, not decoding throughput. No seed uncertainty is estimated. '
        'Left: each learning-rate trial uses512 records and3 epochs; ZIP and token losses have different update counts and computation. '
        'Middle: initial4096-record fits, before LR-selected refitting. Right: frozen ZIP4096 initialization plus512-record rank56 LoRA; charge all4096 initialization records and ZIP fitting cost. '
        'A small proxy difference is not evidence of a speedup.\n')


if __name__=='__main__':main()
