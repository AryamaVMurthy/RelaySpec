"""Export descriptive fitting curves; never label tuning screens as test results."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--results',required=True);p.add_argument('--output',required=True)
a=p.parse_args();data=json.loads(Path(a.results).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
colors={4096:'#31688e',8192:'#35b779',16384:'#e8a628'}
for row,pair in enumerate(['llama','cross']):
    for col,objective in enumerate(['dense','zip']):
        ax=axes[row,col]
        for fit in sorted(data['fits'],key=lambda f:f['trial']['records']):
            t=fit['trial']
            if t['pair']!=pair or t['objective']!=objective or 'refine-' in t['output']:
                continue
            h=fit['history'];n=t['records']
            ax.plot([r['epoch'] for r in h],[r['validation_loss'] for r in h],
                    label=f'{n:,} records',color=colors[n],marker='.',linewidth=1.6)
        ax.set_title(f'{"Llama → Llama" if pair=="llama" else "Qwen → Llama"}: {objective}')
        ax.set_xlabel('Epoch in a 12-epoch cosine schedule');ax.set_ylabel('Validation objective')
        ax.set_xticks([1,3,6,9,12]);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.savefig(out/'validation-curves.png',dpi=180);fig.savefig(out/'validation-curves.pdf');plt.close(fig)

if data['screening']:
    fig,axes=plt.subplots(2,2,figsize=(9,6),constrained_layout=True)
    fig.suptitle('Development screen only: 2 MATH requests, 256-token cap, one fitting seed',fontsize=10)
    for row,pair in enumerate(['llama','cross']):
        for col,objective in enumerate(['dense','zip']):
            ax=axes[row,col];baselines=[]
            for n in [4096,8192,16384]:
                prefix=f'screen-n{n}-{pair}-{objective}-e';points=[]
                for result in data['screening']:
                    name=Path(result['config']).stem
                    if not name.startswith(prefix) or not result['exact_gate']:
                        continue
                    epoch=int(name.removeprefix(prefix))
                    tps=next(v['decode_tokens_per_second'] for k,v in result['methods'].items() if k!='native_ar')
                    if epoch==0:
                        baselines.append(tps)
                    else:
                        points.append((epoch,tps))
                if points:
                    points.sort();ax.plot(*zip(*points),label=f'{n:,} records',marker='o',color=colors[n])
            if baselines:
                ax.axhline(sum(baselines)/len(baselines),color='black',linestyle='--',label='Old mapper (mean control)')
            ax.set_title(f'{"Llama → Llama" if pair=="llama" else "Qwen → Llama"}: {objective}')
            ax.set_xlabel('Saved epoch');ax.set_ylabel('Decode tokens/s')
            ax.set_xticks([1,3,6,12]);ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.savefig(out/'development-throughput.png',dpi=180);fig.savefig(out/'development-throughput.pdf');plt.close(fig)

if data['comparison']:
    fig,ax=plt.subplots(figsize=(6,3.5),constrained_layout=True)
    for i,c in enumerate(data['comparison']):
        value=c['request_speedup_vs_old'];lo,hi=c['paired_request_bootstrap_95ci']
        ax.errorbar(i,value,yerr=[[value-lo],[hi-value]],fmt='o',capsize=5,
                    color=['#31688e','#35b779'][i],markersize=8)
        ax.annotate(f'{100*(value-1):+.1f}%',(i,value),xytext=(10,0),textcoords='offset points',va='center')
    ax.axhline(1,color='black',linestyle='--',linewidth=1)
    ax.set_xticks([0,1],['Llama → Llama','Qwen → Llama']);ax.set_xlim(-.35,1.5)
    ax.set_ylabel('Request throughput / old mapper');ax.grid(axis='y',alpha=.2)
    ax.set_title('16 reserved MATH requests per pair; 1,024-token cap',fontsize=10)
    fig.savefig(out/'heldout-improvement.png',dpi=180);fig.savefig(out/'heldout-improvement.pdf');plt.close(fig)
