"""Plot verified fitting cost, without implying total adaptation cost."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main(a):
    data=json.loads(a.input.read_text());assert data['status']=='complete_fitting_costs_only'
    rows=data['rows'];labels=[r['method'] for r in rows]
    base=[(r['sequential_fit_seconds'] if i<2 else r['initialization_seconds'])/60 for i,r in enumerate(rows)]
    extra=[0 if i<2 else r['second_stage_seconds']/60 for i,r in enumerate(rows)]
    fig,axes=plt.subplots(1,2,figsize=(10,3.8),sharey=True,layout='constrained')
    for ax,multiplier,title in zip(axes,[1,2],['Sequential fitting time (minutes)','Fitting compute (GPU-minutes)']):
        ax.barh(labels,base,color='#466c98',label='Feature-loss fitting')
        heights=[v*multiplier for v in extra]
        ax.barh(labels,heights,left=base,color='#d79543',label='AUF/CE continuation')
        for i,(x,y) in enumerate(zip(base,heights)):
            ax.text(x+y+.25,i,f'{x+y:.1f}',va='center',fontsize=9)
        ax.set_xlim(0,max(x+y for x,y in zip(base,heights))*1.18)
        ax.set_xlabel(title);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis();axes[0].legend(loc='upper right',frameon=False,fontsize=8)
    fig.suptitle('Qwen8 adaptation: measured fitting cost on 4,096 records',fontsize=12)
    fig.supxlabel('Cached features; excludes rollout/capture, loading, gates and queue time. One fitting seed.',fontsize=9)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    for ext in ('.pdf','.png'):fig.savefig(a.out.with_suffix(ext),dpi=180)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
