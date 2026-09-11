"""Final matrix figures from verified, complete collector outputs only."""
import argparse
import json
from pathlib import Path
from .paper_results import build

LABELS={'normal_ce':'RelaySpec map','fusion_r56':'Fusion BA-56','five_maps':'Five dense maps',
        'dense_fusion':'Dense fusion','five_ba56':'Five BA-56','dense_fresh':'Fresh dense'}
ORDER=list(LABELS)


def render(paths,out):
    # Completeness, exactness and paired-reference checks happen before plotting.
    build(paths,out,require_complete=True)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    results=[json.loads(p.read_text()) for p in paths]
    lookup={(r['cell']['family'],r['cell']['kind'],r['cell']['objective']):r for r in results}
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'ps.fonttype':42})
    colors={'ce':'#3569a8','auf':'#d57a28'}
    families=['q8','llama','cross']
    titles=['Qwen transfer','Llama transfer','Qwen-to-Llama transfer']
    fig,axes=plt.subplots(1,3,figsize=(13,4.2),sharey=True,layout='constrained')
    for ax,family,title in zip(axes,families,titles):
        for loss,offset in [('ce',-.18),('auf',.18)]:
            values=[lookup[family,kind,loss]['ratio_to_controls']['zip'] for kind in ORDER]
            ax.barh([i+offset for i in range(len(ORDER))],values,height=.34,color=colors[loss],label=loss.upper())
            # Individual timing repetitions, not fitting-seed error bars.
            for i,kind in enumerate(ORDER):
                repeats=lookup[family,kind,loss]['control_ratios_by_repeat']['zip']
                ax.scatter(repeats,[i+offset]*len(repeats),s=12,color='black',alpha=.55,zorder=3)
        ax.axvline(1,color='gray',linestyle='--',linewidth=1)
        ax.set_title(title);ax.set_xlabel('Throughput / unchanged ZIP')
        ax.set_yticks(range(len(ORDER)),[LABELS[k] for k in ORDER]);ax.grid(axis='x',alpha=.15)
    axes[0].invert_yaxis();axes[-1].legend(loc='best')
    fig.savefig(out/'matrix_vs_zip.pdf');fig.savefig(out/'matrix_vs_zip.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    for ax,family,title in zip(axes,families,titles):
        for kind in ORDER:
            for loss in ('ce','auf'):
                r=lookup[family,kind,loss];fit=r['fit']
                hours=fit['training_seconds']*fit['world_size']/3600
                ax.scatter(hours,r['mean_tps'],c=colors[loss],marker='o' if loss=='ce' else '^')
                ax.annotate(LABELS[kind],(hours,r['mean_tps']),xytext=(3,3),textcoords='offset points',fontsize=7)
        ax.set_title(title);ax.set_xlabel('Final continuation GPU-hours');ax.set_ylabel('Output tokens/s');ax.grid(alpha=.15)
    fig.savefig(out/'matrix_fit_cost.pdf');fig.savefig(out/'matrix_fit_cost.png',dpi=180);plt.close(fig)
    (out/'figure_notes.txt').write_text('128 requests, output cap2048, natural EOS. Bars are mean paired ratios; dots are three timing repetitions from one fitting seed. No fitting-seed uncertainty shown. Fit-cost panels exclude initialization, rollout collection, and LR tuning. Inspect rendered annotations before manuscript inclusion.\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,nargs='+',required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();render(a.results,a.out)
