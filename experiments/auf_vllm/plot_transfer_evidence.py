"""Completed Qwen/Llama objective comparison, with paired request uncertainty."""
import argparse
from pathlib import Path
import numpy as np
from .compare_outputs import compare
from .plot_decoding_evidence import paths,read
from .pilot_data import write,sha


def main(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(8.5,3.8),layout='constrained')
    results={};sources={}
    for ax,family,title in zip(axes,['q8','llama'],['Qwen3: 4B drafter → 8B target','Llama: 3.1-8B drafter → 3.2-3B target']):
        root=args.reports/('q8-selected-dev128' if family=='q8' else 'llama-selected-dev128')
        files={m:(paths(root,'ar8' if m=='ar' else m) if family=='q8' else [root/f'{m}-r0-w0.jsonl'])
               for m in ['ar','zip','ce','auf']}
        for m in ['zip','ce','auf']:
            checked=compare(files['ar'],files[m]);assert checked['count']==checked['exact_matches']==128
        data={m:read(p) for m,p in files.items()};keys=sorted(data['ar'])
        sample=np.random.default_rng(20260911).integers(0,128,(10000,128))
        times={m:np.array([data[m][k]['wall_seconds'] for k in keys]) for m in files}
        total_tokens=sum(data['ar'][k]['output_tokens'] for k in keys)
        scores={}
        for m in ['zip','ce','auf']:
            bootstrap=times['ar'][sample].sum(1)/times[m][sample].sum(1)
            scores[m]={'tps':float(total_tokens/times[m].sum()),'speedup_ar':float(times['ar'].sum()/times[m].sum()),
                       'request_ci95':np.quantile(bootstrap,[.025,.975]).tolist(),'exact_matches':128}
        values=[scores[m]['speedup_ar'] for m in scores]
        errors=np.array([[scores[m]['speedup_ar']-scores[m]['request_ci95'][0] for m in scores],
                         [scores[m]['request_ci95'][1]-scores[m]['speedup_ar'] for m in scores]])
        ax.bar(['ZIP','CE','AUF'],values,yerr=errors,capsize=3,color=['#356494','#d79a50','#c96b62'])
        ax.set_title(title,fontsize=11);ax.set_ylabel('End-to-end throughput / AR')
        ax.set_ylim(0,max(v+e for v,e in zip(values,errors[1]))*1.2)
        for i,m in enumerate(scores):ax.text(i,values[i]+errors[1,i]+.05,f"{scores[m]['tps']:.1f} tok/s",ha='center',fontsize=9)
        ax.spines[['top','right']].set_visible(False)
        results[family]=scores
        sources.update({str(p):sha(p) for group in files.values() for p in group})
    fig.supxlabel('4,096 records · 3 epochs · 128 development requests · cap 2,048\nOne fitting seed and timing; intervals capture paired request variation only',fontsize=9)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','.pdf']:fig.savefig(args.figure.with_suffix(suffix),dpi=180)
    write(args.figure.with_suffix('.json'),{'results':results,'sources':sources,'bootstrap_seed':20260911,
          'resamples':10000,'interpretation':'Separate target/runtime comparisons; no cross-hardware TPS ranking',
          'uncertainty':'Request variation only; excludes fitting, repeat timing, and selection uncertainty'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--reports',type=Path,required=True);p.add_argument('--figure',type=Path,required=True)
    main(p.parse_args())
