"""Paired request-bootstrap development evidence from complete unprofiled rows."""
import argparse
import json
from pathlib import Path
import numpy as np
from .compare_outputs import compare
from .pilot_data import write,sha


def paths(root,method):
    loss='zip' if method in {'ar8','native8','zip'} else method
    name=method if method in {'ar8','native8'} else 'mapped'
    return sorted(root.glob(f'{loss}/worker-*/measurements/development128-cap2048/worker_*/{name}-r0.jsonl'))


def read(files):
    return {r['group_id']:r for p in files for r in (json.loads(l) for l in p.read_text().splitlines())}


def main(args):
    order=['ar8','native8','zip','ce','auf'];data={m:read(paths(args.root,m)) for m in order}
    for method in order[1:]:
        result=compare(paths(args.root,'ar8'),paths(args.root,method))
        assert result['count']==result['exact_matches']==128
    keys=sorted(data['ar8']);sample=np.random.default_rng(20260911).integers(0,128,(10000,128))
    tokens=np.array([data['ar8'][k]['output_tokens'] for k in keys])
    scores={};boot={}
    for method in order:
        seconds=np.array([data[method][k]['wall_seconds'] for k in keys])
        boot[method]=tokens[sample].sum(1)/seconds[sample].sum(1)
        scores[method]={'tps':float(tokens.sum()/seconds.sum()),
                        'tps_request_ci95':np.quantile(boot[method],[.025,.975]).tolist()}
    for method in order:
        ratio=boot[method]/boot['zip']
        scores[method].update(ratio_to_zip=scores[method]['tps']/scores['zip']['tps'],
                              ratio_to_zip_request_ci95=np.quantile(ratio,[.025,.975]).tolist())
    write(args.out,{'results':scores,'bootstrap_seed':20260911,'bootstrap_resamples':10000,
          'requests':128,'output_cap':2048,'split':'Numina development','training_seed':42,'timing_repetitions':1,
          'sources':{str(p):sha(p) for m in order for p in paths(args.root,m)},
          'uncertainty':'Paired request resampling only; excludes fitting-seed, timing-order, and checkpoint-selection uncertainty'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
    labels=['AR','Native','ZIP','CE','AUF'];colors=['#999999','#759cad','#356494','#d79a50','#c96b62']
    values=[scores[m]['tps'] for m in order]
    errors=np.array([[scores[m]['tps']-scores[m]['tps_request_ci95'][0] for m in order],
                     [scores[m]['tps_request_ci95'][1]-scores[m]['tps'] for m in order]])
    axes[0].bar(labels,values,color=colors,yerr=errors,capsize=3)
    axes[0].set_ylabel('End-to-end tokens/s');axes[0].set_ylim(0,205)
    for i,v in enumerate(values):axes[0].text(i,v+errors[1,i]+4,f'{v:.1f}',ha='center',fontsize=9)
    methods=order[1:];ratios=[scores[m]['ratio_to_zip'] for m in methods]
    err=np.array([[scores[m]['ratio_to_zip']-scores[m]['ratio_to_zip_request_ci95'][0] for m in methods],
                  [scores[m]['ratio_to_zip_request_ci95'][1]-scores[m]['ratio_to_zip'] for m in methods]])
    axes[1].bar(labels[1:],ratios,color=colors[1:],yerr=err,capsize=3)
    axes[1].axhline(1,color='black',linestyle='--',linewidth=1)
    axes[1].set_ylabel('Throughput / ZIP');axes[1].set_ylim(0,1.15)
    for ax in axes:ax.spines[['top','right']].set_visible(False)
    fig.supxlabel('128 development requests; cap 2,048; paired request intervals; one fit seed and timing repetition',fontsize=9)
    args.figure.parent.mkdir(parents=True,exist_ok=True)
    for suffix in ['.png','.pdf']:fig.savefig(args.figure.with_suffix(suffix),dpi=180)
    print(json.dumps(scores,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--figure',type=Path,required=True);main(p.parse_args())
