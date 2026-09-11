"""Summarize non-overlapping kernel events; never turn profiled wall time into TPS."""
import argparse
import collections
import gzip
import json
from pathlib import Path
from .pilot_data import sha,write


def category(name):
    name=name.lower()
    if any(x in name for x in ['matmul','gemm','gemv']):return 'Matrix multiplication'
    if any(x in name for x in ['flash_fwd','attention']):return 'Attention'
    if 'norm' in name:return 'Normalization'
    return 'Other kernels'


def analyze(path):
    with gzip.open(path,'rt') as f:trace=json.load(f)
    kernels=[e for e in trace['traceEvents'] if e.get('cat')=='kernel' and e.get('ph')=='X']
    assert kernels
    durations=collections.defaultdict(float);names=collections.defaultdict(float)
    for e in kernels:
        durations[category(e['name'])]+=e['dur']
        names[e['name']]+=e['dur']
    total=sum(durations.values())
    intervals=sorted((e['ts'],e['ts']+e['dur']) for e in kernels)
    merged=[]
    for start,end in intervals:
        if merged and start<=merged[-1][1]:merged[-1][1]=max(end,merged[-1][1])
        else:merged.append([start,end])
    busy=sum(b-a for a,b in merged)
    window=max(b for a,b in intervals)-min(a for a,b in intervals)
    return {'source_sha256':sha(path),'kernel_events':len(kernels),
            'summed_kernel_ms':total/1000,'kernel_union_ms':busy/1000,
            'first_to_last_kernel_ms':window/1000,'kernel_union_fraction':busy/window,
            'category_fraction':{k:v/total for k,v in durations.items()},
            'top_kernels':[{'name':name,'milliseconds':duration/1000,'fraction':duration/total}
                           for name,duration in sorted(names.items(),key=lambda x:-x[1])[:12]],
            'scope':'At most64 profiled engine iterations from four128-cap requests; not a full2048-output profile or SM occupancy measurement'}


def main(args):
    report={p.parent.name:analyze(p) for p in sorted(args.root.glob('*/*.gz'))}
    assert len(report)==5
    write(args.out,report)
    for name,data in report.items():print(name,json.dumps(data['category_fraction']))
    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        order=['ar8','native8','zip','ce','auf'];labels=['AR','Native','ZIP','CE','AUF']
        colors=['#356494','#79a7ce','#d9aa53','#cccccc']
        fig,ax=plt.subplots(figsize=(6.7,3.3),layout='constrained');bottom=[0.]*5
        for kind,color in zip(['Matrix multiplication','Attention','Normalization','Other kernels'],colors):
            heights=[100*next(v for k,v in report.items() if k.startswith(name+'-'))['category_fraction'].get(kind,0) for name in order]
            ax.bar(labels,heights,bottom=bottom,color=color,label=kind)
            bottom=[a+b for a,b in zip(bottom,heights)]
        ax.set_ylim(0,100);ax.set_ylabel('Share of summed GPU kernel duration (%)')
        ax.legend(loc='lower center',bbox_to_anchor=(.5,1),ncol=2,frameon=False)
        fig.supxlabel('Instrumented short window; not end-to-end throughput',fontsize=9)
        args.figure.parent.mkdir(parents=True,exist_ok=True)
        for suffix in ['.pdf','.png']:fig.savefig(args.figure.with_suffix(suffix),dpi=180)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--figure',type=Path)
    main(p.parse_args())
