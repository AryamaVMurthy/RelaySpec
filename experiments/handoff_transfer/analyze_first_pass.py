"""Conditional request analysis of the first full Q8 transfer timing pass."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def main(a):
    paths={'normal':a.reports/'q8-normal-partial/normal-r0-w0.jsonl',
           'BA':a.reports/'q8-handoff-first-pass/handoff-r56-r0-w0.jsonl',
           'five_maps':a.reports/'q8-handoff-first-pass/handoff-five-r0-w0.jsonl'}
    data={k:{r['group_id']:r for r in map(json.loads,p.read_text().splitlines())} for k,p in paths.items()}
    keys=sorted(data['normal']);assert len(keys)==128
    base=np.array([data['normal'][k]['wall_seconds'] for k in keys])
    sample=np.random.default_rng(42).integers(0,128,size=(5000,128))
    results={}
    for name,rows in data.items():
        assert set(rows)==set(keys)
        assert all(rows[k]['output_ids']==data['normal'][k]['output_ids'] and
                   rows[k]['finish_reason']==data['normal'][k]['finish_reason'] for k in keys)
        seconds=np.array([rows[k]['wall_seconds'] for k in keys])
        drafts=sum(r['verification_iterations'] for r in rows.values())
        accepted=sum(r['accepted_draft_tokens'] for r in rows.values())
        ratio=base.sum()/seconds.sum()
        boot=base[sample].sum(axis=1)/seconds[sample].sum(axis=1)
        results[name]={'ratio_to_normal':float(ratio),
            'paired_request_interval_95':np.quantile(boot,[.025,.975]).tolist(),
            'faster_requests':int((seconds<base).sum()),'slower_requests':int((seconds>base).sum()),
            'verification_iterations':drafts,'accepted_draft_tokens_per_verification':accepted/drafts,
            'request_wall_ms_per_verification':1000*seconds.sum()/drafts}
    report={'status':'conditional_first_pass_analysis','results':results,
            'source_sha256':{k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
            'scope':'5000 paired request bootstrap draws; one fitting seed and one timing repetition. '
                    'Intervals exclude fitting-seed and repeated-timing variation. '
                    'Wall time per verification includes prefill; not isolated verifier kernel latency. '
                    'Accepted-token counters are engine metrics, not necessarily emitted output tokens.'}
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(results,indent=2))
    if a.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        keys=['normal','BA','five_maps'];labels=['Normal RelaySpec','AUF BA','AUF five maps'];colors=['#888888','#466c98','#d79543']
        fig,axes=plt.subplots(1,2,figsize=(9,3.8),layout='constrained')
        for i,(k,color) in enumerate(zip(keys,colors)):
            v=results[k];lo,hi=v['paired_request_interval_95'];x=v['ratio_to_normal']
            axes[0].errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=color,capsize=4)
        axes[0].set_yticks(range(3),labels);axes[0].invert_yaxis();axes[0].axvline(1,color='gray',ls='--',lw=1);axes[0].set_xlabel('Throughput / normal RelaySpec')
        values=[results[k]['accepted_draft_tokens_per_verification'] for k in keys]
        axes[1].bar(labels,values,color=colors)
        for i,v in enumerate(values):axes[1].text(i,v+.08,f'{v:.2f}',ha='center',fontsize=9)
        axes[1].set_ylim(0,7);axes[1].set_ylabel('Accepted draft tokens / verification')
        for ax in axes:ax.spines[['top','right']].set_visible(False)
        fig.suptitle('Qwen8: first complete 128-request timing pass',fontsize=12)
        fig.supxlabel('Paired request intervals only; one fit and one timing repetition. Further comparisons pending.',fontsize=9)
        a.figure.parent.mkdir(parents=True,exist_ok=True)
        for ext in ('.png','.pdf'):fig.savefig(a.figure.with_suffix(ext),dpi=180)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reports',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--figure',type=Path)
    main(p.parse_args())
