"""Evidence-backed overview, adaptation-budget and block-size figures."""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BLUE, ORANGE, GREEN, GRAY = '#245B88', '#C77332', '#37836A', '#686868'
TIME = dt.datetime(2026, 9, 8, tzinfo=dt.timezone.utc)
META = {'Creator': 'RelaySpec results visuals', 'CreationDate': TIME, 'ModDate': TIME}
STYLE = {'font.size':9, 'axes.titlesize':9.5, 'axes.labelsize':9,
         'xtick.labelsize':8.5, 'ytick.labelsize':8.5, 'legend.fontsize':8,
         'pdf.fonttype':42, 'ps.fonttype':42, 'axes.spines.top':False,
         'axes.spines.right':False, 'axes.linewidth':.7, 'lines.linewidth':1.5}

def build(root: Path, output: Path):
    root, output = Path(root).resolve(), Path(output)
    inputs = {}
    def read(rel):
        p=root/rel; inputs[rel]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    ar=read('paper/iclr2027/generated/ar_evidence.json')
    budget=read('reports/external-baselines-20260906/timed-budget-decoding.json')
    # These registries are regenerated from raw requests by the manuscript audit.
    # Verify their stored raw hashes as well, making a standalone build fail stale.
    for entries in [ar['source_sha256'], budget['input_sha256']]:
        for name, expected in entries.items():
            p=Path(name) if Path(name).is_absolute() else root/name
            if hashlib.sha256(p.read_bytes()).hexdigest()!=expected:
                raise ValueError(f'Stale raw input: {name}')
            inputs[name]=expected
    figures=output/'figures'; figures.mkdir(parents=True,exist_ok=True)
    generated=output/'generated'; generated.mkdir(parents=True,exist_ok=True)
    plotted={}
    def save(fig,name):
        fig.savefig(figures/f'{name}.pdf',metadata=META)
        fig.savefig(figures/f'{name}.png',dpi=200)
        plt.close(fig)
    with plt.rc_context(STYLE):
        fig,axes=plt.subplots(1,2,figsize=(6.2,2.65),gridspec_kw={'width_ratios':[1.18,1]},layout='constrained')
        labels=[f"{r['family']} {r['target']}" for r in ar['main']]
        points=[]
        for i,r in enumerate(ar['main']):
            for key,label,color,marker,off in [('source','Source reuse',ORANGE,'s',.2),('relay','RelaySpec',BLUE,'o',0),('native','Native drafter',GREEN,'^',-.2)]:
                if r[key] is None: continue
                m=r['methods'][r[key]]; y=m['throughput_ratio'];lo,hi=m['throughput_ci95']
                axes[0].errorbar(y,i+off,xerr=[[y-lo],[hi-y]],fmt=marker,color=color,ms=4,capsize=2,label=label if i==0 else None)
                points.append({'family':r['family'],'target':r['target'],'method':key,'ratio':y,'ci95':[lo,hi],'tps':m['tokens_per_second']})
            relay=r['methods'][r['relay']]; source=r['methods'][r['source']]
            gain=r['secondary']['methods'][r['relay']]
            y=gain['throughput_ratio']; lo,hi=gain['throughput_ci95']
            retention=relay['progress_per_cycle']/source['progress_per_cycle']
            axes[1].plot([retention,y],[i,i],color='#c2c2c2',lw=1,zorder=1)
            axes[1].errorbar(y,i,xerr=[[y-lo],[hi-y]],fmt='o',color=BLUE,ms=4,capsize=2,label='Throughput' if i==0 else None)
            axes[1].plot(retention,i,'s',color=ORANGE,ms=4,label='Progress / cycle' if i==0 else None)
            points.append({'family':r['family'],'target':r['target'],'source_relative_throughput':y,'ci95':[lo,hi],'source_relative_progress':retention})
        for ax in axes:
            ax.set_yticks(range(4),labels if ax is axes[0] else [])
            ax.set_ylim(3.6,-1.2); ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
            ax.axvline(1,color=GRAY,ls='--',lw=.9)
        axes[0].set(xlim=(.8,6),xlabel='End-to-end throughput / matched AR',title='(a) Acceleration with a frozen drafter')
        axes[0].legend(loc='lower right',frameon=False,handlelength=1)
        axes[1].set(xlim=(.65,1.72),xlabel='RelaySpec / source reuse',title='(b) Less progress, faster decoding')
        axes[1].legend(loc='upper right',frameon=False,handlelength=1)
        save(fig,'visual_results_main'); plotted['primary']=points
        fig,axes=plt.subplots(1,2,figsize=(6.2,2.35),layout='constrained')
        records=[]
        for method,label,color,marker,ls in [('relay_feature','Feature MSE',BLUE,'o','-'),('relay_connector_ce','Connector CE',ORANGE,'s','--'),('relay_lora32_s1729','LoRA32, fit 1',GREEN,'^','-.'),('relay_lora32_s1730','LoRA32, fit 2',GRAY,'v',':')]:
            x=[];tps=[];ratio=[];low=[];high=[]
            for mult in (.25,1,4):
                fit=next(r for r in budget['fitting'] if r['method']==method and r['multiplier']==mult)
                assert fit['total_distinct_training_records']==512
                group=budget['matched_budget_against_feature'][str(mult)]
                assert group['requests']==16
                name=f'relay_budget{round(mult*100):03d}_'+method.removeprefix('relay_')
                row=group['methods'][name]
                x.append(fit['nominal_budget_seconds']);tps.append(row['tokens_per_second']);ratio.append(row['throughput_ratio']);low.append(row['throughput_ci95'][0]);high.append(row['throughput_ci95'][1])
            axes[0].plot(x,tps,color=color,marker=marker,ls=ls,ms=4,label=label)
            if method!='relay_feature':
                axes[1].errorbar(x,np.array(ratio)*100,yerr=[100*(np.array(ratio)-low),100*(np.array(high)-ratio)],color=color,marker=marker,ls=ls,ms=4,capsize=2)
            records.append({'method':method,'nominal_seconds':x,'end_to_end_tps':tps,'ratio_to_feature':ratio,'ci95':list(zip(low,high))})
        for ax in axes:
            ax.set_xscale('log',base=4);ax.set_xticks(x,['22.9','91.4','365.6']);ax.set_xlabel('Nominal warm-training budget (s)');ax.grid(alpha=.18);ax.set_axisbelow(True)
        axes[0].set(ylabel='End-to-end tokens/s',ylim=(98,157),title='(a) Feature fitting leads at every budget')
        axes[0].legend(frameon=False,ncol=2,loc='center right',fontsize=7.7,handlelength=1.5,columnspacing=.7)
        axes[1].axhline(100,color=BLUE,lw=1,ls='--');axes[1].text(x[0],98.1,'Feature parity',color=BLUE,fontsize=8,va='top')
        axes[1].set(ylabel='Throughput / feature MSE (%)',ylim=(65,102),title='(b) Paired intervals remain below parity')
        save(fig,'visual_results_budget');plotted['warm_budget']=records
        fig,axes=plt.subplots(1,2,figsize=(6.2,2.6),layout='constrained')
        blockpoints=[]
        for key,label,color,marker in [('native_target_dflash','Native',GREEN,'^'),('relay_f','RelaySpec',BLUE,'o'),('naive_source_reuse','Naive source reuse',ORANGE,'s')]:
            x=[r['block'] for r in ar['blocks']]; data=[r['methods'][key] for r in ar['blocks']]
            y=np.array([r['throughput_ratio'] for r in data]);ci=np.array([r['throughput_ci95'] for r in data])
            axes[0].errorbar(x,y,yerr=[y-ci[:,0],ci[:,1]-y],marker=marker,color=color,capsize=3,label=label)
            axes[1].plot(x,[r['progress_per_cycle'] for r in data],marker=marker,color=color,label=label)
            blockpoints.append({'method':key,'blocks':x,'values':data})
        for ax in axes:
            ax.set_xscale('log',base=2);ax.set_xticks(x,[str(n) for n in x]);ax.set_xlabel('Proposal block length');ax.grid(alpha=.18);ax.set_axisbelow(True)
        axes[0].set(ylabel='End-to-end throughput / matched AR',title='(a) A larger block can reduce speed')
        axes[1].set(ylabel='Recorded progress / cycle',title='(b) Relay progress falls at block 32')
        axes[0].set_ylim(1.7,6.8)
        axes[0].legend(frameon=False,loc='upper left')
        save(fig,'visual_results_blocks');plotted['block_size']=blockpoints
    evidence={'input_sha256':dict(sorted(inputs.items())),'plotted_values':plotted,'scope':{'primary':'Matched L40S BF16 MATH500 runs. Source, relay, native ratios use same-run AR. Source-relative progress is a descriptive point estimate, not a timed cycle decomposition. No native DFlash14B run.', 'warm_budget':budget['scope'],'block_size':'Separate DFlash8B development block screen, 64 requests. Naive source reuse control. Paired 95% request bootstrap intervals conditional on fitted models.'}}
    (generated/'results_visuals_evidence.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    return evidence
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--output',type=Path,default=Path('paper/iclr2027'));a=p.parse_args();build(a.root,a.output)
