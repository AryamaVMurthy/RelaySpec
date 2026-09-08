"""Generate final comparison plots only after complete frozen coverage."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
data=json.loads((ROOT/'reports/confirmation-summary.json').read_text());assert data['status']=='complete'
rows=data['rows']
methods=['native','compact_linear','candidate','reference']
labels=['Original DFlash','Joint compact, linear','Joint compact + DDTree47','Full DDTree63']
fig,axes=plt.subplots(1,2,figsize=(11,3.5),layout='constrained',sharey=True)
for ax,reference,title in zip(axes,['native','reference'],['Relative to original DFlash','Relative to full DDTree']):
    for i,m in enumerate(methods):
        r=next(r for r in rows if r['benchmark']=='all' and r['method']==m)['comparisons'][reference]
        ax.errorbar(r['ratio'],i,xerr=[[r['ratio']-r['ci95'][0]],[r['ci95'][1]-r['ratio']]],fmt='o',capsize=3)
    ax.axvline(1,color='black',lw=1);ax.axvline(1.1,color='green',ls='--',label='10% gain')
    ax.set_title(title);ax.set_xlabel('End-to-end throughput ratio');ax.grid(axis='x',alpha=.2)
axes[0].set_yticks(range(4),labels);axes[0].invert_yaxis();axes[1].legend(loc='best')
fig.savefig(ROOT/'reports/confirmation-ratios.png',dpi=180);fig.savefig(ROOT/'reports/confirmation-ratios.pdf')
benchmarks=['gsm8k','math500','humaneval','mtbench']
matrix=np.array([[next(r for r in rows if r['benchmark']==b and r['method']==m)['comparisons']['native']['ratio'] for b in benchmarks] for m in methods])
fig,ax=plt.subplots(figsize=(8,3.5),layout='constrained');im=ax.imshow(matrix,cmap='RdYlGn',vmin=.85,vmax=1.35,aspect='auto')
ax.set_xticks(range(4),['GSM8K','MATH500','HumanEval','MTBench']);ax.set_yticks(range(4),labels)
for i in range(4):
    for j in range(4):ax.text(j,i,f'{matrix[i,j]:.3f}×',ha='center',va='center')
fig.colorbar(im,ax=ax,label='Throughput / original DFlash');fig.savefig(ROOT/'reports/confirmation-workloads.png',dpi=180);fig.savefig(ROOT/'reports/confirmation-workloads.pdf')
fig,ax=plt.subplots(figsize=(9,3.5),layout='constrained')
for shift,key,label in [(-.2,'exact_native','Exact original DFlash'),(.2,'exact_ar','Exact AR')]:
    values=[next(r for r in rows if r['benchmark']=='all' and r['method']==m)[key] for m in methods]
    bars=ax.bar(np.arange(4)+shift,values,width=.4,label=label);ax.bar_label(bars)
ax.set_xticks(range(4),labels,rotation=12,ha='right');ax.set_ylim(0,140);ax.set_ylabel('Identical output sequences /128');ax.legend();fig.savefig(ROOT/'reports/confirmation-agreement.png',dpi=180);fig.savefig(ROOT/'reports/confirmation-agreement.pdf')
print('Saved confirmation plots')
