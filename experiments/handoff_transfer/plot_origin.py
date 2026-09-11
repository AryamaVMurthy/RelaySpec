"""Descriptive figure for the verified original-package reproduction only."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parent
report=json.loads((root/'reports/origin-full-comparison.json').read_text())
assert report['status']=='verified_complete'
fig,axes=plt.subplots(1,2,figsize=(8.2,3.3))
labels=['AR','Native drafter','AUF fusion']
values=[report['rows'][m]['tps'] for m in ['ar','native','mapper']]
axes[0].bar(labels,values,color=['#999999','#4477aa','#228833'])
for i,v in enumerate(values):axes[0].text(i,v+3,f'{v:.1f}',ha='center',fontsize=10)
axes[0].set(ylabel='Output tokens / second',ylim=(0,180),title='Matched inference throughput')
progress=[report['rows'][m]['accepted_draft_tokens_per_verification'] for m in ['native','mapper']]
axes[1].bar(labels[1:],progress,color=['#4477aa','#228833'])
for i,v in enumerate(progress):axes[1].text(i,v+.08,f'{v:.2f}',ha='center',fontsize=10)
axes[1].set(ylabel='Accepted draft tokens / verification',ylim=(0,5.5),title='Accepted progress')
for ax in axes:ax.spines[['top','right']].set_visible(False);ax.set_axisbelow(True);ax.grid(axis='y',alpha=.2)
fig.suptitle('Original handoff Math reproduction — frozen target LoRA',fontsize=12)
fig.text(.5,.015,'128 requests; max 2,048 tokens, natural EOS; one training seed and timing run. Not a model-transfer result.',ha='center',fontsize=8)
fig.tight_layout(rect=(0,.045,1,.93))
out=root/'figures';out.mkdir(exist_ok=True)
for suffix in ['png','pdf']:fig.savefig(out/f'origin_reproduction.{suffix}',dpi=180,bbox_inches='tight')
