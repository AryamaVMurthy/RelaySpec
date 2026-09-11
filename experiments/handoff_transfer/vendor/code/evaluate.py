"""Portable wrapper around the original benchmark; custom paths only."""
from runtime import *
import argparse
import benchmark_safe_r3 as b
import position_metrics
position_metrics.install()
p=argparse.ArgumentParser();p.add_argument('--variant',choices=['ar','native','mapper'],required=True);p.add_argument('--count',type=int,default=128);p.add_argument('--worker',type=int,default=0);p.add_argument('--workers',type=int,default=1);p.add_argument('--reference-file',required=True);a=p.parse_args()
refs={x['group_id']:x for x in json.loads(Path(a.reference_file).read_text())}
def engine(*args,**kw):
 kw['draft']=R/'exports/full/fusion_r56' if a.variant=='mapper' else PAIR/'draft'
 return llm(*args,**kw)
b.R=R;b.llm=engine;b.request=request;b.rows=lambda split:json.loads((R/'setup'/f'{split}.json').read_text())
original=b.put
def audit(path,z):
 original(path,z)
 for x in z.get('rows',[]):
  y=refs[x['group_id']]
  assert (x['output_ids'],x['finish_reason'])==(y['output_ids'],y['finish_reason']),('STOP: reference disagreement',x['group_id'])
b.put=audit
a.side={'ar':'ar','native':'native','mapper':'mapped'}[a.variant];a.tag='full';a.results_tag=a.variant;a.cap=8192 if DOMAIN=='kicad' else 2048
b.bench(a)
