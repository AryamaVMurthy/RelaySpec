"""Native vLLM per-position counters; snapshots stay outside request timers."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import acceptance_metrics as base
scalar_snapshot=base.snapshot;scalar_delta=base.delta
NAME='vllm:spec_decode_num_accepted_tokens_per_pos'
def snapshot(engine):
 out=scalar_snapshot(engine);vectors=[m.values for m in engine.get_metrics() if m.name==NAME]
 assert vectors and all(len(v)==15 for v in vectors),'Missing native position counters'
 out['accepted_by_position']=[sum(v[k] for v in vectors) for k in range(15)]
 return out
def delta(before,after):
 out=scalar_delta(before,after);v=[b-a for a,b in zip(before['accepted_by_position'],after['accepted_by_position'])]
 assert len(v)==15 and all(x>=0 for x in v) and all(a>=b for a,b in zip(v,v[1:]))
 assert sum(v)==out['accepted_draft_tokens'] and v[0]<=out['verification_iterations']
 out['accepted_by_position']=v;return out
def install():base.snapshot=snapshot;base.delta=delta
if __name__=='__main__':
 # Accepted lengths0,1,3,15: sum of survival counts equals total accepted.
 v=[sum(n>k for n in [0,1,3,15]) for k in range(15)]
 b={k:0 for k in base.FIELDS.values()};b['accepted_by_position']=[0]*15
 a={'verification_iterations':4,'accepted_draft_tokens':19,'proposed_draft_tokens':60,'accepted_by_position':v}
 assert delta(b,a)['accepted_by_position']==v
 print('Position-counter synthetic identity passed')
