import json
import os
from pathlib import Path

root=Path(os.environ['FAMILY_SCALE_CACHE'])/'shared-pilot'
events=[]
for lane in range(4):
    results=json.loads((root/f'lane{lane}'/'results.json').read_text())
    assert len(results)==2 and all(r['exact_gate'] for r in results)
    standalone=Path('outputs/28947')/f'lane{lane}'/'benchmark-rank0.jsonl'
    shared=Path(results[1]['output'])/'benchmark-rank0.jsonl'
    def read(path):
        return {(r['problem_id'],r['method']):r for r in
            [json.loads(s) for s in path.read_text().splitlines()] if 'method' in r}
    old,new=read(standalone),read(shared)
    assert old.keys()==new.keys()
    for key in old:
        assert old[key]['output_hash']==new[key]['output_hash'],key
        assert old[key]['acceptance_lengths']==new[key]['acceptance_lengths'],key
    events.append({'lane':lane,'identical_output_hashes_and_acceptance_trajectories':True})
(root/'equivalence-gate.json').write_text(json.dumps({'status':'pass','results':events},indent=2))
