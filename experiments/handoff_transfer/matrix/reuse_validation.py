"""Reuse a measured tuning AR reference without inventing timing repetitions."""
import argparse,json,shutil
from pathlib import Path
from .prepare import digest


def validate(source,data,family,target):
    summary=source.with_suffix('.summary.json')
    measured=json.loads(summary.read_text());c=measured['contract']
    assert measured['timing_valid'] and c['mode']=='ar' and c['family']==family
    assert c['count']==32 and c['cap']==512 and c['repeat']==0
    assert c['workers']==1 and c['worker_index']==0 and c['export_sha256'] is None
    assert c['runtime_config']['model']==str(target)
    assert c['manifest_sha256']==digest(data/'eval.json')
    expected={r['group_id']:r for r in json.loads((data/'eval.json').read_text())}
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows)==len(expected)==32 and len({r['group_id'] for r in rows})==32
    for row in rows:
        assert row['prompt_ids']==expected[row['group_id']]['prompt_token_ids']
        assert row['timing_valid'] and row['wall_seconds']>0
        assert 0<row['output_tokens']==len(row['output_ids'])<=512
        assert row['finish_reason'] in ('stop','length')
    return {str(p):digest(p) for p in (source,summary)}


def reuse(source,data,dest,family,target):
    dest.mkdir(parents=True,exist_ok=True);output=dest/'ar-r0-w0.jsonl'
    if output.exists() and output.with_suffix('.summary.json').exists():
        return {'status':'retained_existing_measurement','sources':validate(output,data,family,target)}
    assert not output.exists() and not output.with_suffix('.summary.json').exists(), 'Partial AR output requires explicit recovery'
    hashes=validate(source,data,family,target)
    for original,copy in [(source,output),(source.with_suffix('.summary.json'),output.with_suffix('.summary.json'))]:
        shutil.copyfile(original,copy)
    provenance={'status':'reused_tuning_reference','sources':hashes,
        'counts_as_new_timing_repetitions':0,'scope':'32/cap512 tuning only; final repetitions remain separate'}
    (dest/'ar-reuse.json').write_text(json.dumps(provenance,indent=2)+'\n')
    return provenance


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['source','data','dest','target']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--family',choices=['q8','llama','cross'],required=True)
    a=p.parse_args();print(json.dumps(reuse(a.source,a.data,a.dest,a.family,a.target)))
