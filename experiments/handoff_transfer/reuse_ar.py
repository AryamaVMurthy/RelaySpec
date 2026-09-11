"""Reuse one completed AR repetition only under an identical measured contract."""
import argparse,hashlib,json,shutil
from pathlib import Path

def reuse(source,gate,evaluation,dest):
    summary_path=source.with_suffix('.summary.json')
    measured=json.loads(summary_path.read_text())
    expected=dict(json.loads(gate.read_text())['contract']);expected.update(count=128,cap=2048,repeat=0)
    assert expected['mode']=='ar' and measured['contract']==expected,'AR measurement contract differs'
    assert measured['timing_valid']
    assert measured['contract']['manifest_sha256']==hashlib.sha256(evaluation.read_bytes()).hexdigest()
    required={r['group_id']:r for r in json.loads(evaluation.read_text())}
    rows=[json.loads(line) for line in source.read_text().splitlines()]
    assert len(rows)==len(required)==128 and len({r['group_id'] for r in rows})==128
    for r in rows:
        assert r['group_id'] in required and r['prompt_ids']==required[r['group_id']]['prompt_token_ids']
        assert r['timing_valid'] and r['wall_seconds']>0
        assert r['output_tokens']==len(r['output_ids'])<=2048
        assert r['finish_reason'] in ['stop','length']
    dest.mkdir(parents=True,exist_ok=True)
    hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,summary_path,gate,evaluation]}
    for p in [source,summary_path]:
        output=dest/p.name
        if output.exists():assert output.read_bytes()==p.read_bytes(),'Refuse to overwrite an existing measurement'
        else:
            temporary=output.with_suffix(output.suffix+'.copying');shutil.copyfile(p,temporary);temporary.replace(output)
    (dest/'ar-r0-reuse-provenance.json').write_text(json.dumps({'status':'verified_reuse','source_sha256':hashes,
        'counts_as_timing_repetitions':1,'remaining_fresh_repetitions':[1,2],
        'note':'Original measured rows, job ID and summary retained verbatim. Not a new timing measurement.'},indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--gate',type=Path,required=True)
    p.add_argument('--evaluation',type=Path,required=True);p.add_argument('--dest',type=Path,required=True);a=p.parse_args()
    reuse(a.source,a.gate,a.evaluation,a.dest);print('VERIFIED_AR_REUSE',flush=True)
