"""Reuse verified4096 dense captures inside the canonical16384-rollout manifest."""
import argparse
import json
import os
from pathlib import Path
from .pilot_data import rows,sha,write


def main(args):
    examples=[];sources=[]
    for path in sorted((args.work/'common/rollouts/train').glob('*.jsonl')):
        examples.extend(rows(path));sources.append({'path':str(path),'sha256':sha(path)})
    assert len(examples)==16384
    canonical=rows(args.work/'common/dataset/train.jsonl')
    assert [x['group_id'] for x in examples]==[x['group_id'] for x in canonical]
    assert len({x['group_id'] for x in examples})==16384
    assert all(x['full_ids']==x['prompt_token_ids']+x['output_ids'] and x['temperature']==0 for x in examples)
    old=json.loads((args.small/'train.json').read_text())
    assert len(old)==4096
    assert all(a['group_id']==b['group_id'] and a['full_ids']==b['full_ids'] for a,b in zip(old,examples))
    args.out.mkdir(parents=True,exist_ok=True)
    manifest=args.out/'train.json'
    if manifest.exists():assert json.loads(manifest.read_text())==examples
    else:write(manifest,examples)
    write(args.out/'dev.json',[])
    before_hash=sha(args.small/'train.json');after_hash=sha(manifest)
    total_bytes=0
    for index,row in enumerate(old):
        source=args.small/f'features/8/train/{index:05d}.pt'
        meta=json.loads(source.with_suffix('.json').read_text())
        assert meta['manifest_sha256']==before_hash
        assert meta['sha256']==sha(source)
        assert meta['tokens']==len(row['full_ids'])
        destination=args.out/f'features/8/train/{index:05d}.pt'
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():assert destination.resolve()==source.resolve()
        else:destination.symlink_to(source.resolve())
        write(destination.with_suffix('.json'),dict(meta,manifest_sha256=after_hash,
              reused_source=str(source),reused_manifest_sha256=before_hash,
              reuse_verification='Full rollout token identity and SHA256 of captured tensor',
              reuse_job_id=os.environ.get('SLURM_JOB_ID')))
        total_bytes+=source.stat().st_size
        if (index+1)%512==0:print(json.dumps({'reused_verified':index+1,'bytes_checked':total_bytes}),flush=True)
    write(args.out/'preparation.json',{'records':16384,'sources':sources,'train_sha256':after_hash,
          'reused_dense_records':4096,'remaining_dense_records':12288,'reused_bytes_checked':total_bytes,
          'target_adapters':None,'job_id':os.environ.get('SLURM_JOB_ID'),
          'status':'manifest_prepared_first4096_verified_remaining_capture_required'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--work',type=Path,required=True)
    p.add_argument('--small',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
