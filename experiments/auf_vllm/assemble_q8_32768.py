"""Join verified dense target captures and preserve the ZIP quarter-position cache."""
import argparse
import json
import os
from pathlib import Path
from .pilot_data import rows,sha,write
from .runtime_zip.sampling import positions


def link(source,destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():assert destination.resolve()==source.resolve()
    else:destination.symlink_to(source.resolve(),target_is_directory=source.is_dir())


def main(args):
    import torch
    old=json.loads((args.old/'train.json').read_text())
    assert len(old)==16384
    extended=rows(args.manifest)
    assert len(extended)==32768 and [x['group_id'] for x in old]==[x['group_id'] for x in extended[:16384]]
    all_rows=list(old);parts=[]
    for shard in range(64):
        root=args.extra/f'part-{shard:05d}'
        report=json.loads((root/'generation.json').read_text())
        assert report['sha256']==sha(root/'train.json')
        data=json.loads((root/'train.json').read_text());assert len(data)==256
        assert [r['group_id'] for r in data]==[r['group_id'] for r in extended[16384+256*shard:16384+256*(shard+1)]]
        for size in [8,4]:
            completed=json.loads((root/f'capture-{size}-0-256.json').read_text())
            assert completed['status']=='complete' and len(completed['records'])==256
            assert completed['manifest_sha256']==report['sha256']
        all_rows.extend(data);parts.append((root,data,report['sha256']))
    assert len({r['group_id'] for r in all_rows})==32768
    args.out.mkdir(parents=True,exist_ok=True)
    if (args.out/'train.json').exists():assert json.loads((args.out/'train.json').read_text())==all_rows
    else:write(args.out/'train.json',all_rows)
    new_hash=sha(args.out/'train.json');old_hash=sha(args.old/'train.json')
    write(args.out/'dev.json',[])
    link(args.work/'models',args.out/'models')
    for split in ['dev','eval']:link(args.work/f'common/dataset/{split}.jsonl',args.out/f'common/dataset/{split}.jsonl')
    link(args.manifest,args.out/'common/dataset/train.jsonl')
    for path in sorted((args.work/'common/rollouts/train').glob('*')):
        link(path,args.out/'common/rollouts/train'/path.name)
    for size in [8,4]:
        for path in sorted((args.work/f'cache/features/train/{size}').glob('*')):
            link(path,args.out/f'cache/features/train/{size}'/path.name)
    old_cache=json.loads((args.work/'cache/manifest.json').read_text())
    assert old_cache['complete'] and old_cache['examples']==16384
    total_positions=old_cache['positions']
    checked_bytes=0
    for index,row in enumerate(old):
        source=args.old/f'features/8/train/{index:05d}.pt'
        meta=json.loads(source.with_suffix('.json').read_text())
        assert meta['manifest_sha256']==old_hash and meta['sha256']==sha(source)
        assert meta['tokens']==len(row['full_ids'])
        dest=args.out/f'features/8/train/{index:05d}.pt';link(source,dest)
        write(dest.with_suffix('.json'),dict(meta,manifest_sha256=new_hash,reused_source=str(source),reused_manifest_sha256=old_hash))
        checked_bytes+=source.stat().st_size
        if (index+1)%1024==0:print(json.dumps({'old_dense_verified':index+1,'bytes_checked':checked_bytes}),flush=True)
    for shard,(root,data,manifest_hash) in enumerate(parts):
        for half in range(2):
            subset=data[half*128:(half+1)*128]
            number=128+shard*2+half
            rollout=args.out/f'common/rollouts/train/{number:05d}.jsonl'
            serialized=''.join(json.dumps(r)+'\n' for r in subset)
            if rollout.exists():assert rollout.read_text()==serialized
            else:rollout.write_text(serialized)
            indices=[torch.tensor(positions(len(r['full_ids']),len(r['prompt_token_ids']),r['group_id'])) for r in subset]
            total_positions+=sum(len(x) for x in indices)
            for size,width in [(8,4096),(4,2560)]:
                sampled=[]
                for offset,(row,ix) in enumerate(zip(subset,indices)):
                    local=half*128+offset
                    source=root/f'features/{size}/train/{local:05d}.pt'
                    meta=json.loads(source.with_suffix('.json').read_text())
                    assert meta['manifest_sha256']==manifest_hash and meta['sha256']==sha(source)
                    tensors=torch.load(source,map_location='cpu',weights_only=True)
                    assert tensors['group_id']==row['group_id'] and tensors['layers']==[1,9,17,25,33]
                    features=tensors['features']
                    assert features.shape==(len(row['full_ids']),5*width) and features.dtype==torch.bfloat16
                    assert torch.isfinite(features).all()
                    sampled.append(features[ix].contiguous())
                    if size==8:
                        global_index=16384+shard*256+local
                        dest=args.out/f'features/8/train/{global_index:05d}.pt';link(source,dest)
                        write(dest.with_suffix('.json'),dict(meta,manifest_sha256=new_hash,reused_source=str(source),reused_manifest_sha256=manifest_hash))
                    checked_bytes+=source.stat().st_size
                cache={'features':sampled,'positions':indices,'group_ids':[r['group_id'] for r in subset],
                       'prompt_lengths':[len(r['prompt_token_ids']) for r in subset],'rollout_sha256':sha(rollout)}
                dest=args.out/f'cache/features/train/{size}/{number:05d}.pt'
                dest.parent.mkdir(parents=True,exist_ok=True)
                temporary=dest.with_suffix('.part');torch.save(cache,temporary);temporary.replace(dest)
                write(dest.with_suffix('.meta.json'),{'sha256':sha(dest),'rollout_sha256':sha(rollout),'size':size,
                      'positions':sum(len(x) for x in indices),'records':128,'job_id':os.environ.get('SLURM_JOB_ID')})
                del sampled,cache,features,tensors
        print(json.dumps({'new_dense_and_paired_records':(shard+1)*256,'bytes_checked':checked_bytes}),flush=True)
    write(args.out/'cache/manifest.json',{'complete':True,'examples':32768,'positions':total_positions,'layers':[1,9,17,25,33]})
    write(args.out/'assembly.json',{'records':32768,'manifest_sha256':new_hash,'dataset_manifest_sha256':sha(args.manifest),
          'dense_target_records':32768,'paired_quarter_records':32768,'bytes_checked':checked_bytes,
          'preserved_old_quarter_cache':True,'sampling':'original deterministic25% prompt/response-stratified positions',
          'job_id':os.environ.get('SLURM_JOB_ID'),'status':'complete'})


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ['old','extra','work','manifest','out']:p.add_argument('--'+name,type=Path,required=True)
    main(p.parse_args())
