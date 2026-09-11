"""Repack verified dense target captures into the handoff's32-record mmap shards."""
import argparse,hashlib,json,os
from pathlib import Path
import torch

def sha(p):
    digest=hashlib.sha256()
    with p.open('rb') as stream:
        for chunk in iter(lambda:stream.read(16*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

def write(p,data):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2)+'\n');tmp.replace(p)

def main(a):
    width={'q8':20480,'q14':25600,'llama':15360}[a.family]
    name={'q8':'main-q8-n4096','q14':'main-q14-n4096','llama':'main-l3-n4096'}[a.family]
    data=a.study/name
    manifests=[data/'train.json'] if (data/'train.json').exists() else sorted(data.glob('part-*/train.json'),key=lambda p:int(p.parent.name.split('-')[-1]))
    items=[];hashes={}
    for p in manifests:
        hashes[str(p)]=sha(p)
        for i,row in enumerate(json.loads(p.read_text())):
            role='8' if a.family=='q8' else 'target'
            items.append((p,p.parent/f'features/{role}/train/{i:05d}.pt',row))
    assert len(items)==4096 and len({r['group_id'] for _,_,r in items})==4096
    a.out.mkdir(parents=True,exist_ok=True)
    shard_hashes={};checked=0
    for start in range(0,4096,32):
        dest=a.out/f'features/full/{start:05d}.pt';meta=dest.with_suffix('.json')
        selected=items[start:start+32]
        if dest.exists() and meta.exists():
            existing=json.loads(meta.read_text())
            assert existing['manifests']==hashes and existing['sha256']==sha(dest)
            shard_hashes[str(dest)]=existing['sha256'];continue
        features=[];rows=[];sources=[]
        for manifest,path,row in selected:
            capture=json.loads(path.with_suffix('.json').read_text())
            assert capture['manifest_sha256']==hashes[str(manifest)]
            assert capture['sha256']==sha(path)
            value=torch.load(path,weights_only=True,mmap=True,map_location='cpu')
            assert value['group_id']==row['group_id']
            assert row['full_ids']==row['prompt_token_ids']+row['output_ids']
            h=value['features']
            assert h.shape==(len(row['full_ids']),width) and h.dtype==torch.bfloat16
            assert h.is_contiguous() and torch.isfinite(h).all()
            if h.untyped_storage().nbytes()!=h.numel()*h.element_size():h=h.clone()
            features.append(h);rows.append(row);sources.append({'path':str(path),'sha256':capture['sha256']})
            checked+=path.stat().st_size
        dest.parent.mkdir(parents=True,exist_ok=True)
        temporary=dest.with_suffix('.part');torch.save({'rows':rows,'features':features},temporary);temporary.replace(dest)
        digest=sha(dest);shard_hashes[str(dest)]=digest
        write(meta,{'sha256':digest,'manifests':hashes,'sources':sources,'records':32,'input_width':width})
        print(json.dumps({'records_packed':start+32,'bytes_verified':checked}),flush=True)
    fit=a.study/f'fits/{a.family}-n4096-zip-lr1e-3-s42/epoch-3/export'
    assert (fit/'model.safetensors').is_file()
    if a.family=='q8':
        base=Path('/scratch/aryama.murthy/transfer-reproduction-20260907/work/models/4b')
        source,draft=base/'target',base/'draft'
    else:
        source=a.study/'models'/('qwen4-source' if a.family=='q14' else 'llama8-source')
        draft=a.study/'models'/('qwen4-draft' if a.family=='q14' else 'llama8-draft')
    config={'family':a.family,'input_width':width,'source':str(source),'draft':str(draft),'base_export':str(fit),
            'base_sha256':sha(fit/'model.safetensors'),'records':4096,'base_training_epochs':3,
            'target_adapters':None,'feature_manifests':hashes,'shards':shard_hashes,'status':'complete'}
    write(a.out/'transfer.json',config)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--family',choices=['q8','q14','llama'],required=True)
    p.add_argument('--study',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    main(p.parse_args())
