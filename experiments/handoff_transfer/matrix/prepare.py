"""Create an isolated cell; never overwrite baseline artifacts or share resumes."""
import argparse,hashlib,json
from pathlib import Path

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def prepare(base,out,kind,objective,lr):
    transfer=json.loads((base/'transfer.json').read_text())
    assert transfer['status']=='complete' and transfer['records']==4096
    if kind=='normal_ce':
        normal=base/'normal/export'
        transfer.update(normal_export=str(normal),normal_sha256=digest(normal/'model.safetensors'))
    contract={'family':transfer['family'],'kind':kind,'objective':objective,'lr':lr,
        'num_anchors':512,'objective_chunk_blocks':16,'records':4096,'global_batch':8,
        'transfer_source_sha256':digest(base/'transfer.json')}
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('transfer.json',transfer),('cell.json',contract)]:
        dest=out/name
        if dest.exists():assert json.loads(dest.read_text())==data,'Cell identity changed'
        else:dest.write_text(json.dumps(data,indent=2)+'\n')
    features=out/'features';features.mkdir(exist_ok=True)
    full=features/'full'
    if full.is_symlink():assert full.resolve()==(base/'features/full').resolve()
    else:full.symlink_to(base/'features/full',target_is_directory=True)
    check=features/'check';check.mkdir(exist_ok=True)
    first=sorted((base/'features/full').glob('*.pt'))[0]
    dest=check/'00000.pt'
    if dest.is_symlink():assert dest.resolve()==first.resolve()
    else:dest.symlink_to(first)
    return contract

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--kind',required=True);p.add_argument('--objective',choices=['ce','auf','decay'],required=True)
    p.add_argument('--lr',type=float,required=True);a=p.parse_args();prepare(a.base,a.out,a.kind,a.objective,a.lr)
