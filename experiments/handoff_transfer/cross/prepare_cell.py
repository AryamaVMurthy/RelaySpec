"""Isolate full cross fits so CE/AUF never share mutable contracts or resumes."""
import argparse,json
from pathlib import Path
from experiments.handoff_transfer.matrix.prepare import digest
from experiments.handoff_transfer.matrix.task import KINDS


def prepare(initializers,index,out,kind,objective,lr):
    assert kind in KINDS and objective in ('ce','auf') and lr in (.0001,.0003,.0006)
    source=initializers/'transfer.json';transfer=json.loads(source.read_text())
    assert transfer['status']=='complete' and transfer['family']=='cross'
    assert transfer['records']==transfer['initializer_records']==4096
    assert transfer['target_adapters'] is None and transfer['full_data_index_sha256']==digest(index)
    if kind=='normal_ce':assert 'normal_export' in transfer and 'normal_sha256' in transfer
    contract=dict(family='cross',kind=kind,objective=objective,lr=lr,records=4096,
        global_batch=8,num_anchors=512,initializer_transfer_sha256=digest(source),
        full_data_index_sha256=digest(index),label_units='source tokens',context_units='target tokens')
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('transfer.json',transfer),('cell.json',contract)]:
        path=out/name
        if path.exists():assert json.loads(path.read_text())==data,'Cross cell identity changed'
        else:
            with path.open('x') as f:json.dump(data,f,indent=2)
    return contract


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ['initializers','index','out']:p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--kind',required=True);p.add_argument('--objective',required=True);p.add_argument('--lr',type=float,required=True)
    a=p.parse_args();prepare(a.initializers,a.index,a.out,a.kind,a.objective,a.lr)
