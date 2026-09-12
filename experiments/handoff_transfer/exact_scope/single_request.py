"""Seven existing checkpoints/controls, paired single-request evaluations."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.auf_vllm.compare_outputs import compare

ROOT=Path('/scratch/aryama.murthy/handoff-transfer-20260911')
STUDY=ROOT.parent/'relayspec-auf-20260911'
CAMPAIGN='single128e1-20260912'
LABELS=('ar','native','five_maps-auf','five_maps-ce','dense_fusion-auf','dense_fusion-ce','original')


def read(path):
    return json.loads(path.read_text())


def run(family,worker):
    original=ROOT/'exact32e1b8-results'/family
    target=(ROOT.parent/'transfer-reproduction-20260907/work/models/8b/target'
            if family=='q8' else STUDY/'models/llama8-source')
    native=(ROOT.parent/'transfer-reproduction-20260907/work/models/8b/draft'
            if family=='q8' else STUDY/'models/llama8-draft')
    initial=ROOT/'q8' if family=='q8' else ROOT/'cross-full4096/initializers-e1'
    out=ROOT/CAMPAIGN/family/f'worker-{worker}'
    out.mkdir(parents=True,exist_ok=True)
    if family=='cross':os.environ['CROSS_SOURCE_TOKENIZER']=str(STUDY/'models/qwen4-source')
    paths={};rows=[];device=None
    # AR establishes the within-worker reference. Rotate the six non-AR arms
    # across shards so every method is not always measured in the same order.
    arms=LABELS[1:];offset=worker%len(arms)
    order=('ar',*arms[offset:],*arms[:offset])
    for label in order:
        mode=label if label in ('ar','native') else 'matrix'
        extra=[]
        expected_export=None
        if label=='native':extra=['--native-draft',native]
        elif label=='original':
            extra=['--export',initial/'normal/export']
            expected_export=read(original/'original/matrix-r0-w0-b128.summary.json')['contract']['export_sha256']
        elif label!='ar':
            kind,objective=label.rsplit('-',1)
            export=ROOT/'matrix32e1b8'/family/f'{kind}-{objective}-lr0.0001/exports/steps-512/{kind}'
            extra=['--export',export]
            expected_export=read(original/f'{kind}-{objective}-lr0.0001/matrix-r0-w0-b128.summary.json')['contract']['export_sha256']
        dest=out/label
        command=[sys.executable,'-u','-m','experiments.auf_vllm.family_benchmark',
            '--family',family,'--models',STUDY/'models','--target-path',target,
            '--data',original/'evaluation','--out',dest,'--mode',mode,
            '--count',128,'--cap',2048,'--repeat',0,'--request-batch-size',1,
            '--workers',4,'--worker-index',worker,*extra]
        subprocess.run(list(map(str,command)),check=True)
        path=dest/f'{mode}-r0-w{worker}.jsonl';paths[label]=path
        summary=read(path.with_suffix('.summary.json'))
        contract=summary['contract']
        assert contract['count']==128 and contract['cap']==2048 and contract['workers']==4
        assert contract['worker_index']==worker and contract.get('request_batch_size',1)==1
        assert contract['export_sha256']==expected_export
        uuid=summary['gpu_after'][0]['device_uuid']
        assert summary['gpu_before'][0]['device_uuid']==uuid
        if device is None:device=uuid
        assert uuid==device,'Paired methods must use one physical GPU per shard'
        measured=compare([paths['ar']],[path])
        rows.append(dict(label=label,device_uuid=uuid,**measured))
        (out/'progress.json').write_text(json.dumps(rows,indent=2)+'\n')
        print(json.dumps(dict(family=family,worker=worker,**rows[-1])),flush=True)
        if not measured['count']==measured['exact_matches']==measured['finish_matches']==32:
            raise ValueError(f'Single-request AR disagreement: {family}/{worker}/{label}; saved in progress.json')
    (out/'complete.json').write_text(json.dumps(dict(status='complete',family=family,worker=worker,
        requests=32,total_cohort=128,cap=2048,batch=1,job_id=os.environ['SLURM_JOB_ID'],
        device_uuid=device,labels=list(LABELS),comparisons=rows),indent=2)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--family',choices=['q8','cross'],required=True)
    parser.add_argument('--worker',type=int,choices=range(4),required=True)
    args=parser.parse_args();run(args.family,args.worker)
