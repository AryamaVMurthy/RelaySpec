"""Fresh matched-runtime AR plus six existing speculative checkpoints."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from experiments.auf_vllm.compare_outputs import compare
from .single_request import ROOT, STUDY, CAMPAIGN, LABELS

NEW_CAMPAIGN=os.environ.get('RELAYSPEC_EVAL_CAMPAIGN','optimized-all128-20260912')


def read(path):return json.loads(path.read_text())


def run(pilot=False):
    import torch
    uuid=str(torch.cuda.get_device_properties(0).uuid)
    task=int(os.environ['SLURM_ARRAY_TASK_ID'])
    cases=[('q8','native'),('q8','original'),('cross','five_maps-auf'),('cross','dense_fusion-ce')]
    if os.environ.get('RELAYSPEC_NUMERICS'):
        cases=[('q8','native'),('q8','dense_fusion-auf'),('cross','native'),('cross','five_maps-auf')]
    pilot_count=int(os.environ.get('RELAYSPEC_PILOT_COUNT','2'))
    pilot_cap=int(os.environ.get('RELAYSPEC_PILOT_CAP','128'))
    for family in ([cases[task][0]] if pilot else ['q8','cross']):
        matches=[]
        for worker in range(4):
            prior=ROOT/CAMPAIGN/family/f'worker-{worker}'
            if read(prior/'ar'/f'ar-r0-w{worker}.summary.json')['gpu_after'][0]['device_uuid']==uuid:
                matches.append(worker)
        assert len(matches)==1,(uuid,matches)
        worker=matches[0]
        base=ROOT/'exact32e1b8-results'/family
        target=(ROOT.parent/'transfer-reproduction-20260907/work/models/8b/target'
                if family=='q8' else STUDY/'models/llama8-source')
        initial=ROOT/'q8' if family=='q8' else ROOT/'cross-full4096/initializers-e1'
        root=ROOT/NEW_CAMPAIGN
        if pilot:root=root/'pilot'/f'task-{task}'
        out=root/family/f'worker-{worker}';out.mkdir(parents=True,exist_ok=True)
        data=base/'evaluation';count=128;cap=2048;workers=4;index=worker
        if pilot:
            data=out/'data';data.mkdir(exist_ok=True)
            (data/'eval.json').write_text(json.dumps(read(base/'evaluation/eval.json')[:pilot_count]))
            (data/'warmup.json').write_text(json.dumps(read(base/'evaluation/warmup.json')[:1]))
            count=pilot_count;cap=pilot_cap;workers=1;index=0
        if family=='cross':os.environ['CROSS_SOURCE_TOKENIZER']=str(STUDY/'models/qwen4-source')
        arms=LABELS[1:];offset=worker%len(arms)
        order=('ar',cases[task][1]) if pilot else ('ar',*arms[offset:],*arms[:offset])
        paths={};progress=[];effective=None
        for label in order:
            mode=label if label in ('ar','native') else 'matrix'
            extra=[];expected=None
            if label=='native':
                native=ROOT.parent/'transfer-reproduction-20260907/work/models/8b/draft' if family=='q8' else STUDY/'models/llama8-draft'
                extra=['--native-draft',native]
            elif label=='original':
                extra=['--export',initial/'normal/export']
                expected=read(base/'original/matrix-r0-w0-b128.summary.json')['contract']['export_sha256']
            elif label!='ar':
                kind,objective=label.rsplit('-',1)
                extra=['--export',ROOT/'matrix32e1b8'/family/f'{kind}-{objective}-lr0.0001/exports/steps-512/{kind}']
                expected=read(base/f'{kind}-{objective}-lr0.0001/matrix-r0-w0-b128.summary.json')['contract']['export_sha256']
            command=[sys.executable,'-u','-m','experiments.auf_vllm.family_benchmark',
                     '--family',family,'--models',STUDY/'models','--target-path',target,
                     '--data',data,'--out',out/label,'--mode',mode,'--count',count,'--cap',cap,
                     '--workers',workers,'--worker-index',index,'--request-batch-size',1,
                     '--runtime-profile',('numerics' if os.environ.get('RELAYSPEC_NUMERICS') else 'optimized'),*extra]
            print(json.dumps(dict(family=family,worker=worker,label=label,pilot=pilot,uuid=uuid)),flush=True)
            subprocess.run(list(map(str,command)),check=True)
            path=out/label/f'{mode}-r0-w{index}.jsonl';paths[label]=path
            s=read(path.with_suffix('.summary.json'))
            assert s['contract']['export_sha256']==expected
            assert s['gpu_before'][0]['device_uuid']==s['gpu_after'][0]['device_uuid']==uuid
            if effective is None:effective=s['effective_runtime']
            assert s['effective_runtime']==effective,(label,effective,s['effective_runtime'])
            compared=compare([paths['ar']],[path]);assert compared['count']==(pilot_count if pilot else 32)
            # Numerical divergence is recorded; never relabel non-identical output as lossless.
            progress.append(dict(label=label,**compared))
            (out/'progress.json').write_text(json.dumps(progress,indent=2)+'\n')
            print(json.dumps(progress[-1]),flush=True)
            if os.environ.get('RELAYSPEC_REQUIRE_EXACT')=='1':
                assert compared['exact_matches']==compared['count'] and compared['finish_matches']==compared['count'],compared
        (out/'complete.json').write_text(json.dumps(dict(status='complete',family=family,worker=worker,
            uuid=uuid,pilot=pilot,job_id=os.environ['SLURM_JOB_ID'],effective_runtime=effective,
            comparisons=progress),indent=2)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--pilot',action='store_true');a=p.parse_args();run(a.pilot)
