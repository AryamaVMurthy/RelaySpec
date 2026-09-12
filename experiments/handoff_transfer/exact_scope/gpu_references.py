"""Measure references on each allocated physical GPU, one worker per GPU."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path('/scratch/aryama.murthy/handoff-transfer-20260911')
STUDY=ROOT.parent/'relayspec-auf-20260911'


def read(path):return json.loads(path.read_text())


def run(module,*arguments):
    subprocess.run([sys.executable,'-u','-m',module,*map(str,arguments)],check=True)


def required_devices(folder):
    """Only benchmark reference cards actually used by the seven retained token-fit cells."""
    cases=sorted(p for p in folder.glob('*/complete.json') if not p.parent.name.startswith('feature-'))
    if len(cases)!=7:
        raise ValueError(f'Expected all seven completed trained cells in {folder}; found {len(cases)}')
    batch=read(folder/'evaluation-batch.json')['request_batch_size']
    devices=set()
    for complete in cases:
        cell=read(complete)
        if cell['status']!='complete' or cell['timing_repetitions']!=3:
            raise ValueError(f'Incomplete evaluation: {complete}')
        for repeat in range(3):
            summary=read(complete.parent/f'matrix-r{repeat}-w0-b{batch}.summary.json')
            uuid=summary['gpu_after'][0]['device_uuid']
            if summary['gpu_before'][0]['device_uuid']!=uuid or not summary['timing_valid']:
                raise ValueError(f'Invalid physical-GPU measurement: {complete}, repeat {repeat}')
            devices.add(uuid)
    return devices


def worker(uuid):
    for family in ('q8','llama','cross'):
        old=ROOT/'exact32e1b8-results'/family
        if uuid not in required_devices(old):
            print(json.dumps(dict(family=family,device_uuid=uuid,status='not_required',
                                  reason='No trained-method measurement used this physical GPU')),flush=True)
            continue
        batch=read(old/'evaluation-batch.json')['request_batch_size']
        target={'q8':ROOT.parent/'transfer-reproduction-20260907/work/models/8b/target',
                'llama':STUDY/'models/llama3-target','cross':STUDY/'models/llama8-source'}[family]
        initial=ROOT/'cross-full4096/initializers-e1' if family=='cross' else ROOT/family
        transfer=read(initial/'transfer.json')
        exports={'ar':None,'original':initial/'normal/export','zip':Path(transfer['base_export'])}
        if family=='cross':os.environ['CROSS_SOURCE_TOKENIZER']=str(STUDY/'models/qwen4-source')
        out=ROOT/'gpu-matched-references'/family/uuid
        for repeat in range(3):
            for label,export in exports.items():
                mode='ar' if label=='ar' else 'matrix'
                stem=f'{mode}-r{repeat}-w0-b{batch}'
                dest=out/label;dest.mkdir(parents=True,exist_ok=True)
                source_summary=old/label/f'{stem}.summary.json'
                # Reuse only measurements already made on this exact card.
                if not (dest/f'{stem}.summary.json').exists() and source_summary.exists():
                    previous=read(source_summary)
                    if previous['gpu_after'][0]['device_uuid']==uuid:
                        shutil.copyfile(old/label/f'{stem}.jsonl',dest/f'{stem}.jsonl')
                        shutil.copyfile(source_summary,dest/f'{stem}.summary.json')
                extra=['--export',export] if export is not None else []
                run('experiments.auf_vllm.family_benchmark','--family',family,'--models',STUDY/'models',
                    '--target-path',target,'--data',old/'evaluation','--out',dest,'--mode',mode,
                    '--count',128,'--cap',2048,'--repeat',repeat,'--request-batch-size',batch,*extra)
                summary=read(dest/f'{stem}.summary.json')
                assert summary['gpu_after'][0]['device_uuid']==uuid
                if label!='ar':
                    run('experiments.auf_vllm.compare_outputs','--ar',out/f'ar/ar-r{repeat}-w0-b{batch}.jsonl',
                        '--method',dest/f'{stem}.jsonl','--out',dest/f'comparison-r{repeat}.json',
                        '--expected-count',128,'--require-exact')
        (out/'complete.json').write_text(json.dumps(dict(status='complete',family=family,device_uuid=uuid,
            repetitions=3,requests=128,cap=2048,batch=batch,job_id=os.environ['SLURM_JOB_ID']))+'\n')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--worker-uuid');args=parser.parse_args()
    if args.worker_uuid:
        worker(args.worker_uuid);return
    allocated=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
    assert len(allocated)==4,'Reference phase needs all four allocated node07 GPUs'
    devices={index.strip():uuid.strip().removeprefix('GPU-') for index,uuid in (
        line.split(',') for line in subprocess.check_output(
            ['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader'],text=True).splitlines())}
    out=ROOT/'gpu-matched-references';out.mkdir(parents=True,exist_ok=True)
    available={devices[d] if d in devices else d.removeprefix('GPU-') for d in allocated}
    for family in ('q8','llama','cross'):
        needed=required_devices(ROOT/'exact32e1b8-results'/family)
        if not needed<=available:
            raise ValueError(f'{family} requires unavailable reference GPUs: {needed-available}')
    children=[]
    for device in allocated:
        uuid=devices[device] if device in devices else device.removeprefix('GPU-')
        assert uuid in devices.values()
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device)
        handle=(out/f'worker-{uuid}-{os.environ["SLURM_JOB_ID"]}.log').open('w')
        process=subprocess.Popen([sys.executable,'-u','-m',__name__ if __name__!='__main__' else
            'experiments.handoff_transfer.exact_scope.gpu_references','--worker-uuid',uuid],
            env=env,stdout=handle,stderr=subprocess.STDOUT)
        children.append((uuid,process,handle))
    failures=[]
    for uuid,process,handle in children:
        code=process.wait();handle.close()
        if code:failures.append((uuid,code))
    if failures:raise RuntimeError(f'Reference workers failed: {failures}')


if __name__=='__main__':main()
