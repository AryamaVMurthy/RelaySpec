"""Paired native versus existing five-map AUF checkpoints; no training."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

from experiments.auf_vllm.compare_outputs import compare
from experiments.handoff_transfer.exact_scope.matched import require_same_device


ROOT=Path('/scratch/aryama.murthy/handoff-transfer-20260911')
STUDY=ROOT.parent/'relayspec-auf-20260911'


def read(path):
    return json.loads(path.read_text())


def main():
    job=os.environ['SLURM_JOB_ID']
    out=ROOT/'native-comparison-e1b8'/job
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    for family in ('q8','cross'):
        target=(ROOT.parent/'transfer-reproduction-20260907/work/models/8b/target'
                if family=='q8' else STUDY/'models/llama8-source')
        native=(ROOT.parent/'transfer-reproduction-20260907/work/models/8b/draft'
                if family=='q8' else STUDY/'models/llama8-draft')
        export=ROOT/'matrix32e1b8'/family/'five_maps-auf-lr0.0001/exports/steps-512/five_maps'
        original=ROOT/'exact32e1b8-results'/family
        expected_hash=read(original/'five_maps-auf-lr0.0001/matrix-r0-w0-b128.summary.json')['contract']['export_sha256']
        if family=='cross':os.environ['CROSS_SOURCE_TOKENIZER']=str(STUDY/'models/qwen4-source')
        for repeat in range(3):
            # Alternate order. Each subprocess owns the GPU exclusively and
            # performs the standard warmup/compilation checks before timing.
            order=('native','matrix') if repeat%2==0 else ('matrix','native')
            paths={}
            for mode in order:
                dest=out/family/mode
                extra=['--native-draft',native] if mode=='native' else ['--export',export]
                command=[sys.executable,'-u','-m','experiments.auf_vllm.family_benchmark',
                    '--family',family,'--models',STUDY/'models','--target-path',target,
                    '--data',original/'evaluation','--out',dest,'--mode',mode,
                    '--count',128,'--cap',2048,'--repeat',repeat,'--request-batch-size',128,*extra]
                subprocess.run(list(map(str,command)),check=True)
                paths[mode]=dest/f'{mode}-r{repeat}-w0-b128.jsonl'
            summaries={mode:read(path.with_suffix('.summary.json')) for mode,path in paths.items()}
            uuid=require_same_device(summaries['matrix'],summaries['native'])
            if summaries['matrix']['contract']['export_sha256']!=expected_hash:
                raise ValueError('Native comparison used a different AUF checkpoint')
            # Compare sequence identity to the existing AR output independently
            # of timing. Do not borrow another physical card's AR speed.
            checks={mode:compare([original/f'ar/ar-r{repeat}-w0-b128.jsonl'],[path])
                    for mode,path in paths.items()}
            for mode,checked in checks.items():
                if checked['count']!=128 or checked['exact_matches']!=128 or checked['finish_matches']!=128:
                    (out/f'{family}-{mode}-r{repeat}-disagreement.json').write_text(json.dumps(checked,indent=2)+'\n')
                    raise ValueError(f'Native comparison disagrees with AR: {family}/{mode}')
            paired=compare([paths['native']],[paths['matrix']])
            results.append(dict(family=family,repeat=repeat,device_uuid=uuid,
                native_tps=paired['ar_tps'],auf_tps=paired['method_tps'],
                auf_over_native=paired['throughput_ratio'],requests=128,cap=2048,batch=128,
                exact_native=paired['exact_matches'],exact_ar=checks['matrix']['exact_matches'],
                native_block=read(native/'config.json')['block_size'],
                auf_block=read(export/'config.json')['block_size'],
                native_sha256=summaries['native']['contract']['native_draft_sha256'],
                auf_sha256=expected_hash,job_id=job))
            (out/'progress.json').write_text(json.dumps(results,indent=2)+'\n')
            print(json.dumps(results[-1]),flush=True)
    report=dict(status='complete',scope='Same physical GPU, fixed batch128,128 development requests,cap2048; three timing repetitions of existing checkpoints. Native Llama block10 versus reused Qwen block16; no training or block-size sweep.',rows=results,
        summary=[dict(family=family,
            native_tps=statistics.mean(r['native_tps'] for r in results if r['family']==family),
            auf_tps=statistics.mean(r['auf_tps'] for r in results if r['family']==family),
            auf_over_native=statistics.mean(r['auf_over_native'] for r in results if r['family']==family))
            for family in ('q8','cross')])
    (out/'complete.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
