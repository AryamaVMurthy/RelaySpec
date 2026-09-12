"""Overlap the six pending cross adapter measurements with shared references.

This fills the existing evaluator's exact output paths and contracts. It does
not declare the comparison complete: evaluate.py still checks all repetitions
against their corresponding AR references after the shared-reference job ends.
"""
import json
import os
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def main():
    tasks=read(Path(__file__).with_name('cross-tasks.json'))
    index=int(os.environ['SLURM_ARRAY_TASK_ID'])
    assert 1<=index<=6, 'The first adapter remains owned by the reference job'
    task=tasks[index]
    assert task['family']=='cross' and task['kind']!='feature'
    root=Path('/scratch/aryama.murthy/handoff-transfer-20260911')
    study=root.parent/'relayspec-auf-20260911'
    out=root/'exact32e1b8-results/cross'
    # Both baseline adapters must already support the exact full serving batch.
    for label in ('original','zip'):
        checked=read(out/label/'comparison-b128-r0.json')
        assert checked['count']==checked['exact_matches']==checked['finish_matches']==128
        assert checked['request_batch_size']==128
    kind,obj,lr=task['kind'],task['objective'],task['lr']
    fit=root/f'matrix32e1b8/cross/{kind}-{obj}-lr{lr}'
    summary=read(fit/f'modules/steps-512/{kind}/summary.json')
    assert read(fit/f'modules/steps-512/{kind}/verification.json')['status']=='passed'
    assert summary['processed_examples']==4096 and summary['optimizer_steps']==512
    assert summary['anchors_per_example']==32 and summary['objective']==obj
    export=fit/f'exports/steps-512/{kind}'
    case=out/f'{kind}-{obj}-lr{lr}'
    os.environ['CROSS_SOURCE_TOKENIZER']=str(study/'models/qwen4-source')
    for repeat in range(3):
        args=['--family','cross','--models',study/'models','--target-path',study/'models/llama8-source',
              '--data',out/'evaluation','--count',128,'--cap',2048,'--request-batch-size',128,
              '--repeat',repeat,'--mode','matrix','--export',export,'--out',case]
        subprocess.run([sys.executable,'-u','-m','experiments.auf_vllm.family_benchmark',*map(str,args)],check=True)
        # A correctness check while later AR timing repetitions are still running.
        # Final evaluation recomputes comparisons using the matching repetition.
        args=['--ar',out/'ar/ar-r0-w0-b128.jsonl','--method',case/f'matrix-r{repeat}-w0-b128.jsonl',
              '--out',case/f'premeasurement-output-check-r{repeat}.json','--expected-count',128,'--require-exact']
        subprocess.run([sys.executable,'-u','-m','experiments.auf_vllm.compare_outputs',*map(str,args)],check=True)
    (case/'premeasure-complete.json').write_text(json.dumps(dict(
        status='measurements_complete_pending_final_reference_checks',job=os.environ['SLURM_JOB_ID'],
        task=task,requests=128,cap=2048,batch=128,repetitions=3,correctness_reference_repeat=0))+'\n')


if __name__=='__main__':
    main()
