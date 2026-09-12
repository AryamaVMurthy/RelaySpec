"""Evaluate only requested exports; identical AR requests, caps and batch sizes."""
import json, os, subprocess, sys
from pathlib import Path

def run(module,*args):
    subprocess.run([sys.executable,'-u','-m',module,*map(str,args)],check=True)

family=os.environ['FAMILY'];kind=os.environ['KIND'];objective=os.environ['MATRIX_OBJECTIVE'];lr=os.environ['MATRIX_LR']
root=Path('/scratch/aryama.murthy/handoff-transfer-20260911')
study=Path('/scratch/aryama.murthy/relayspec-auf-20260911')
target={'q8':Path('/scratch/aryama.murthy/transfer-reproduction-20260907/work/models/8b/target'),'llama':study/'models/llama3-target','cross':study/'models/llama8-source'}[family]
if family=='cross':os.environ['CROSS_SOURCE_TOKENIZER']=str(study/'models/qwen4-source')
out=root/'exact32-results'/family;data=out/'evaluation';out.mkdir(parents=True,exist_ok=True)
run('experiments.auf_vllm.prepare_family_benchmark','--family',family,'--models',study/'models','--target-path',target,'--manifest',study/'manifests/dev.jsonl','--out',data)
if kind=='feature':
    fit=root/'feature-objectives'/family/objective
    summary=json.loads((fit/'summary.json').read_text());assert summary['status']=='complete'
    export=fit/'export'
else:
    fit=root/'matrix32'/family/f'{kind}-{objective}-lr{lr}'
    verification=json.loads((fit/f'modules/steps-2000/{kind}/verification.json').read_text());assert verification['status']=='passed'
    contract=json.loads((fit/f'model-contract-{kind}.json').read_text());assert contract['num_anchors']==32
    summary=json.loads((fit/f'modules/steps-2000/{kind}/summary.json').read_text());assert summary['processed_examples']==16000 and summary['anchors_per_example']==32
    export=fit/f'exports/steps-2000/{kind}'
initial_root=root/'cross-full4096/initializers' if family=='cross' else root/family
transfer=json.loads((initial_root/'transfer.json').read_text())
reference_exports={'original':initial_root/'normal/export','zip':Path(transfer['base_export'])}
case=out/f'{kind}-{objective}-lr{lr}';case.mkdir(exist_ok=True)
common=['--family',family,'--models',study/'models','--target-path',target,'--data',data,'--count',128,'--cap',2048]
for batch in (4,1):
    suffix=f'-b{batch}' if batch>1 else ''
    for repeat in range(3):
        tail=['--request-batch-size',batch,'--repeat',repeat]
        run('experiments.auf_vllm.family_benchmark',*common,*tail,'--mode','ar','--out',out/'ar')
        for label,reference_export in reference_exports.items():
            refcase=out/label
            run('experiments.auf_vllm.family_benchmark',*common,*tail,'--mode','matrix','--export',reference_export,'--out',refcase)
            run('experiments.auf_vllm.compare_outputs','--ar',out/f'ar/ar-r{repeat}-w0{suffix}.jsonl','--method',refcase/f'matrix-r{repeat}-w0{suffix}.jsonl','--out',refcase/f'comparison-b{batch}-r{repeat}.json','--expected-count',128,'--require-exact')
        run('experiments.auf_vllm.family_benchmark',*common,*tail,'--mode','matrix','--export',export,'--out',case)
        run('experiments.auf_vllm.compare_outputs','--ar',out/f'ar/ar-r{repeat}-w0{suffix}.jsonl','--method',case/f'matrix-r{repeat}-w0{suffix}.jsonl','--out',case/f'comparison-b{batch}-r{repeat}.json','--expected-count',128,'--require-exact')
(case/'complete.json').write_text(json.dumps({'status':'complete','family':family,'kind':kind,'objective':objective,'training_summary':summary,'requests':128,'cap':2048,'batches':[4,1],'timing_repetitions':3,'export':str(export)},indent=2)+'\n')
